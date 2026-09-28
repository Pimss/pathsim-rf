"""
netlist_to_statespace.py

Turns a small SPICE-like netlist of R, L, C, K (mutual inductance / coupling
coefficient), V (independent voltage source) and I (independent current
source) elements into an explicit LTI state-space model

    xdot = A x + B u
    y    = C x + D u

via Modified Nodal Analysis (MNA) + elimination of the algebraic
(non-storage) unknowns. This is the standard "state-variable method" used
to derive circuit ODEs by hand, automated in code so it scales to
messy topologies and handles coupled inductors correctly.

Only LINEAR elements are supported (no diodes/switches/etc - PathSim itself
handles those fine via its algebraic-loop solver and Function blocks, see
the accompanying explanation).
"""

from __future__ import annotations
import sympy as sp
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import logging
import re
from typing import Literal
import warnings

from pathsim.blocks.lti import StateSpace


# --------------------------------------------------------------------------
# Netlist parsing
# --------------------------------------------------------------------------

@dataclass
class Element:
    kind: str      # 'R','L','C','V','I','K'
    name: str
    n1: str = None
    n2: str = None
    value: float = None
    # for K elements:
    l1: str = None
    l2: str = None


SI_MULTIPLIERS = {
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "m": 1e-3,
    "": 1.0,
    "k": 1e3,
    "K": 1e3,
    "meg": 1e6,
    "Meg": 1e6,
    "M": 1e6,
    "g": 1e9,
    "G": 1e9,
    "t": 1e12,
    "T": 1e12,
}

_PARAM_PATTERN = re.compile(r"^\.param\s+([A-Za-z_]\w*)\s*=\s*(.+)$")
_VALUE_PATTERN = re.compile(
    r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)([A-Za-zµ]*)\s*$"
)
_VALID_KINDS = {"R", "L", "C", "V", "I", "K"}
_IGNORED_LINE_STARTS = (".", "*", '"')
ReductionMode = Literal["symbolic", "fast"]
_MUMPS_AVAILABLE = importlib.util.find_spec("mumps") is not None
_LOGGER = logging.getLogger("pathsim.netlist_to_statespace")


def _normalize_node(node: str) -> str:
    """Normalize ground aliases to canonical node '0'."""
    return "0" if node.strip().lower() in {"0", "gnd"} else node


def _get_si_multiplier(suffix: str) -> float:
    """Return numeric multiplier for an SI prefix/suffix string."""
    if suffix in SI_MULTIPLIERS:
        return SI_MULTIPLIERS[suffix]
    if suffix.lower() in SI_MULTIPLIERS:
        return SI_MULTIPLIERS[suffix.lower()]

    candidates = sorted(SI_MULTIPLIERS.keys(), key=len, reverse=True)
    for key in candidates:
        if not key:
            continue
        if suffix.startswith(key):
            return SI_MULTIPLIERS[key]
        low_key = key.lower()
        if suffix.lower().startswith(low_key):
            return SI_MULTIPLIERS[low_key]
    raise ValueError(f"Unknown SI prefix: '{suffix}'")


def parse_value_with_units(value_str: str, params: dict[str, str] | None = None) -> float:
    """
    Parse SPICE-like numeric tokens with SI prefixes and optional unit tails.

    Supported forms include:
    - plain/scientific floats: ``1``, ``100.e-3``, ``2.2e6``
    - SI-prefixed tokens: ``4u``, ``10k``, ``3Meg``, ``2.2kOhm``, ``1mH``
    - parameter references: ``{RMAIN}`` when ``params`` is provided
    """
    if value_str is None:
        raise ValueError("Missing value")

    value = value_str.strip()
    try:
        return float(value)
    except ValueError:
        pass

    if value.startswith("{") and value.endswith("}"):
        if params is None:
            raise ValueError(f"Unknown parameter reference: '{value}'")
        name = value[1:-1].strip()
        if name not in params:
            raise ValueError(f"Unknown parameter: '{name}'")
        return parse_value_with_units(params[name], params)

    match = _VALUE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid value format: '{value_str}'")

    number, suffix = match.groups()
    return float(number) * _get_si_multiplier(suffix)


def _parse_optional_source_value(raw: str, params: dict[str, str]) -> float | None:
    """Parse source literal values, returning None for waveform expressions."""
    try:
        return parse_value_with_units(raw, params)
    except ValueError:
        # Many source statements use waveforms (PWL, SIN, EXP, etc.); those
        # are runtime waveforms and do not affect linearization here.
        return None


def _parse_behavioral_source(parts: list[str], line_number: int, line: str) -> Element:
    """
    Parse LTspice behavioral source shorthand:
      Bx n+ n- I=<expr>  -> modeled as current source input placeholder
      Bx n+ n- V=<expr>  -> modeled as voltage source input placeholder
    """
    if len(parts) < 4:
        raise ValueError(f"Malformed behavioral source at line {line_number}: '{line}'")

    name = parts[0]
    n1, n2 = _normalize_node(parts[1]), _normalize_node(parts[2])
    expr = " ".join(parts[3:]).strip()
    expr_upper = expr.upper()
    if expr_upper.startswith("I="):
        return Element(kind="I", name=name, n1=n1, n2=n2, value=None)
    if expr_upper.startswith("V="):
        return Element(kind="V", name=name, n1=n1, n2=n2, value=None)
    raise ValueError(
        f"Unsupported behavioral source expression at line {line_number}: '{line}'"
    )


def parse_netlist(text: str) -> list[Element]:
    """
    Parse netlist text into linear element records.

    One element per line, whitespace separated; '#' or ';' starts a comment.
      R<name> n1 n2 value
      L<name> n1 n2 value
      C<name> n1 n2 value
      V<name> n+ n- value      (value is a placeholder; real waveform comes
                                 from the PathSim Source block at sim time)
      I<name> n+ n- value      (same)
      B<name> n+ n- I=<expr>   (behavioral current source placeholder)
      B<name> n+ n- V=<expr>   (behavioral voltage source placeholder)
      K<name> Lname1 Lname2 k  (coupling coefficient, -1<=k<=1)
    Node '0' (or 'gnd') is ground.
    Deck/meta lines beginning with '.', '*', or a quoted header are ignored.
    """
    params: dict[str, str] = {}
    raw_element_lines: list[tuple[int, str]] = []

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#")[0].split(";")[0].strip()
        if not line:
            continue
        match = _PARAM_PATTERN.fullmatch(line)
        if match is not None:
            params[match.group(1)] = match.group(2).strip()
            continue
        raw_element_lines.append((line_number, line))

    elements: list[Element] = []
    for line_number, line in raw_element_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0].startswith(_IGNORED_LINE_STARTS):
            continue

        name = parts[0]
        kind = name[0].upper()
        if kind == "B":
            elements.append(_parse_behavioral_source(parts, line_number, line))
            continue

        if kind not in _VALID_KINDS:
            raise ValueError(
                f"Unsupported element '{name}' at line {line_number}: '{line}'"
            )

        if kind == 'K':
            if len(parts) < 4:
                raise ValueError(f"Malformed K element at line {line_number}: '{line}'")
            elements.append(Element(kind='K', name=name, l1=parts[1], l2=parts[2],
                                     value=parse_value_with_units(parts[3], params)))
        else:
            if len(parts) < 3:
                raise ValueError(f"Malformed element at line {line_number}: '{line}'")
            n1, n2 = _normalize_node(parts[1]), _normalize_node(parts[2])
            val_token = " ".join(parts[3:]) if len(parts) > 3 else None
            if kind in {"R", "L", "C"}:
                if val_token is None:
                    raise ValueError(f"Missing value for element '{name}' at line {line_number}")
                val = parse_value_with_units(val_token, params)
            else:
                val = _parse_optional_source_value(val_token, params) if val_token else None
            elements.append(Element(kind=kind, name=name, n1=n1, n2=n2, value=val))
    return elements


def parse_netlist_file(path: str | Path) -> list[Element]:
    """Load and parse a netlist file."""
    with open(path, "r", encoding="utf-8") as handle:
        return parse_netlist(handle.read())


GND = {'0', 'gnd', 'GND'}


# --------------------------------------------------------------------------
# MNA + symbolic DAE -> explicit state space reduction
# --------------------------------------------------------------------------

class CircuitModel:
    """
    Build an explicit state-space model from a linear circuit netlist.

    The circuit is first stamped as the modified nodal analysis (MNA) system

    ``E * dz/dt + G * z = B * u``,

    where ``z`` contains node voltages, inductor currents, and ideal voltage
    source currents. The MNA unknowns are partitioned into differential states
    ``x_d`` and algebraic unknowns ``x_a``. Eliminating ``x_a`` gives

    ``x_a = Phi * x_d + Psi * u``

    and the final explicit model

    ``dx_d/dt = A * x_d + B_ss * u``.

    Both reduction modes produce the same matrices and state ordering. They
    differ only in the arithmetic and linear solver used during elimination.

    Parameters
    ----------
    elements:
        Output of :func:`parse_netlist` / :func:`parse_netlist_file`.
    reduction_mode:
        Method used to eliminate the algebraic MNA variables:

        ``"symbolic"``
            Stamp and reduce with SymPy matrices. Matrix inverses and products
            retain symbolic arithmetic until the final conversion to floating
            point. This is the default and is useful for small circuits,
            reference results, and diagnosing rank deficiencies. Its runtime
            and memory use can grow quickly on large or densely coupled
            circuits.

        ``"fast"``
            Stamp directly into floating-point arrays. Eliminate the algebraic
            subsystem with a sparse MUMPS factorization, then solve the
            differential mass system numerically. This mode is intended for
            large RLC/K netlists and repeated output construction. A singular
            algebraic solve is retried with progressively larger diagonal
            regularization and emits :class:`RuntimeWarning` when regularization
            is used. When ``python-mumps`` is unavailable, construction falls
            back to symbolic reduction and logs a warning through PathSim.

        The mode affects model construction only

    Attributes
    ----------
    A:
        State matrix of the reduced explicit system.
    B_ss:
        Input matrix of the reduced explicit system.
    Phi:
        Map from differential states to eliminated algebraic variables.
    Psi:
        Map from inputs to eliminated algebraic variables.
    state_labels:
        Labels matching the row and column ordering of the state matrices.
    """

    def __init__(
        self,
        elements: list[Element],
        reduction_mode: ReductionMode = "symbolic",
    ):
        if reduction_mode not in {"symbolic", "fast"}:
            raise ValueError(
                f"Unsupported reduction_mode '{reduction_mode}'. "
                "Expected 'symbolic' or 'fast'."
            )
        if reduction_mode == "fast" and not _MUMPS_AVAILABLE:
            _LOGGER.warning(
                "Fast netlist reduction requires python-mumps; "
                "falling back to symbolic reduction."
            )
            reduction_mode = "symbolic"
        self.reduction_mode = reduction_mode
        self.elements = elements

        self.R = [e for e in elements if e.kind == 'R']
        self.L = [e for e in elements if e.kind == 'L']
        self.C = [e for e in elements if e.kind == 'C']
        self.V = [e for e in elements if e.kind == 'V']
        self.I = [e for e in elements if e.kind == 'I']
        self.K = [e for e in elements if e.kind == 'K']
        self.dipoles_by_name = {
            e.name: e for e in elements if e.kind in {"R", "L", "C", "V", "I"}
        }

        # ---- nodes ----
        nodes = []
        for e in elements:
            if e.kind == 'K':
                continue
            for n in (e.n1, e.n2):
                if n not in GND and n not in nodes:
                    nodes.append(n)
        self.nodes = nodes                      # ordered list of non-ground node names
        self.node_idx = {n: i for i, n in enumerate(nodes)}
        self.n_nodes = len(nodes)

        self.L_names = [e.name for e in self.L]
        self.L_idx = {name: i for i, name in enumerate(self.L_names)}
        self.n_L = len(self.L)

        self.V_names = [e.name for e in self.V]
        self.V_idx = {name: i for i, name in enumerate(self.V_names)}
        self.n_V = len(self.V)

        # unknown ordering: [node voltages] + [inductor currents] + [V-source currents]
        self.n_total = self.n_nodes + self.n_L + self.n_V

        def col_node(n):
            return None if n in GND else self.node_idx[n]

        def col_iL(name):
            return self.n_nodes + self.L_idx[name]

        def col_iV(name):
            return self.n_nodes + self.n_L + self.V_idx[name]

        self._col_node, self._col_iL, self._col_iV = col_node, col_iL, col_iV
        numeric = self.reduction_mode == "fast"

        # ---- inductance matrix (with mutual terms) ----
        Lmat = (
            np.zeros((self.n_L, self.n_L), dtype=float)
            if numeric
            else sp.zeros(self.n_L, self.n_L)
        )
        l_values = {e.name: e.value for e in self.L}
        for e in self.L:
            i = self.L_idx[e.name]
            Lmat[i, i] = float(e.value) if numeric else sp.nsimplify(e.value)
        for k in self.K:
            i, j = self.L_idx[k.l1], self.L_idx[k.l2]
            Li = l_values[k.l1]
            Lj = l_values[k.l2]
            if numeric:
                M = float(k.value) * np.sqrt(float(Li) * float(Lj))
            else:
                M = sp.nsimplify(k.value) * sp.sqrt(sp.nsimplify(Li) * sp.nsimplify(Lj))
            Lmat[i, j] += M
            Lmat[j, i] += M
        self.Lmat = Lmat

        # ---- inputs: one column per V source, then one column per I source ----
        self.input_names = self.V_names + [e.name for e in self.I]
        self.n_u = len(self.input_names)

        # ---- build E (dynamic) and G (algebraic) matrices, and B (input map) ----
        n = self.n_total
        if numeric:
            E = np.zeros((n, n), dtype=float)
            G = np.zeros((n, n), dtype=float)
            B = np.zeros((n, self.n_u), dtype=float)
        else:
            E = sp.zeros(n, n)
            G = sp.zeros(n, n)
            B = sp.zeros(n, self.n_u)

        def stamp_G(row, col, val):
            if row is not None and col is not None:
                G[row, col] += val

        # Resistors: contribute to node KCL rows only
        for e in self.R:
            a, b = col_node(e.n1), col_node(e.n2)
            g = (1.0 / float(e.value)) if numeric else (1 / sp.nsimplify(e.value))
            stamp_G(a, a, g); stamp_G(b, b, g)
            stamp_G(a, b, -g); stamp_G(b, a, -g)

        # Capacitors: contribute dv/dt terms to node KCL rows (the E matrix)
        for e in self.C:
            a, b = col_node(e.n1), col_node(e.n2)
            c = float(e.value) if numeric else sp.nsimplify(e.value)
            if a is not None: E[a, a] += c
            if b is not None: E[b, b] += c
            if a is not None and b is not None:
                E[a, b] -= c
                E[b, a] -= c

        # Inductors: KCL stamp (current unknown enters/leaves nodes) +
        # dedicated branch row  v_na - v_nb - sum_j Lij * d(iLj)/dt = 0
        for e in self.L:
            a, b = col_node(e.n1), col_node(e.n2)
            iL_col = col_iL(e.name)
            row = iL_col  # branch row shares index with its current unknown
            stamp_G(a, iL_col, 1)
            stamp_G(b, iL_col, -1)
            if a is not None: G[row, a] += 1
            if b is not None: G[row, b] -= 1
            # branch eqn: v_na - v_nb - L*d(iL)/dt - sum_j M_ij*d(iLj)/dt = 0
            # => E[row, iLj] = -Lij  (note the minus sign!)
            i = self.L_idx[e.name]
            for j, name_j in enumerate(self.L_names):
                Lij = self.Lmat[i, j]
                if Lij != 0:
                    E[row, col_iL(name_j)] -= Lij

        # Voltage sources: KCL stamp + branch row v_na - v_nb = u(t)
        for e in self.V:
            a, b = col_node(e.n1), col_node(e.n2)
            iV_col = col_iV(e.name)
            row = iV_col
            stamp_G(a, iV_col, 1)
            stamp_G(b, iV_col, -1)
            if a is not None: G[row, a] += 1
            if b is not None: G[row, b] -= 1
            u_col = self.input_names.index(e.name)
            B[row, u_col] = 1

        # Current sources: pure RHS injection into node KCL rows.
        # Convention: positive I flows from n1 -> n2 *through the source*,
        # i.e. it delivers current INTO n2 and draws it OUT of n1 from the
        # external circuit's point of view.
        for e in self.I:
            a, b = col_node(e.n1), col_node(e.n2)
            u_col = self.input_names.index(e.name)
            if a is not None: B[a, u_col] -= 1
            if b is not None: B[b, u_col] += 1

        self.E, self.G, self.B = E, G, B

        # ---- differential / algebraic partition ----
        diff_rows = [r for r in range(n) if any(E[r, c] != 0 for c in range(n))]
        alg_rows = [r for r in range(n) if r not in diff_rows]
        self.diff_rows, self.alg_rows = diff_rows, alg_rows

        # sanity: the state columns should be exactly node-voltages that own a
        # nonzero E column plus all inductor currents; algebraic columns = rest
        diff_cols = sorted(set(c for r in diff_rows for c in range(n) if E[r, c] != 0))
        alg_cols = [c for c in range(n) if c not in diff_cols]
        if len(diff_cols) != len(diff_rows) or len(alg_cols) != len(alg_rows):
            raise ValueError(
                "Circuit is degenerate for this reduction (e.g. an all-capacitor "
                "loop, an all-inductor cutset, or a floating node). "
                f"diff_rows={len(diff_rows)} diff_cols={len(diff_cols)} "
                f"alg_rows={len(alg_rows)} alg_cols={len(alg_cols)}"
            )
        self.diff_cols, self.alg_cols = diff_cols, alg_cols

        self._reduce()
        self._selected_output_rows: list[tuple[np.ndarray, np.ndarray]] = []
        self.output_labels: list[str] = []

    def _reduce(self):
        """
        Reduce the stamped MNA differential-algebraic system.

        This dispatcher selects the arithmetic backend requested by
        :attr:`reduction_mode`. Both implementations populate ``A``, ``B_ss``,
        ``Phi``, and ``Psi`` and preserve the same differential-state ordering.
        State labels are assigned only after a successful reduction.
        """
        if self.reduction_mode == "fast":
            self._reduce_fast()
        else:
            self._reduce_symbolic()
        self._set_state_labels()

    def _set_state_labels(self):
        """Set labels for the differential state vector in `self.diff_cols` order."""
        dc = self.diff_cols
        labels = []
        for c in dc:
            if c < self.n_nodes:
                labels.append(f"v_{self.nodes[c]}")
            else:
                labels.append(f"i_{self.L_names[c - self.n_nodes]}")
        self.state_labels = labels
        self.state_label_idx = {label: i for i, label in enumerate(labels)}

    def _raise_algebraic_singular(self):
        raise ValueError(
            "Algebraic subsystem is singular. Usually means a loop made "
            "purely of ideal voltage sources (and/or 0-ohm shorts), or a "
            "node with no DC path to ground."
        )

    def _raise_state_singular(self):
        raise ValueError(
            "State/mass matrix is singular: this circuit's capacitor "
            "voltages (or inductor currents) aren't independent, so the "
            "naive 'one state per capacitor-touched node' selection "
            "over-counted states. Classic cause: a capacitor whose *both* "
            "terminals only reach the rest of the circuit through that "
            "same capacitor (no other cap ties either node down "
            "independently) -- e.g. 'R1 in a / C1 a b / R2 b 0' with "
            "nothing else at a or b. Only one of v_a, v_b is really an "
            "independent state there; the other is pinned by KCL. This "
            "reduction doesn't do full tree/cotree state selection, so it "
            "can't detect that automatically yet. Workarounds: (1) add a "
            "TINY STRAY CAPACITANCE FROM ONE OF THE FLOATING NODES TO "
            "GROUND (not a resistor -- the degeneracy lives in the "
            "capacitor/mass matrix, a parallel resistor doesn't touch it "
            "at all). e.g. 'Cstray b 0 1e-15' -- this gives that node's "
            "own row independent rank; it adds one extra, extremely fast "
            "eigenvalue (a numerical artifact, orders of magnitude faster "
            "than your real dynamics) alongside the correct physical "
            "pole(s), or (2) hand-pick the true independent capacitor "
            "voltage as the state and eliminate the redundant node "
            "yourself before building the netlist."
        )

    def _solve_with_mumps(self, mat: np.ndarray, rhs: np.ndarray, context: str) -> np.ndarray:
        """
        Solve one or more right-hand sides with one MUMPS factorization.

        Parameters
        ----------
        mat:
            Square coefficient matrix.
        rhs:
            Vector or matrix of right-hand sides. Each matrix column is solved
            independently while reusing the factorization of ``mat``.
        context:
            Subsystem name included in errors and regularization warnings.

        Returns
        -------
        numpy.ndarray
            Solution with the same one- or two-dimensional convention as
            ``rhs``.

        Notes
        -----
        The unmodified matrix is attempted first. If factorization or solution
        fails, the method retries ``mat + eps * I`` for increasing ``eps``.
        Regularization permits reduction of numerically singular algebraic
        systems, but it perturbs their constraints; every successful
        regularized solve therefore emits :class:`RuntimeWarning`.

        MUMPS deliberately remains in its default unsymmetric mode. General
        MNA systems may contain voltage-source saddle-point blocks, so they
        are not positive definite, and the reduced transient matrix is not
        generally symmetric. Selecting MUMPS ``sym=1`` (SPD) or passing
        ``symmetric=True`` would therefore impose an invalid matrix property.
        """
        try:
            import mumps
            import scipy.sparse as sps
        except ImportError as exc:
            raise ImportError(
                "Fast reduction mode requires `python-mumps` (module `mumps`) "
                "and scipy.sparse to be available."
            ) from exc

        rhs_2d = rhs if rhs.ndim == 2 else rhs.reshape((-1, 1))
        if mat.shape[0] != mat.shape[1]:
            raise ValueError(f"{context} matrix must be square, got {mat.shape}")
        if rhs_2d.shape[0] != mat.shape[0]:
            raise ValueError(
                f"{context} rhs has incompatible shape {rhs_2d.shape} for matrix {mat.shape}"
            )

        last_error: Exception | None = None
        for eps in (0.0, 1e-15, 1e-12, 1e-9, 1e-6):
            try:
                if eps > 0.0:
                    mat_reg = mat + np.eye(mat.shape[0], dtype=float) * eps
                else:
                    mat_reg = mat

                ctx = mumps.Context()
                ctx.set_matrix(sps.csc_matrix(mat_reg))
                ctx.factor()

                cols = []
                for j in range(rhs_2d.shape[1]):
                    bj = np.array(rhs_2d[:, j], dtype=float, order="F")
                    xj = np.array(ctx.solve(bj), dtype=float)
                    cols.append(xj)
                x = np.column_stack(cols)

                if eps > 0.0:
                    warnings.warn(
                        f"Fast reduction regularized singular {context} with eps={eps:.1e}",
                        RuntimeWarning,
                    )
                return x if rhs.ndim == 2 else x[:, 0]
            except (mumps.MUMPSError, ValueError, RuntimeError, TypeError) as exc:
                last_error = exc
                continue

        raise ValueError(
            f"Failed to solve {context} even after regularization attempts."
        ) from last_error

    @staticmethod
    def _safe_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiply without relying on NumPy BLAS-backed matmul.
        This avoids environment-specific native linear algebra crashes.
        """
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Incompatible shapes for matmul: {a.shape} and {b.shape}")
        out = np.zeros((a.shape[0], b.shape[1]), dtype=float)
        for i in range(a.shape[0]):
            for k in range(a.shape[1]):
                aik = a[i, k]
                if aik != 0.0:
                    out[i, :] += aik * b[k, :]
        return out

    def _solve_with_sympy_numeric(self, mat: np.ndarray, rhs: np.ndarray, context: str) -> np.ndarray:
        """Solve linear system using numeric SymPy LU; supports regularization ladder."""
        rhs_2d = rhs if rhs.ndim == 2 else rhs.reshape((-1, 1))
        m = sp.Matrix(mat)
        r = sp.Matrix(rhs_2d)

        try:
            sol = m.LUsolve(r)
            x = np.array(sol, dtype=float)
            return x if rhs.ndim == 2 else x[:, 0]
        except (ValueError, sp.matrices.exceptions.NonInvertibleMatrixError):
            pass

        for eps in (1e-15, 1e-12, 1e-9, 1e-6):
            try:
                reg = m + sp.eye(m.rows) * eps
                sol = reg.LUsolve(r)
                warnings.warn(
                    f"Fast reduction regularized singular {context} with eps={eps:.1e}",
                    RuntimeWarning,
                )
                x = np.array(sol, dtype=float)
                return x if rhs.ndim == 2 else x[:, 0]
            except (ValueError, sp.matrices.exceptions.NonInvertibleMatrixError):
                continue

        raise ValueError(f"Failed to solve {context} with numeric LU.")

    def _reduce_symbolic(self):
        """
        Eliminate algebraic variables using symbolic SymPy arithmetic.

        With differential and algebraic variables denoted by ``x_d`` and
        ``x_a``, the algebraic MNA rows are

        ``G_alg_d*x_d + G_alg_a*x_a = B_alg*u``.

        The method computes

        ``Phi = -inv(G_alg_a)*G_alg_d`` and
        ``Psi = inv(G_alg_a)*B_alg``,

        so that ``x_a = Phi*x_d + Psi*u``. Substitution into the differential
        rows yields

        ``A = -inv(E_dd)*(G_diff_d + G_diff_a*Phi)`` and
        ``B_ss = inv(E_dd)*(B_diff - G_diff_a*Psi)``.

        SymPy retains symbolic values through elimination and converts only the
        final state matrices to ``float`` arrays. This makes the mode useful as
        a correctness reference for small circuits, but explicit symbolic
        inverses can become expensive in time and memory as the netlist grows.
        Singular algebraic and state matrices are reported without numerical
        regularization.
        """
        E, G, B = self.E, self.G, self.B
        dr, ar = self.diff_rows, self.alg_rows
        dc, ac = self.diff_cols, self.alg_cols

        E_dd = E[dr, dc]                       # square, the "mass"/coupling matrix
        G_alg_d = G[ar, dc]
        G_alg_a = G[ar, ac]
        G_diff_d = G[dr, dc]
        G_diff_a = G[dr, ac]
        B_alg = B[ar, :]
        B_diff = B[dr, :]

        try:
            G_alg_a_inv = G_alg_a.inv()
        except sp.matrices.exceptions.NonInvertibleMatrixError:
            self._raise_algebraic_singular()
        Phi = -G_alg_a_inv * G_alg_d           # x_a = Phi * x_d + Psi * u
        Psi = G_alg_a_inv * B_alg

        try:
            E_dd_inv = E_dd.inv()
        except sp.matrices.exceptions.NonInvertibleMatrixError:
            self._raise_state_singular()
        A = -E_dd_inv * (G_diff_d + G_diff_a * Phi)
        Bmat = E_dd_inv * (B_diff - G_diff_a * Psi)

        self.A = np.array(A.evalf(), dtype=float)
        self.B_ss = np.array(Bmat.evalf(), dtype=float)
        self.Phi = Phi
        self.Psi = Psi

    def _reduce_fast(self):
        """
        Eliminate algebraic variables using floating-point linear solves.

        This method implements the same block elimination and equations as
        :meth:`_reduce_symbolic`, but avoids symbolic matrix inversion:

        1. Extract ``E_dd`` and the differential/algebraic blocks of ``G`` and
           ``B`` as ``float`` arrays.
        2. Factor ``G_alg_a`` with MUMPS and solve for ``Phi`` and ``Psi``.
        3. Form the Schur-complement terms involving ``G_diff_a``.
        4. Solve the differential mass system ``E_dd`` for ``A`` and ``B_ss``.

        The algebraic MUMPS factorization is reused across right-hand sides.
        A small diagonal regularization ladder is available for numerically
        singular systems and is always announced with a warning. The state
        mass solve uses numeric SymPy LU in this implementation to avoid
        environment-specific native dense-linear-algebra failures.

        ``"fast"`` changes only construction cost and numerical precision; it
        does not simplify the circuit, discard states, or alter output
        equations. It is generally preferred for large transmission-line and
        densely coupled RLC/K models.
        """
        E, G, B = self.E, self.G, self.B
        dr, ar = self.diff_rows, self.alg_rows
        dc, ac = self.diff_cols, self.alg_cols

        if isinstance(E, np.ndarray):
            E_dd = E[np.ix_(dr, dc)]
            G_alg_d = G[np.ix_(ar, dc)]
            G_alg_a = G[np.ix_(ar, ac)]
            G_diff_d = G[np.ix_(dr, dc)]
            G_diff_a = G[np.ix_(dr, ac)]
            B_alg = B[np.ix_(ar, range(self.n_u))]
            B_diff = B[np.ix_(dr, range(self.n_u))]
        else:
            E_dd = np.array(E[dr, dc].evalf(), dtype=float)
            G_alg_d = np.array(G[ar, dc].evalf(), dtype=float)
            G_alg_a = np.array(G[ar, ac].evalf(), dtype=float)
            G_diff_d = np.array(G[dr, dc].evalf(), dtype=float)
            G_diff_a = np.array(G[dr, ac].evalf(), dtype=float)
            B_alg = np.array(B[ar, :].evalf(), dtype=float)
            B_diff = np.array(B[dr, :].evalf(), dtype=float)

        try:
            Phi = -self._solve_with_mumps(G_alg_a, G_alg_d, "algebraic subsystem")
            Psi = self._solve_with_mumps(G_alg_a, B_alg, "algebraic subsystem")
        except (ImportError, ValueError):
            self._raise_algebraic_singular()

        try:
            gphi = self._safe_matmul(G_diff_a, Phi)
            gpsi = self._safe_matmul(G_diff_a, Psi)
            A = -self._solve_with_sympy_numeric(E_dd, G_diff_d + gphi, "state/mass subsystem")
            Bmat = self._solve_with_sympy_numeric(E_dd, B_diff - gpsi, "state/mass subsystem")
        except (ImportError, ValueError):
            self._raise_state_singular()

        self.A = A
        self.B_ss = Bmat
        self.Phi = Phi
        self.Psi = Psi

    def _build_output_rows(
        self,
        coefficient_rows: list[dict[str, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project raw MNA output expressions to state-space rows.

        Each coefficient mapping uses keys of the form ``node:name``,
        ``iL:name``, or ``iV:name``. Algebraic unknowns are substituted through
        ``Phi`` and ``Psi`` so each result satisfies ``y = C*x + D*u``.
        """
        if not coefficient_rows:
            return (
                np.empty((0, len(self.state_labels)), dtype=float),
                np.empty((0, self.n_u), dtype=float),
            )

        vec = np.zeros((len(coefficient_rows), self.n_total), dtype=float)
        for row, coeffs in enumerate(coefficient_rows):
            for key, val in coeffs.items():
                try:
                    typ, name = key.split(":", 1)
                except ValueError as exc:
                    raise ValueError(f"Invalid output coefficient key '{key}'") from exc

                if typ == "node":
                    col = self._col_node(name)
                elif typ == "iL":
                    col = self._col_iL(name)
                elif typ == "iV":
                    col = self._col_iV(name)
                else:
                    raise ValueError(f"Unknown output coefficient type '{typ}'")
                if col is not None:
                    vec[row, col] += float(val)

        vec_d = vec[:, self.diff_cols]
        vec_a = vec[:, self.alg_cols]
        if self.reduction_mode == "fast":
            C = vec_d + self._safe_matmul(vec_a, self.Phi)
            D = self._safe_matmul(vec_a, self.Psi)
            return C, D

        vec_sym = sp.Matrix(vec.tolist())
        vec_d_sym = vec_sym[:, self.diff_cols]
        vec_a_sym = vec_sym[:, self.alg_cols]
        C = vec_d_sym + vec_a_sym * self.Phi
        D = vec_a_sym * self.Psi
        return np.array(C.evalf(), dtype=float), np.array(D.evalf(), dtype=float)

    def _build_output_row(
        self,
        coefficients: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project one raw MNA expression to one state-space output row."""
        C, D = self._build_output_rows([coefficients])
        return C[0], D[0]

    def _state_derivative_row(
        self,
        node: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return rows representing ``dV(node)/dt = C*x + D*u``."""
        n_states = len(self.state_labels)
        if node in GND:
            return np.zeros(n_states), np.zeros(self.n_u)
        label = f"v_{node}"
        if label not in self.state_labels:
            raise ValueError(
                f"node '{node}' has no capacitor attached, so its voltage isn't "
                f"a state and dv/dt isn't defined this way. States: {self.state_labels}"
            )
        idx = self.state_label_idx[label]
        return self.A[idx, :], self.B_ss[idx, :]

    def add_node_voltage_output(self, node_name: str) -> None:
        """
        Add a node-to-ground voltage to the selected system outputs.

        Parameters
        ----------
        node_name:
            Netlist node whose voltage is measured relative to ground.

        Raises
        ------
        ValueError
            If ``node_name`` is not present in the circuit.
        """
        node = _normalize_node(node_name)
        if node not in GND and node not in self.node_idx:
            raise ValueError(f"Unknown node '{node_name}'")

        coefficients = {} if node in GND else {f"node:{node}": 1.0}
        self._selected_output_rows.append(self._build_output_row(coefficients))
        self.output_labels.append(f"V({node})")

    def add_dipole_current_output(self, dipole_name: str) -> None:
        """
        Add the current through a two-terminal netlist element.

        Parameters
        ----------
        dipole_name:
            Name of a resistor, inductor, capacitor, voltage source, or current
            source. Positive current follows the element declaration from
            ``n1`` to ``n2``.

        Raises
        ------
        ValueError
            If no supported dipole has the requested name.

        Notes
        -----
        Resistor current is derived from Ohm's law. Inductor and voltage-source
        currents are MNA branch unknowns. Capacitor current is computed from its
        voltage derivative. Current-source current is its corresponding input
        signal directly.
        """
        try:
            element = self.dipoles_by_name[dipole_name]
        except KeyError as exc:
            raise ValueError(f"Unknown dipole '{dipole_name}'") from exc

        if element.kind == "R":
            coefficients = {}
            if element.n1 not in GND:
                coefficients[f"node:{element.n1}"] = 1.0 / element.value
            if element.n2 not in GND:
                key = f"node:{element.n2}"
                coefficients[key] = coefficients.get(key, 0.0) - 1.0 / element.value
            row = self._build_output_row(coefficients)
        elif element.kind == "L":
            row = self._build_output_row({f"iL:{element.name}": 1.0})
        elif element.kind == "C":
            A_n1, B_n1 = self._state_derivative_row(element.n1)
            A_n2, B_n2 = self._state_derivative_row(element.n2)
            row = (
                element.value * (A_n1 - A_n2),
                element.value * (B_n1 - B_n2),
            )
        elif element.kind == "V":
            row = self._build_output_row({f"iV:{element.name}": 1.0})
        elif element.kind == "I":
            C_row = np.zeros(len(self.state_labels), dtype=float)
            D_row = np.zeros(self.n_u, dtype=float)
            D_row[self.input_names.index(element.name)] = 1.0
            row = C_row, D_row
        else:
            raise ValueError(
                f"Element '{dipole_name}' of type '{element.kind}' is not a dipole"
            )

        self._selected_output_rows.append(row)
        self.output_labels.append(f"I({element.name})")

    def get_system(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the reduced system with all selected outputs.

        Returns
        -------
        A, B, C, D:
            Matrices satisfying ``dx/dt = A*x + B*u`` and
            ``y = C*x + D*u``. Rows of ``C`` and ``D`` follow the order in
            which outputs were added. With no selected outputs, ``C`` and ``D``
            have zero rows and retain the correct state/input column counts.
        """
        if not self._selected_output_rows:
            return (
                self.A,
                self.B_ss,
                np.empty((0, len(self.state_labels)), dtype=float),
                np.empty((0, self.n_u), dtype=float),
            )
        C = np.vstack([row[0] for row in self._selected_output_rows])
        D = np.vstack([row[1] for row in self._selected_output_rows])
        return self.A, self.B_ss, C, D


class NetlistStateSpace(StateSpace):
    """
    PathSim state-space block constructed directly from a linear netlist.

    Parameters
    ----------
    netlist:
        Existing netlist path or inline netlist text. A :class:`str` is treated
        as a path when it names an existing file and as netlist text otherwise.
        A :class:`Path` is always treated as an explicit file path.
    output_voltages:
        Node names exposed as node-to-ground voltage outputs.
    output_currents:
        Names of R, L, C, voltage-source, or current-source dipoles exposed as
        current outputs. Positive current follows each netlist ``n1 -> n2``
        declaration.
    reduction_mode:
        Circuit reduction backend passed to :class:`CircuitModel`.
    initial_value:
        Initial differential state. Defaults to zero for every state.

    Notes
    -----
    Output ports are ordered with all requested voltages first, followed by all
    requested currents. Their labels are ``V(node)`` and ``I(dipole)``.
    Input ports retain the voltage-source-then-current-source ordering of the
    netlist model. The underlying :class:`CircuitModel` is available as
    :attr:`model`.

    Examples
    --------
    >>> block = NetlistStateSpace(
    ...     "filter.net",
    ...     output_voltages=["n1"],
    ...     output_currents=["Rload"],
    ...     reduction_mode="fast",
    ... )
    """

    def __init__(
        self,
        netlist: str | Path,
        output_voltages: list[str] | None = None,
        output_currents: list[str] | None = None,
        reduction_mode: ReductionMode = "symbolic",
        initial_value: np.ndarray | None = None,
    ):
        if isinstance(netlist, Path):
            elements = parse_netlist_file(netlist)
        elif isinstance(netlist, str):
            candidate = Path(netlist)
            try:
                is_file = candidate.is_file()
            except OSError:
                is_file = False
            elements = parse_netlist_file(candidate) if is_file else parse_netlist(netlist)
        else:
            raise TypeError("netlist must be a string or pathlib.Path")

        self.model = CircuitModel(elements, reduction_mode=reduction_mode)
        for node_name in output_voltages or []:
            self.model.add_node_voltage_output(node_name)
        for dipole_name in output_currents or []:
            self.model.add_dipole_current_output(dipole_name)

        A, B, C, D = self.model.get_system()
        super().__init__(
            A=A,
            B=B,
            C=C,
            D=D,
            initial_value=initial_value,
            state_labels=list(self.model.state_labels),
            input_labels=list(self.model.input_names),
            output_labels=list(self.model.output_labels),
        )