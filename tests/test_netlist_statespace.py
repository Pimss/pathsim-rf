########################################################################################
##
##                                  TESTS FOR
##                              'netlist_to_statespace.py'
##
########################################################################################

# IMPORTS ==============================================================================
import unittest
from pathlib import Path
import numpy as np
import importlib.util

from pathsim_rf.netlist_to_statespace import CircuitModel, NetlistStateSpace, parse_netlist_file, parse_netlist
from pathsim.blocks.lti import StateSpace

# TESTS ================================================================================

class TestNetlistStateSpace(unittest.TestCase):
    """Test the NetlistStateSpace block"""

    def test_path_string_matches_manual_circuit_model(self):
        """Test CircuitModel is correctly used within NetlistStateSpace"""
        netlist_path = "filter.net"
        model = CircuitModel(parse_netlist_file(netlist_path), reduction_mode="symbolic")
        model.add_node_voltage_output("n1")
        model.add_dipole_current_output("Rload")
        A, B, C, D = model.get_system()

        block = NetlistStateSpace(
            str(netlist_path),
            output_voltages=["n1"],
            output_currents=["Rload"],
            reduction_mode="symbolic",
        )

        self.assertIsInstance(block, StateSpace)
        np.testing.assert_allclose(block.A, A)
        np.testing.assert_allclose(block.B, B)
        np.testing.assert_allclose(block.C, C)
        np.testing.assert_allclose(block.D, D)
        self.assertEqual(block.state_labels, model.state_labels)
        self.assertEqual(block.input_labels, ["V1"])
        self.assertEqual(block.output_labels, ["V(n1)", "I(Rload)"])

    def test_path_object_preserves_feedback_source_input(self):
        """Test current sources feedback used to chain models"""
        block = NetlistStateSpace("filter_without_load.net",
            output_voltages=["n1"],
            output_currents=["Lfilter"],
            reduction_mode="symbolic",
        )

        self.assertEqual(block.input_labels, ["V1", "Bload"])
        self.assertEqual(block.output_labels, ["V(n1)", "I(Lfilter)"])
        self.assertEqual(len(block.inputs), 2)
        self.assertEqual(len(block.outputs), 2)

    def test_inline_netlist_string(self):
        """Test direct netlist use to class for instance generation"""
        block = NetlistStateSpace(
            """
            V1 in 0 1
            R1 in out 10
            C1 out 0 1u
            """,
            output_voltages=["out"],
            output_currents=["R1", "V1"],
            reduction_mode="symbolic",
        )

        self.assertEqual(block.input_labels, ["V1"])
        self.assertEqual(block.output_labels, ["V(out)", "I(R1)", "I(V1)"])
        self.assertEqual(block.C.shape[0], 3)
        self.assertEqual(block.D.shape[0], 3)

    def test_explicit_missing_path_raises(self):
        """Test missing .net file provided"""
        with self.assertRaises(FileNotFoundError):
            NetlistStateSpace(Path("missing.net"))

    def test_invalid_output_names_raise(self):
        """Test invalid output names provided for state space generated"""
        netlist_path = "filter.net"
        with self.assertRaisesRegex(ValueError, "Unknown node"):
            NetlistStateSpace(netlist_path, output_voltages=["missing"])
        with self.assertRaisesRegex(ValueError, "Unknown dipole"):
            NetlistStateSpace(netlist_path, output_currents=["R404"])

@unittest.skipUnless(importlib.util.find_spec("mumps"), "python-mumps is not installed")
class TestFastReduction(unittest.TestCase):
    """Class validation the fast implementation for reduction mode using mumps
       using python-mumps"""
    def test_fast_mode_matches_symbolic_on_small_case(self):
        """Test if fast reduction mode is equivalent to symbolic
            skipped if mumps not installed"""
        elements = parse_netlist(
            """
            V1 n1 0 1
            R1 n1 n2 2k
            C1 n2 0 2u
            L1 n2 n3 3m
            L2 n3 0 5m
            R2 n3 0 4k
            K1 L1 L2 0.1
            """
        )
        symbolic = CircuitModel(elements, reduction_mode="symbolic")
        fast = CircuitModel(elements, reduction_mode="fast")

        np.testing.assert_allclose(fast.A, symbolic.A, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(fast.B_ss, symbolic.B_ss, rtol=1e-9, atol=1e-12)


# RUN TESTS LOCALLY ====================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)