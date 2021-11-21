#!/usr/bin/env python3
#
# Tests if the degradation (toy) model works.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np
from pints.toy.stochastic import DegradationModel


class TestDegradationModel(unittest.TestCase):
    """
    Tests if the degradation (toy) model works.
    """
    def test_n_parameters(self):
        x_0 = 20
        model = DegradationModel(x_0)
        self.assertEqual(model.n_parameters(), 1)

    def test_simulation_length(self):
        x_0 = 20
        model = DegradationModel(x_0)
        times = np.linspace(0, 1, 100)
        k = [0.1]
        values = model.simulate(k, times)
        self.assertEqual(len(values), 100)

    def test_propensities(self):
        x_0 = 20
        k = [0.1]
        model = DegradationModel(x_0)
        self.assertTrue(
            np.allclose(
                model._propensities([x_0], k),
                np.array([2.0])))


if __name__ == '__main__':
    unittest.main()
