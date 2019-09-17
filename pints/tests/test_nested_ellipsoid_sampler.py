#!/usr/bin/env python
#
# Tests ellipsoidal nested sampler.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import numpy as np

import pints
import pints.toy

# Unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

debug = False


class TestNestedEllipsoidSampler(unittest.TestCase):
    """
    Unit (not functional!) tests for :class:`NestedEllipsoidSampler`.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare for the test. """
        # Create toy model
        model = pints.toy.LogisticModel()
        cls.real_parameters = [0.015, 500]
        times = np.linspace(0, 1000, 1000)
        values = model.simulate(cls.real_parameters, times)

        # Add noise
        np.random.seed(1)
        cls.noise = 10
        values += np.random.normal(0, cls.noise, values.shape)
        cls.real_parameters.append(cls.noise)

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(model, times, values)

        # Create a uniform prior over both the parameters and the new noise
        # variable
        cls.log_prior = pints.UniformLogPrior(
            [0.01, 400],
            [0.02, 600]
        )

        # Create a log-likelihood
        cls.log_likelihood = pints.GaussianKnownSigmaLogLikelihood(
            problem, cls.noise)

    def test_construction_errors(self):
        """ Tests if invalid constructor calls are picked up. """

        # First arg must be a log likelihood
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogPrior',
            pints.NestedEllipsoidSampler, self.log_likelihood)

    def test_hyper_params(self):
        """
        Tests the hyper parameter interface is working.
        """
        sampler = pints.NestedEllipsoidSampler(self.log_prior)
        self.assertEqual(sampler.n_hyper_parameters(), 6)
        sampler.set_hyper_parameters([220, 130, 2.0, 133, 1, 0.8])
        self.assertEqual(sampler.n_active_points(), 220)
        self.assertEqual(sampler.n_rejection_samples(), 130)
        self.assertEqual(sampler.enlargement_factor(), 2.0)
        self.assertEqual(sampler.ellipsoid_update_gap(), 133)
        self.assertTrue(sampler.dynamic_enlargement_factor())
        self.assertTrue(sampler.alpha(), 0.8)

    def test_getters_and_setters(self):
        """
        Tests various get() and set() methods.
        """
        sampler = pints.NestedEllipsoidSampler(self.log_prior)

        # Active points
        x = sampler.n_active_points() + 1
        self.assertNotEqual(sampler.n_active_points(), x)
        sampler.set_n_active_points(x)
        self.assertEqual(sampler.n_active_points(), x)
        self.assertRaisesRegex(
            ValueError, 'greater than 5', sampler.set_n_active_points, 5)

        # Rejection samples
        x = sampler.n_rejection_samples() + 1
        self.assertNotEqual(sampler.n_rejection_samples(), x)
        sampler.set_n_rejection_samples(x)
        self.assertEqual(sampler.n_rejection_samples(), x)
        self.assertRaisesRegex(
            ValueError, 'negative', sampler.set_n_rejection_samples, -1)

        # Enlargement factor
        x = sampler.enlargement_factor() * 2
        self.assertNotEqual(sampler.enlargement_factor(), x)
        sampler.set_enlargement_factor(x)
        self.assertEqual(sampler.enlargement_factor(), x)
        self.assertRaisesRegex(
            ValueError, 'exceed 1', sampler.set_enlargement_factor, 0.5)
        self.assertRaisesRegex(
            ValueError, 'exceed 1', sampler.set_enlargement_factor, 1)

        # Ellipsoid update gap
        x = sampler.ellipsoid_update_gap() * 2
        self.assertNotEqual(sampler.ellipsoid_update_gap(), x)
        sampler.set_ellipsoid_update_gap(x)
        self.assertEqual(sampler.ellipsoid_update_gap(), x)
        self.assertRaisesRegex(
            ValueError, 'exceed 1', sampler.set_ellipsoid_update_gap, 0.5)
        self.assertRaisesRegex(
            ValueError, 'exceed 1', sampler.set_ellipsoid_update_gap, 1)

        # dynamic enlargement factor
        self.assertTrue(not sampler.dynamic_enlargement_factor())
        sampler.set_dynamic_enlargement_factor(1)
        self.assertTrue(sampler.dynamic_enlargement_factor())

        # alpha
        self.assertRaises(ValueError, sampler.set_alpha, -0.2)
        self.assertRaises(ValueError, sampler.set_alpha, 1.2)
        self.assertEqual(sampler.alpha(), 1)
        sampler.set_alpha(0.4)
        self.assertEqual(sampler.alpha(), 0.4)

        # initial phase
        self.assertTrue(sampler.needs_initial_phase())
        self.assertTrue(sampler.in_initial_phase())
        sampler.set_initial_phase(False)
        self.assertTrue(not sampler.in_initial_phase())
        self.assertEqual(sampler.name(), 'Nested ellipsoidal sampler')


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
