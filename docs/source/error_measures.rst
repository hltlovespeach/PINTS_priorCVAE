**************
Error measures
**************

.. currentmodule:: pints

Error measures are callable objects that return some scalar representing the
error between a model and an experiment.

Example::

    error = pints.SumOfSquaresError(problem)
    x = [1,2,3]
    fx = error(x)

Overview:

- :class:`ErrorMeasure`
- :class:`MeanSquaredError`
- :class:`NormalisedRootMeanSquaredError`
- :class:`ProbabilityBasedError`
- :class:`ProblemErrorMeasure`
- :class:`RootMeanSquaredError`
- :class:`SumOfErrors`
- :class:`SumOfSquaresError`


.. autoclass:: ErrorMeasure

.. autoclass:: MeanSquaredError

.. autoclass:: NormalisedRootMeanSquaredError

.. autoclass:: ProbabilityBasedError

.. autoclass:: ProblemErrorMeasure

.. autoclass:: RootMeanSquaredError

.. autoclass:: SumOfErrors

.. autoclass:: SumOfSquaresError
