Metrics API
===========

The metrics module provides comprehensive evaluation capabilities for synthetic data.

Core Classes
------------

.. automodule:: tabeval.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Metrics Evaluator
-----------------

.. autoclass:: tabeval.metrics.eval.Metrics
   :members:
   :undoc-members:
   :show-inheritance:

Statistical Metrics
-------------------

.. automodule:: tabeval.metrics.eval_statistical
   :members:
   :undoc-members:
   :show-inheritance:

Statistical Evaluator Base Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_statistical.StatisticalEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

Individual Statistical Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_statistical.JensenShannonDistance
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_statistical.KolmogorovSmirnovTest
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_statistical.WassersteinDistance
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_statistical.ChiSquaredTest
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_statistical.MaximumMeanDiscrepancy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_statistical.PRDCScore
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_statistical.AlphaPrecision
   :members:
   :undoc-members:
   :show-inheritance:

Performance Metrics
-------------------

.. automodule:: tabeval.metrics.eval_performance
   :members:
   :undoc-members:
   :show-inheritance:

Performance Evaluator Base Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_performance.PerformanceEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

Individual Performance Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_performance.PerformanceEvaluatorLinear
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_performance.PerformanceEvaluatorMLP
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_performance.PerformanceEvaluatorXGB
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_performance.FeatureImportanceRankDistance
   :members:
   :undoc-members:
   :show-inheritance:

Privacy Metrics
---------------

.. automodule:: tabeval.metrics.eval_privacy
   :members:
   :undoc-members:
   :show-inheritance:

Privacy Evaluator Base Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_privacy.PrivacyEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

Individual Privacy Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_privacy.kAnonymization
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_privacy.lDiversityDistinct
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_privacy.DeltaPresence
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_privacy.IdentifiabilityScore
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_privacy.DomiasMIABNAF
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_privacy.DomiasMIAKDE
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_privacy.DomiasMIAPrior
   :members:
   :undoc-members:
   :show-inheritance:

Detection Metrics
-----------------

.. automodule:: tabeval.metrics.eval_detection
   :members:
   :undoc-members:
   :show-inheritance:

Detection Evaluator Base Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_detection.DetectionEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

Individual Detection Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_detection.SyntheticDetectionLinear
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_detection.SyntheticDetectionMLP
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_detection.SyntheticDetectionXGB
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_detection.SyntheticDetectionGMM
   :members:
   :undoc-members:
   :show-inheritance:

Sanity Check Metrics
--------------------

.. automodule:: tabeval.metrics.eval_sanity
   :members:
   :undoc-members:
   :show-inheritance:

Basic Metric Evaluator
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_sanity.BasicMetricEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

Individual Sanity Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_sanity.DataMismatchScore
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_sanity.CommonRowsProportion
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_sanity.NearestSyntheticNeighborDistance
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_sanity.CloseValuesProbability
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tabeval.metrics.eval_sanity.DistantValuesProbability
   :members:
   :undoc-members:
   :show-inheritance:

Utility Classes
---------------

Score Evaluator
~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.scores.ScoreEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

Weighted Metrics
~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.weighted_metrics.WeightedMetrics
   :members:
   :undoc-members:
   :show-inheritance:

Core Metric Interface
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.core.MetricEvaluator
   :members:
   :undoc-members:
   :show-inheritance:
