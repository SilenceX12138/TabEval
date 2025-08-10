Metrics
=====================

The TabEval metrics module provides a comprehensive suite of evaluation metrics for synthetic tabular data. All metrics inherit from the base :class:`~tabeval.metrics.core.MetricEvaluator` class and are organized by evaluation category.

Overview
--------

TabEval provides metrics across several evaluation dimensions:

- **Statistical Metrics**: Distribution-based comparisons between real and synthetic data
- **Privacy Metrics**: Privacy protection and differential privacy guarantees
- **Structure Metrics**: Feature-level utility and relationships
- **Density Metrics**: High-order and low-order data quality assessments

Main Interface
--------------

.. autoclass:: tabeval.metrics.Metrics
   :members:
   :undoc-members:
   :show-inheritance:

   The main interface for evaluating synthetic data quality. Provides a unified API for running all available metrics.

   **Example Usage:**

   .. code-block:: python

      from tabeval.metrics import Metrics
      import pandas as pd

      # Load your data
      X_real = pd.read_csv("real_data.csv")
      X_synthetic = pd.read_csv("synthetic_data.csv")

      # Evaluate all default metrics
      results = Metrics.evaluate(X_real, X_synthetic)

      # Evaluate specific metrics
      custom_metrics = {
          'stats': ['jensenshannon_dist', 'ks_test', 'wasserstein_dist'],
          'privacy': ['dcr']
      }
      results = Metrics.evaluate(X_real, X_synthetic, metrics=custom_metrics)

      # List all available metrics
      available_metrics = Metrics.list()
      print(available_metrics)

Base Classes
------------

.. autoclass:: tabeval.metrics.core.MetricEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

   Base class for all metrics. Provides common functionality including:
   
   - Caching mechanism for expensive computations
   - Standardized evaluation interface
   - Reduction operations (mean, max, min, median)
   - OneClass representation for advanced metrics

Statistical Metrics
-------------------

Statistical metrics compare the distributional properties between real and synthetic data.

.. autoclass:: tabeval.metrics.eval_statistical.StatisticalEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

   Base class for all statistical metrics.

Jensen-Shannon Distance
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_statistical.JensenShannonDistance
   :members:
   :undoc-members:
   :show-inheritance:

   Evaluates the average Jensen-Shannon distance between probability distributions.
   
   **Score Range**: [0, 1]
   
   **Direction**: minimize (0 = identical distributions, 1 = completely different)

Inverse KL Divergence
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_statistical.InverseKLDivergence
   :members:
   :undoc-members:
   :show-inheritance:

   Returns the average inverse of the Kullback–Leibler Divergence metric.
   
   **Score Range**: [0, 1]
   
   **Direction**: maximize (1 = same distribution, 0 = different distributions)

Kolmogorov-Smirnov Test
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_statistical.KolmogorovSmirnovTest
   :members:
   :undoc-members:
   :show-inheritance:

   Performs the Kolmogorov-Smirnov test for goodness of fit.
   
   **Score Range**: [0, 1]
   
   **Direction**: maximize (1 = identical distributions, 0 = totally different)

Chi-Squared Test
~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_statistical.ChiSquaredTest
   :members:
   :undoc-members:
   :show-inheritance:

   Performs the one-way chi-square test.
   
   **Score Range**: [0, 1]
   
   **Direction**: maximize (1 = identical distributions, 0 = different distributions)

Maximum Mean Discrepancy
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_statistical.MaximumMeanDiscrepancy
   :members:
   :undoc-members:
   :show-inheritance:

   Empirical maximum mean discrepancy with support for multiple kernels.
   
   **Supported Kernels**: "rbf", "linear", "polynomial"
   
   **Score Range**: [0, ∞)
   
   **Direction**: minimize (0 = same distributions, higher = more different)

Wasserstein Distance
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_statistical.WassersteinDistance
   :members:
   :undoc-members:
   :show-inheritance:

   Compare Wasserstein distance between original and synthetic data.
   
   **Score Range**: [0, ∞)
   
   **Direction**: minimize (0 = identical distributions)

PRDC Score
~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_statistical.PRDCScore
   :members:
   :undoc-members:
   :show-inheritance:

   Computes precision, recall, density, and coverage given two manifolds.
   
   **Returns**: Dictionary with precision, recall, density, and coverage scores
   
   **Direction**: maximize (all metrics range from 0 to 1)

Alpha Precision
~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_statistical.AlphaPrecision
   :members:
   :undoc-members:
   :show-inheritance:

   Evaluates alpha-precision, beta-recall, and authenticity scores for sample-level quality assessment.
   
   **Returns**: Dictionary with delta_precision_alpha, delta_coverage_beta, and authenticity scores
   
   **Direction**: maximize (all metrics range from 0 to 1)

Survival KM Distance
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_statistical.SurvivalKMDistance
   :members:
   :undoc-members:
   :show-inheritance:

   Distance between two Kaplan-Meier plots for survival analysis data.
   
   **Task Type**: survival_analysis only
   
   **Direction**: minimize

Frechet Inception Distance
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_statistical.FrechetInceptionDistance
   :members:
   :undoc-members:
   :show-inheritance:

   Calculates the Frechet Inception Distance (FID) for image data evaluation.
   
   **Data Type**: images only
   
   **Direction**: minimize

Privacy Metrics
---------------

Privacy metrics assess the privacy protection offered by synthetic data generation.

.. autoclass:: tabeval.metrics.eval_privacy.PrivacyEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

   Base class for all privacy metrics.

DCR (Baseline Protection)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_privacy.DCR
   :members:
   :undoc-members:
   :show-inheritance:

   Evaluates the differential privacy guarantees of synthetic data using the DCR baseline protection metric.
   
   **Returns**: Dictionary with score, syn2real_median, and random2real_median
   
   **Direction**: maximize

Structure Metrics
-----------------

Structure metrics evaluate the utility and relationships preserved in synthetic data.

.. autoclass:: tabeval.metrics.eval_structure.StructureEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

   Base class for all structure metrics.

Utility Per Feature
~~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_structure.UtilityPerFeature
   :members:
   :undoc-members:
   :show-inheritance:

   Computes the utility per feature of synthetic data by training predictive models.
   
   **Returns**: Dictionary with negative_RMSE (regression tasks) and balanced_accuracy (classification tasks)
   
   **Direction**: maximize

Density Metrics
---------------

Density metrics assess the quality of synthetic data through density-based comparisons.

.. autoclass:: tabeval.metrics.eval_density.DensityEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

   Base class for all density metrics.

Low Order Metrics
~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_density.LowOrderMetrics
   :members:
   :undoc-members:
   :show-inheritance:

   Evaluates low-order metrics including Trend and Shape using SDMetrics quality reports.
   
   **Returns**: Dictionary with shape and trend scores
   
   **Direction**: maximize

High Order Metrics
~~~~~~~~~~~~~~~~~~

.. autoclass:: tabeval.metrics.eval_density.HighOrderMetrics
   :members:
   :undoc-members:
   :show-inheritance:

   Evaluates high-order data quality using alpha-precision, beta-recall, and authenticity metrics.
   
   **Returns**: Dictionary with alpha_precision, beta_recall, and authenticity scores
   
   **Direction**: maximize

Visualization
-------------

The metrics module also provides visualization utilities for comparing real and synthetic data.

.. automodule:: tabeval.metrics.plots
   :members:
   :undoc-members:
   :show-inheritance:

   **Available Functions:**
   
   - :func:`plot_marginal_comparison`: Creates marginal distribution comparison plots
   - :func:`plot_tsne`: Generates t-SNE plots for data comparison

Utility Functions
-----------------

.. automodule:: tabeval.metrics._utils
   :members:
   :undoc-members:
   :show-inheritance:

   Internal utility functions for metric computation.

.. automodule:: tabeval.metrics.scores
   :members:
   :undoc-members:
   :show-inheritance:

   Score evaluation and aggregation utilities.

Metric Categories and Default Configuration
-------------------------------------------

The metrics are organized into the following categories with their default configurations:

.. code-block:: python

   {
       'stats': [
           'jensenshannon_dist', 'chi_squared_test', 'inv_kl_divergence', 
           'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 
           'prdc', 'alpha_precision', 'survival_km_distance'
       ],
       'privacy': ['dcr'],
       'structure': ['utility_per_feature'],
       'density': ['low_order', 'high_order']
   }

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from tabeval.metrics import Metrics
   import pandas as pd

   # Load your datasets
   real_data = pd.read_csv("real_data.csv")
   synthetic_data = pd.read_csv("synthetic_data.csv")

   # Run all default metrics
   results = Metrics.evaluate(real_data, synthetic_data)
   print(results)

Custom Metric Selection
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Select specific metrics to run
   custom_metrics = {
       'stats': ['jensenshannon_dist', 'wasserstein_dist'],
       'privacy': ['dcr']
   }
   
   results = Metrics.evaluate(
       real_data, 
       synthetic_data, 
       metrics=custom_metrics,
       task_type='classification',
       n_folds=3,
       workspace=Path('my_workspace')
   )

Survival Analysis Example
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For survival analysis data
   results = Metrics.evaluate(
       X_gt=survival_real_data,
       X_syn=survival_synthetic_data,
       task_type='survival_analysis',
       metrics={'stats': ['survival_km_distance']}
   )

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Advanced configuration with all parameters
   results = Metrics.evaluate(
       X_gt=real_data,
       X_syn=synthetic_data,
       X_train=training_data,  # Optional: training data for some metrics
       reduction='median',     # Aggregation method: 'mean', 'median', 'min', 'max'
       n_histogram_bins=20,    # Number of bins for histogram-based metrics
       task_type='regression', # Task type affects model evaluation
       random_state=42,        # For reproducibility
       workspace=Path('cache'), # Caching directory
       use_cache=True,         # Enable result caching
       n_folds=5              # Cross-validation folds
   )

Notes
-----

- All metrics support caching to avoid recomputation of expensive operations
- Most metrics work with any tabular data, but some are specialized (e.g., survival analysis, images)
- The evaluation framework automatically handles data encoding and preprocessing
- Results are returned as pandas DataFrames for easy analysis and visualization
