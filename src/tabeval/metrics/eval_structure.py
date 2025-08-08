# stdlib
import hashlib
import os
import platform
import random
import time
from abc import abstractmethod
from typing import Any, Dict

# third party
import numpy as np
import pandas as pd
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular import TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from pgmpy.estimators.CITests import chi_square, pearsonr, pillai_trace
from pydantic import validate_arguments
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# tabeval absolute
from tabeval.metrics.core import MetricEvaluator
from tabeval.plugins.core.dataloader import DataLoader
from tabeval.utils.reproducibility import clear_cache
from tabeval.utils.serialization import load_from_file, save_to_file


class StructureEvaluator(MetricEvaluator):
    """
    .. inheritance-diagram:: tabeval.metrics.eval_structure.StructureEvaluator
        :parts: 1

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "structure"

    @abstractmethod
    def timestamp(self) -> str:
        """Used to determine the version of the metric."""
        raise NotImplementedError("Subclasses must implement this timestamp method")

    @abstractmethod
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader, **kwargs) -> Dict:
        raise NotImplementedError("Subclasses must implement this evaluation method")

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader, **kwargs) -> Dict:
        # Create hashable representation of kwargs
        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{self.timestamp()}_{X_gt.hash()}_{X_syn.hash()}_{self._get_deterministic_hash(kwargs)}_{self._reduction}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            return load_from_file(cache_file)

        clear_cache()
        results = self._evaluate(X_gt, X_syn, **kwargs)
        save_to_file(cache_file, results)
        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> float:
        return self.evaluate(X_gt, X_syn)[self._default_metric]

    def _get_deterministic_hash(self, obj):
        """Generate a deterministic hash for an object."""
        # Note: the inherent `hash` function in Python is not deterministic across different runs: https://stackoverflow.com/questions/27522626/hash-function-in-python-3-3-returns-different-results-between-sessions
        # First convert to a hashable type
        hashable_obj = self._make_hashable(obj)

        # Convert to a string representation
        obj_str = str(hashable_obj)

        # Create hash using the string bytes
        hasher = hashlib.sha256()
        hasher.update(obj_str.encode("utf-8"))
        return hasher.hexdigest()

    def _make_hashable(self, obj):
        """Convert unhashable types to hashable ones in a deterministic way."""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, list):
            return tuple(self._make_hashable(item) for item in obj)
        elif isinstance(obj, set):
            # Sort set items to ensure consistent ordering
            return tuple(sorted([self._make_hashable(item) for item in obj], key=lambda x: (str(type(x)), str(x))))
        elif isinstance(obj, dict):
            # Sort by string representation of key type and then key value
            # This ensures consistent ordering even with mixed key types
            items = []
            for k, v in obj.items():
                k_hashable = self._make_hashable(k)
                v_hashable = self._make_hashable(v)
                items.append((k_hashable, v_hashable))
            # Sort by key type name first, then by key string representation
            return tuple(sorted(items, key=lambda x: (str(type(x[0])), str(x[0]))))
        else:
            # For other objects, use class name + str representation
            # Strip any memory addresses from the string
            class_name = obj.__class__.__name__
            obj_str = str(obj)
            # Remove memory addresses like '0x7f8b2d0b3c10'
            # stdlib
            import re

            obj_str = re.sub(r" at 0x[0-9a-f]+", "", obj_str)
            return f"{class_name}:{obj_str}"


class CITest(StructureEvaluator):
    """
    .. inheritance-diagram:: tabeval.metrics.eval_structure.CITest
        :parts: 1

    Compute the AUROC of binary CI detection task.

    Args:
        X: original data
        X_syn: synthetically generated data

    Returns:
        results: dict
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "CI_test"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def timestamp(self):
        return "2025-04-04"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X: DataLoader,
        X_syn: DataLoader,
        column_list: list,  # list of column names (the last one is default to be the target)
        dependency_dict: dict,  # GT dependency dictionary
        test_method: str,  # CI test method
        significance_level: float = 0.05,  # Significance level
        max_ratio_ci_test: float = 1,  # Maximum ratio of CI relationships to test
    ) -> Dict:
        # === Prepare the GT dependency ===
        ci_list = dependency_dict["conditional_independent_set"]
        cd_list = dependency_dict["conditional_dependent_set"]
        ci_list_sampled = random.sample(ci_list, int(max_ratio_ci_test * len(ci_list)))
        cd_list_sampled = random.sample(cd_list, int(max_ratio_ci_test * len(cd_list)))
        dependency_list = ci_list_sampled + cd_list_sampled

        # === Highlight the local structure w.r.t. each feature ===
        target2local_structure_index = {col: set() for col in column_list}
        for i, dependency in enumerate(dependency_list):
            X, Y, S = dependency
            target2local_structure_index[X].add(i)
            target2local_structure_index[Y].add(i)
            for col in S:
                target2local_structure_index[col].add(i)

        # === Evaluate the CI (y=1) and CD (y=0) on synthetic data ===
        X_syn_df = pd.DataFrame(X_syn.data, columns=column_list)
        y_gt = [1] * len(ci_list_sampled) + [0] * len(cd_list_sampled)
        y_pred = []
        for dependency in dependency_list:
            X, Y, S = dependency
            match test_method:
                case "pillai":
                    # Condition on all other features
                    _, p_value = pillai_trace(X, Y, S, X_syn_df, boolean=False)
                case "chi_square":
                    _, p_value, _ = chi_square(X, Y, S, X_syn_df, boolean=False)
                case "pearsonr":
                    _, p_value = pearsonr(X, Y, S, X_syn_df, boolean=False)
            # If p-value is smaller than or equal to significance level, it means the two features are dependent
            # And thus we should have an edge between them
            if p_value <= significance_level:
                y_pred.append(0)
            else:
                y_pred.append(1)

        # === Compute the global score ===
        auroc_global = roc_auc_score(y_gt, y_pred, average="weighted")
        balanced_accuracy_global = balanced_accuracy_score(y_gt, y_pred)

        # === Compute the local score ===
        auroc_local_dict = {
            col: [
                (
                    roc_auc_score(
                        np.array(y_gt)[list(target2local_structure_index[col])],
                        np.array(y_pred)[list(target2local_structure_index[col])],
                        average="weighted",
                    )
                    if len(target2local_structure_index[col]) > 0
                    else 1  # For some columns, the local structure is empty (no conditional independence/dependence relationships)
                )
            ]
            for col in column_list
        }
        balanced_accuracy_local_dict = {
            col: [
                (
                    balanced_accuracy_score(
                        np.array(y_gt)[list(target2local_structure_index[col])],
                        np.array(y_pred)[list(target2local_structure_index[col])],
                    )
                    if len(target2local_structure_index[col]) > 0
                    else 1
                )
            ]
            for col in column_list
        }
        auroc_local_mean = np.mean(list(auroc_local_dict.values()))
        balanced_accuracy_local_mean = np.mean(list(balanced_accuracy_local_dict.values()))

        return {
            "auroc_global": auroc_global,
            "balanced_accuracy_global": balanced_accuracy_global,
            "auroc_local_mean": auroc_local_mean,
            "balanced_accuracy_local_mean": balanced_accuracy_local_mean,
            "auroc_local": auroc_local_dict,
            "balanced_accuracy_local": balanced_accuracy_local_dict,
        }


class UtilityPerFeature(StructureEvaluator):
    """
    .. inheritance-diagram:: tabeval.metrics.eval_structure.UtilityPerFeature
        :parts: 1

    Compute the utility per feature of the synthetic data.

    Args:
        X: original data
        X_syn: synthetically generated data

    Returns:
        results: dict
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "utility_per_feature"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def timestamp(self):
        return "2025-04-09"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X: DataLoader,
        X_syn: DataLoader,
        column_list: list,  # list of column names (the last one is default to be the target)
        time_limit: int,  # limit for total training time (second)
    ) -> Dict:
        # === Prepare the data ===
        X = pd.DataFrame(X.data, columns=column_list)
        X_syn_df = pd.DataFrame(X_syn.data, columns=column_list)

        # === Prepare predictors ===
        # Only keep some default models
        custom_hyperparameters = get_hyperparameter_config("very_light")
        included_model_list = [
            # "NN_TORCH",
            "XGB",
        ]
        custom_hyperparameters = {k: v for k, v in custom_hyperparameters.items() if k in included_model_list}
        # Add other models
        # custom_hyperparameters["LR"] = {}
        custom_hyperparameters["KNN"] = {}
        custom_hyperparameters[CustomTabPFNModel] = {}

        # === Enumerate all features ===
        target2regression_score_mean = {}  # negative RMSE
        target2classification_score_mean = {}  # balanced accuracy
        for col in column_list:
            if X_syn_df[col].nunique() == 1:
                # Skip constant columns
                target2classification_score_mean[col] = [1]
                continue

            predictor = TabularPredictor(
                label=col,
                path=os.path.join(
                    self._workspace, "AutogluonModels", time.strftime("%Y%m%d_%H%M%S"), str(hash(str(column_list)))
                ),
                log_to_file=True,
                verbosity=0,
            ).fit(
                train_data=X_syn_df,
                tuning_data=None,
                hyperparameters=custom_hyperparameters,
                fit_weighted_ensemble=False,
                presets="medium_quality",
                time_limit=time_limit,
            )

            # As leaderboard will reload the model, if multiple threads are used, there could be conflicts.
            # e.g., run 1 trained an XGB model on dataset A, and run 2 trained another XGB model on dataset B
            # Then run 1 may evaluate the XGB from run 2, thus leading to crashed runs
            if predictor.problem_type == "regression":
                leaderboard = predictor.leaderboard(X, extra_metrics=["root_mean_squared_error"])
                target2regression_score_mean[col] = [leaderboard["score_test"].mean()]
            else:
                leaderboard = predictor.leaderboard(X, extra_metrics=["balanced_accuracy"])
                target2classification_score_mean[col] = [leaderboard["balanced_accuracy"].mean()]

        return {
            "negative_RMSE": target2regression_score_mean,
            "balanced_accuracy": target2classification_score_mean,
        }


class CustomTabPFNModel(AbstractModel):
    def __init__(self, **kwargs):
        # Simply pass along kwargs to parent, and init our internal `_feature_generator` variable to None
        super().__init__(**kwargs)
        self._feature_generator = None

    # The `_preprocess` method takes the input data and transforms it to the internal representation usable by the model.
    # `_preprocess` is called by `preprocess` and is used during model fit and model inference.
    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        X = super()._preprocess(X, **kwargs)

        if is_train:
            # X will be the training data.
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            # This converts categorical features to numeric via stateful label encoding.
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        # Add a fillna call to handle missing values.
        # Some algorithms will be able to handle NaN values internally (LightGBM).
        # In those cases, you can simply pass the NaN values into the inner model.
        # Finally, convert to numpy for optimized memory usage and because sklearn RF works with raw numpy input.
        return X.fillna(0).to_numpy(dtype=np.float32)

    # The `_fit` method takes the input training data (and optionally the validation data) and trains the model.
    def _fit(
        self,
        X: pd.DataFrame,  # training data
        y: pd.Series,  # training labels
        # X_val=None,  # val data (unused in RF model)
        # y_val=None,  # val labels (unused in RF model)
        # time_limit=None,  # time limit in seconds (ignored in tutorial)
        **kwargs,
    ):  # kwargs includes many other potential inputs, refer to AbstractModel documentation for details

        # Limit the number of rows to 10000 for training.
        if X.shape[0] > 10000:
            # If the training data is large, we will use a smaller subset of the data to fit the feature generator.
            # This is done to avoid overfitting to the training data.
            # The feature generator will be fit on a random sample of 10,000 rows from the training data.
            X = X.sample(n=10000, random_state=42)
            y = y.loc[X.index]

        # First we import the required dependencies for the model. Note that we do not import them outside of the method.
        # This enables AutoGluon to be highly extensible and modular.
        # For an example of best practices when importing model dependencies, refer to LGBModel.
        # third party
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        # Valid self.problem_type values include ['binary', 'multiclass', 'regression', 'quantile', 'softclass']
        if self.problem_type in ["regression", "softclass"]:
            model_cls = TabPFNRegressor
        else:
            model_cls = TabPFNClassifier
            # Limit the number of classes to 10 for training.
            if len(y.unique()) > 10:
                raise ValueError("TabPFN only supports up to 10 classes. Please use a different model for this task.")

        # Make sure to call preprocess on X near the start of `_fit`.
        # This is necessary because the data is converted via preprocess during predict, and needs to be in the same format as during fit.
        X = self.preprocess(X, is_train=True)
        # This fetches the user-specified (and default) hyperparameters for the model.
        params = self._get_model_params()
        # self.model should be set to the trained inner model, so that internally during predict we can call `self.model.predict(...)`
        self.model = model_cls(**params)
        self.model.fit(X, y)

    # The `_set_default_params` method defines the default hyperparameters of the model.
    # User-specified parameters will override these values on a key-by-key basis.
    def _set_default_params(self):
        default_params = {
            "n_estimators": 1,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # The `_get_default_auxiliary_params` method defines various model-agnostic parameters such as maximum memory usage and valid input column dtypes.
    # For most users who build custom models, they will only need to specify the valid/invalid dtypes to the model here.
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            # the total set of raw dtypes are: ['int', 'float', 'category', 'object', 'datetime']
            # object feature dtypes include raw text and image paths, which should only be handled by specialized models
            # datetime raw dtypes are generally converted to int in upstream pre-processing,
            # so models generally shouldn't need to explicitly support datetime dtypes.
            valid_raw_types=["int", "float", "category"],
            # Other options include `valid_special_types`, `ignored_type_group_raw`, and `ignored_type_group_special`.
            # Refer to AbstractModel for more details on available options.
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
