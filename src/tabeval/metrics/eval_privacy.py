# stdlib
import hashlib
import platform
from abc import abstractmethod
from typing import Any, Dict

# third party
# tabeval absolute
from pydantic import validate_call
from sdmetrics.single_table import DCRBaselineProtection

# tabeval absolute
from tabeval.plugins.core.dataloader import DataLoader
from tabeval.utils.reproducibility import clear_cache
from tabeval.utils.serialization import load_from_file, save_to_file

# tabeval relative
from .core import MetricEvaluator


class PrivacyEvaluator(MetricEvaluator):
    """
    .. inheritance-diagram:: tabeval.metrics.eval_privacy.PrivacyEvaluator
        :parts: 1

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "privacy"

    @abstractmethod
    def timestamp(self) -> str:
        """Used to determine the version of the metric."""
        raise NotImplementedError("Subclasses must implement this timestamp method")

    @abstractmethod
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader, **kwargs) -> Dict:
        raise NotImplementedError("Subclasses must implement this evaluation method")

    @validate_call(config=dict(arbitrary_types_allowed=True))
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

    @validate_call(config=dict(arbitrary_types_allowed=True))
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
            return tuple(
                sorted(
                    [self._make_hashable(item) for item in obj],
                    key=lambda x: (str(type(x)), str(x)),
                )
            )
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


class DCR(PrivacyEvaluator):
    """
    .. inheritance-diagram:: tabeval.metrics.eval_privacy.DCR
        :parts: 1

    Evaluates the differential privacy guarantees of the synthetic data.

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
        return "dcr"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def timestamp(self):
        return "2025-09-11"

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X: DataLoader,
        X_syn: DataLoader,
        metadata: dict,
        fast_mode: bool,
        max_num_samples: int | None = None,
    ) -> Dict:
        # === Set parameters for sub-sampling ===
        if fast_mode or max_num_samples is not None:
            num_rows_subsample = min(X.dataframe().shape[0], X_syn.dataframe().shape[0])
            # Apply max_num_samples if provided
            if max_num_samples is not None:
                num_rows_subsample = min(max_num_samples, num_rows_subsample)
            # If fast_mode is enabled, limit to 1000 rows
            if fast_mode:
                num_rows_subsample = min(1000, num_rows_subsample)
        else:
            num_rows_subsample = None

        # === Compute the metric ===
        metric_dict = DCRBaselineProtection.compute_breakdown(
            real_data=X.dataframe(),
            synthetic_data=X_syn.dataframe(),
            metadata=metadata,
            num_rows_subsample=num_rows_subsample,
            num_iterations=1,
        )

        return {
            "score": metric_dict["score"],
            "syn2real_median": metric_dict["median_DCR_to_real_data"]["synthetic_data"],
            "random2real_median": metric_dict["median_DCR_to_real_data"]["random_data_baseline"],
        }
