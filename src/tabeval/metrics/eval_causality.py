# stdlib
import platform
from abc import abstractmethod
from typing import Any, Dict, Optional

# third party
import numpy as np
# from cdt.metrics import SHD, precision_recall
from pydantic import validate_arguments

# tabeval absolute
from tabeval.metrics.core import MetricEvaluator
from tabeval.plugins.core.dataloader import DataLoader
from tabeval.utils.reproducibility import clear_cache
from tabeval.utils.serialization import load_from_file, save_to_file

# from causallearn.search.FCMBased import lingam


class CausalityEvaluator(MetricEvaluator):
    """
    .. inheritance-diagram:: tabeval.metrics.eval_causality.CausalityEvaluator
        :parts: 1

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "causality"

    @abstractmethod
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict: ...

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            return load_from_file(cache_file)

        clear_cache()
        results = self._evaluate(X_gt, X_syn)
        save_to_file(cache_file, results)
        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> float:
        return self.evaluate(X_gt, X_syn)[self._default_metric]


# class StructuralHammingDistance(CausalityEvaluator):
#     """
#     .. inheritance-diagram:: tabeval.metrics.eval_causality.StructuralHammingDistance
#         :parts: 1

#     Compare structural hamming distance between original data and synthetic data.
#     If the GT causal graph is known, the structural hamming distance can be used to evaluate the quality of the synthetic data.
#     Otherwise, causal graph of the original data will be derived by a causal discovery method, and the SHD can only serve as a reference.

#     Args:
#         X: original data
#         X_syn: synthetically generated data

#     Returns:
#         results: dict
#     """

#     def __init__(
#         self,
#         causal_discovery_method="lingam",
#         num_boostrapping=10,
#         **kwargs: Any,
#     ) -> None:
#         super().__init__(default_metric="score", **kwargs)

#         self.causal_discovery_method = causal_discovery_method
#         self.num_boostrapping = num_boostrapping

#     @staticmethod
#     def name() -> str:
#         return "SHD"

#     @staticmethod
#     def direction() -> str:
#         return "minimize"

#     @validate_arguments(config=dict(arbitrary_types_allowed=True))
#     def _evaluate(
#         self,
#         X: DataLoader,
#         X_syn: DataLoader,
#         causal_graph_GT: Optional[Any] = None,
#     ) -> Dict:
#         X_ = np.ascontiguousarray(X.numpy().reshape(len(X), -1))
#         X_syn_ = np.ascontiguousarray(X_syn.numpy().reshape(len(X_syn), -1))

#         shd_list = []
#         precision_list = []
#         recall_list = []
#         f1_list = []
#         for _ in range(self.num_boostrapping):
#             # === Randomly sample the data for faster evaluation ===
#             idx = np.random.choice(len(X_), min(len(X_), len(X_syn_), 1000), replace=False)

#             # === Prepare the causal graph of the original data ===
#             if causal_graph_GT is None:
#                 # Derive the causal graph of the original data
#                 causal_graph_GT = self.causal_discovery(X_[idx, :])

#             # === Prepare the causal graph of the synthetic data ===
#             causal_graph_syn = self.causal_discovery(X_syn_[idx, :])

#             # === Compute the Structural Hamming Distance ===
#             shd_dict = self.compute_shd(causal_graph_GT, causal_graph_syn)
#             shd_list.append(shd_dict["shd"])
#             precision_list.append(shd_dict["precision"])
#             recall_list.append(shd_dict["recall"])
#             f1_list.append(shd_dict["f1"])

#         return {
#             "score": np.mean(shd_list),
#             "precision": np.mean(precision_list),
#             "recall": np.mean(recall_list),
#             "f1": np.mean(f1_list),
#         }

#     def causal_discovery(self, X: DataLoader) -> Any:
#         self.cd_model = StructuralHammingDistance.causal_discovery_method_handler(self.causal_discovery_method)
#         self.cd_model.fit(X)

#         return self.cd_model.adjacency_matrix_

#     @staticmethod
#     def causal_discovery_method_handler(causal_discovery_method: str) -> Any:
#         match causal_discovery_method:
#             case "lingam":
#                 return lingam.DirectLiNGAM()
#             case _:
#                 raise ValueError(f"Unknown causal discovery method: {causal_discovery_method}")

#     @staticmethod
#     def compute_shd(causal_graph_GT: Any, causal_graph_syn: Any) -> int:
#         # === Compute the Structural Hamming Distance ===
#         shd = SHD(((causal_graph_syn != 0).T) * 1.0, ((causal_graph_GT != 0).T) * 1.0)

#         # === Compute the precision, recall, and F1 score ===
#         _, curve = precision_recall(((causal_graph_syn != 0).T) * 1.0, ((causal_graph_GT != 0).T) * 1.0)
#         precision, recall = curve[1]
#         f1 = 2 * precision * recall / (precision + recall)

#         return {
#             "shd": shd,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#         }
