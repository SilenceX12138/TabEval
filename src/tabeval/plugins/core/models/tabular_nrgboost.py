# stdlib
from abc import ABCMeta
from typing import Any

# third party
import pandas as pd
from pydantic import validate_call

try:
    # third party
    from nrgboost import Dataset, NRGBooster
except ImportError:
    raise ImportError(
        """
nrgboost is not installed. Please install it with pip install nrgboost.
"""
    )

# tabeval relative
from .tabular_encoder import TabularEncoder


class TabularNRGBoost(metaclass=ABCMeta):
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        num_trees: int = 200,
        shrinkage: float = 0.15,
        line_search: bool = True,
        max_leaves: int = 256,
        max_ratio_in_leaf: float = 2.0,
        min_data_in_leaf: float = 0.0,
        initial_uniform_mixture: float = 0.1,
        categorical_split_one_vs_all: bool = False,
        feature_frac: float = 1.0,
        splitter: str = "best",
        num_model_samples: int = 80_000,
        p_refresh: float = 0.1,
        num_chains: int = 16,
        burn_in: int = 100,
        encoder_max_clusters: int = 20,
        encoder_whitelist: list = [],
        **kwargs: Any,
    ):
        """
        .. inheritance-diagram:: tabeval.plugins.core.models.tabular_nrgboost.TabularNRGBoost
        :parts: 1
        """
        super(TabularNRGBoost, self).__init__()

        self.model_params = {
            "num_trees": num_trees,
            "shrinkage": shrinkage,
            "line_search": line_search,
            "max_leaves": max_leaves,
            "max_ratio_in_leaf": max_ratio_in_leaf,
            "min_data_in_leaf": min_data_in_leaf,
            "initial_uniform_mixture": initial_uniform_mixture,
            "categorical_split_one_vs_all": categorical_split_one_vs_all,
            "feature_frac": feature_frac,
            "splitter": splitter,
            "num_model_samples": num_model_samples,
            "p_refresh": p_refresh,
            "num_chains": num_chains,
            "burn_in": burn_in,
        }

        self.encoder = TabularEncoder(max_clusters=encoder_max_clusters, whitelist=encoder_whitelist)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(X)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def decode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.inverse_transform(X)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        X: pd.DataFrame,
    ) -> Any:
        # === Encode the data ===
        self.encoder = self.encoder.fit(X)
        X = self.encode(X)
        self.columns = X.columns

        # NRGBoost also encodes/decodes the data internally
        train_ds = Dataset(X)

        # Fit the model
        self.model = NRGBooster.fit(train_ds, self.model_params, seed=42)

        return self

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: int,
    ) -> pd.DataFrame:
        # NRGBoost return DataFrame by default
        samples = self.model.sample(count, seed=42)

        return self.decode(pd.DataFrame(samples, columns=self.columns))
