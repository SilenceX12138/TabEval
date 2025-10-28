"""
Reference: "Adversarial random forests for density estimation and generative modeling" Authors: David S. Watson, Kristin Blesch, Jan Kapar, and Marvin N. Wright
"""

# stdlib
from pathlib import Path
from typing import Any, List, Union

# third party
import pandas as pd
import torch

# Necessary packages
from pydantic import validate_call

# tabeval absolute
from tabeval.plugins.core.dataloader import DataLoader
from tabeval.plugins.core.distribution import Distribution
from tabeval.plugins.core.models.tabular_nrgboost import TabularNRGBoost
from tabeval.plugins.core.plugin import Plugin
from tabeval.plugins.core.schema import Schema
from tabeval.utils.constants import DEVICE


class NRGBoostPlugin(Plugin):
    """
    .. inheritance-diagram:: tabeval.plugins.generic.plugin_arf.ARFPlugin
        :parts: 1

    Args:


    Example:
        >>> from sklearn.datasets import load_iris
        >>> from tabeval.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("nrgboost")
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)

    """

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        # fitting hyperparameters
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
        # sampling hyperparameters
        num_model_samples: int = 80_000,
        p_refresh: float = 0.1,
        num_chains: int = 16,
        burn_in: int = 100,
        # core plugin arguments
        device: Union[str, torch.device] = DEVICE,
        random_state: int = 0,
        sampling_patience: int = 500,
        workspace: Path = Path("logs/tabeval_workspace"),
        compress_dataset: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        .. inheritance-diagram:: tabeval.plugins.generic.plugin_nrgboost.NRGBoostPlugin
        :parts: 1

        NRGBoost is a gradient boosting framework that uses a novel approach to model the data distribution.

        Args:
            Fitting parameters
            - num_trees (int): Total number of trees to fit. Defaults to 200.
            - shrinkage (float): Constant scaling factor for each step. Defaults to 0.15.
            - line_search (bool): Use a line search to find optimal step size.
                Shrinkage factor is applied on top. Defaults to True.
            - max_leaves (int): Maximum number of leaves for each tree. Defaults to 256.
            - max_ratio_in_leaf (float): Maximum ratio of data / model data per leaf. Defaults to 2.
            - min_data_in_leaf (float): Minimum number of data points per leaf. Defaults to 0.
            - initial_uniform_mixture (float): Mixture coeficient for the starting point of boosting:
                - 0 means starting from the product of training marginals.
                - 1 means starting from a uniform distribution.
                Defaults to 0.1.
            - categorical_split_one_vs_all (bool): If True, categorical splits are always one vs all.
                Otherwise they are many vs many. Defaults to False.
            - feature_frac (float): Fraction of features to randomly consider for splitting each node. Defaults to 1.
            - splitter (str): Determines how trees are grown. "best" is best first
                and "depth" is breadth first. Defaults to "best".

            Sampling parameters
            - num_model_samples (int): Defaults to 80_000.
            - p_refresh (float): Fraction of samples to independently refresh at each round.
              Defaults to 0.1.
            - num_chains (int): Number of chains used to draw samples. Defaults to 16.
            - burn_in (int): Number of samples to burn at start of each chain. Defaults to 100.
            - num_threads (int): Defaults to 0.

            Plugin parameters
            device: Union[str, torch.device] = tabeval.utils.constants.DEVICE
                The device that the model is run on. Defaults to "cuda" if cuda is available else "cpu".
            random_state: int = 0
                random_state used. Defaults to 0.
            sampling_patience: int = 500
                Max inference iterations to wait for the generated data to match the training schema. Defaults to 500.
            workspace: Path
                Path for caching intermediary results. Defaults to Path("logs/tabeval_workspace").
            compress_dataset: bool. Default = False
                Drop redundant features before training the generator. Defaults to False.
            dataloader_sampler: Any = None
                Optional sampler for the dataloader. Defaults to None.
        """
        super().__init__(
            device=device,
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
            **kwargs,
        )
        self.num_trees = num_trees
        self.shrinkage = shrinkage
        self.line_search = line_search
        self.max_leaves = max_leaves
        self.max_ratio_in_leaf = max_ratio_in_leaf
        self.min_data_in_leaf = min_data_in_leaf
        self.initial_uniform_mixture = initial_uniform_mixture
        self.categorical_split_one_vs_all = categorical_split_one_vs_all
        self.feature_frac = feature_frac
        self.splitter = splitter
        self.num_model_samples = num_model_samples
        self.p_refresh = p_refresh
        self.num_chains = num_chains
        self.burn_in = burn_in

    @staticmethod
    def name() -> str:
        return "nrgboost"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "NRGBoostPlugin":
        """_summary_

        Args:
            X (DataLoader): _description_
            kwargs (Any): keyword arguments passed on to an SKLearn RandomForestClassifier

        Raises:
            NotImplementedError: _description_

        Returns:
            NRGBoostPlugin: _description_
        """
        self.model = TabularNRGBoost(
            self.num_trees,
            self.shrinkage,
            self.line_search,
            self.max_leaves,
            self.max_ratio_in_leaf,
            self.min_data_in_leaf,
            self.initial_uniform_mixture,
            self.categorical_split_one_vs_all,
            self.feature_frac,
            self.splitter,
            self.num_model_samples,
            self.p_refresh,
            self.num_chains,
            self.burn_in,
            **kwargs,
        )
        if "cond" in kwargs:
            if kwargs["cond"] is not None:
                raise NotImplementedError(
                    "conditional generation is not currently available for the NRGBoost plugin."
                )
            kwargs.pop("cond")

        self.model.fit(X.dataframe(), **kwargs)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        if "cond" in kwargs and kwargs["cond"] is not None:
            raise NotImplementedError(
                "conditional generation is not currently available for the NRGBoost plugin."
            )

        return self._safe_generate(self.model.generate, count, syn_schema)


plugin = NRGBoostPlugin
