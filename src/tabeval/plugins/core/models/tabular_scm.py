# stdlib
from abc import ABCMeta
from typing import Any

# third party
import lingam
import networkx as nx
import numpy as np
import pandas as pd
from causallearn.search.FCMBased import lingam as cl_lingam
from pydantic import validate_call

# tabeval relative
from .tabular_encoder import TabularEncoder


class TabularSCM(metaclass=ABCMeta):
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        cd_method: str,
        encoder_max_clusters: int = 20,
        encoder_whitelist: list = [],
        **kwargs: Any,
    ):
        """
        .. inheritance-diagram:: tabeval.plugins.core.models.tabular_scm.TabularSCM
        :parts: 1
        """
        super(TabularSCM, self).__init__()

        self.cd_method = cd_method
        match self.cd_method:
            case "direct-lingam":
                self.model = cl_lingam.DirectLiNGAM()
            case "lim":
                self.model = lingam.LiM()

        self.encoder = TabularEncoder(
            max_clusters=encoder_max_clusters, whitelist=encoder_whitelist
        )

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
        **kwargs: Any,
    ) -> Any:
        # === Encode the data ===
        self.encoder = self.encoder.fit(X)
        X = self.encode(X)
        self.num_nodes = X.shape[1]
        self.columns = X.columns

        # === Fit the model ===
        if self.cd_method in ["lim"]:
            dis_con = np.full((1, self.num_nodes), np.inf)
            for feature in range(self.num_nodes):
                # After one-hot encoding, the number of unique values of each categorical feature (binary) is 2
                if X.iloc[:, feature].nunique() == 2:
                    dis_con[0, feature] = 0  # 1:continuous;   0:discrete
                else:
                    dis_con[0, feature] = 1
            # Set only_global=True otherwise the 2nd stage of local search will be very time-consuming
            self.model.fit(X.values, dis_con, only_global=True)
        else:
            self.model.fit(X)

        # === Store the model's attributes ===
        self.adjacency_matrix = self.model.adjacency_matrix_
        if hasattr(self.model, "causal_order_"):
            self.causal_order = self.model.causal_order_
        else:
            self.causal_order = self.infer_causal_order()

        return self

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def infer_causal_order(
        self,
    ) -> list:
        # In causal adjacency matrix, adj[i, j] = 1 means i <- j (https://lingam.readthedocs.io/en/stable/tutorial/lingam.html)
        G = nx.DiGraph(self.adjacency_matrix.T)
        causal_order = list(nx.topological_sort(G))

        return causal_order

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: int,
    ) -> pd.DataFrame:
        samples = np.zeros((count, self.num_nodes))
        for feature in self.causal_order:
            samples[:, feature] = self.adjacency_matrix[feature, :].dot(
                samples.T
            ) + np.random.uniform(size=count)

        # Using self.columns to ensure the correct dtypes of the columns
        return self.decode(pd.DataFrame(samples, columns=self.columns))
