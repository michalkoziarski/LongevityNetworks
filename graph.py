import itertools
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from stellargraph import StellarGraph

DATA_PATH = Path(__file__).parent / "data"


def _load_gen_df() -> pd.DataFrame:
    df_gen = pd.read_csv(DATA_PATH / "GlobalGraph_PPI.csv").drop_duplicates()
    df_gen["Type"] = "G2G"

    phenotypes = _get_phenotypes(_load_phn_df())

    df_gen = df_gen[
        ~(df_gen["Nod_A"].isin(phenotypes) | df_gen["Nod_B"].isin(phenotypes))
    ]

    return df_gen


def _load_phn_df(drop_disconnected: bool = True) -> pd.DataFrame:
    df_phn = pd.read_csv(DATA_PATH / "Edg_GenPhn.csv")
    df_phn["Type"] = "P2G"

    if drop_disconnected:
        df_phn = df_phn[~df_phn["Phn"].isin(["D1098", "D1158", "D1597", "D1089"])]

    df_phn = pd.DataFrame(
        {"Nod_A": df_phn["Phn"], "Nod_B": df_phn["Gen"], "Type": df_phn["Type"]}
    )

    return df_phn


def _merge_dfs(df_gen: pd.DataFrame, df_phn: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([df_gen, df_phn]).reset_index(drop=True)

    return df


def _get_genes(df_gen: pd.DataFrame, df_phn: pd.DataFrame) -> np.ndarray:
    return pd.concat([df_gen["Nod_A"], df_gen["Nod_B"], df_phn["Nod_B"]]).unique()


def _get_phenotypes(df_phn: pd.DataFrame) -> np.ndarray:
    return df_phn[df_phn["Type"] == "P2G"]["Nod_A"].unique()


def _get_positive_negative_edges(
    df: pd.DataFrame, genes: np.ndarray, phenotypes: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dx = pd.DataFrame(
        list(itertools.product(phenotypes, genes)), columns=["Nod_A", "Nod_B"]
    )
    dx["Type"] = "P2G"

    marked_edges = dx.merge(df[df["Type"] == "P2G"], how="left", indicator=True)

    positive_edges = marked_edges[marked_edges["_merge"] == "both"].drop(
        "_merge", axis=1
    )
    negative_edges = marked_edges[marked_edges["_merge"] == "left_only"].drop(
        "_merge", axis=1
    )

    return positive_edges, negative_edges


def _load_go_terms_df(n: int = -1) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH / "GoDataset_1000.csv", index_col=0)
    df = df[df.sum().sort_values(ascending=False).index[:n]]

    return df


def _make_graph(
    df_gen: pd.DataFrame,
    df_phn: pd.DataFrame,
    genes: Optional[np.ndarray] = None,
    phenotypes: Optional[np.ndarray] = None,
    homogenous: bool = True,
    use_go_terms: bool = False,
    n_go_terms: int = -1,
) -> StellarGraph:
    df = _merge_dfs(df_gen, df_phn)

    if genes is None:
        genes = _get_genes(df_gen, df_phn)

    if phenotypes is None:
        phenotypes = _get_phenotypes(df_phn)

    if homogenous:
        x_gen = np.zeros((len(genes), len(phenotypes) + 1))
        x_gen[:, -1] = 1
        x_phn = np.zeros((len(phenotypes), len(phenotypes) + 1))

        for i, _ in enumerate(phenotypes):
            x_phn[i, i] = 1

        if use_go_terms:
            df_go = _load_go_terms_df(n_go_terms)
            x_go = df_go.reindex(genes, fill_value=0).values.astype(int)

            x_gen = np.concatenate((x_gen, x_go), axis=1)
            x_phn = np.concatenate(
                (x_phn, np.zeros((x_phn.shape[0], x_go.shape[1]))), axis=1
            )

        graph = StellarGraph(
            pd.concat(
                [
                    pd.DataFrame(x_gen, index=genes),
                    pd.DataFrame(x_phn, index=phenotypes),
                ]
            ),
            edges=df.drop("Type", axis=1),
            source_column="Nod_A",
            target_column="Nod_B",
        )
    else:
        graph = StellarGraph(
            {
                "genes": pd.DataFrame(index=genes),
                "phenotypes": pd.DataFrame(index=phenotypes),
            },
            edges=df,
            source_column="Nod_A",
            target_column="Nod_B",
            edge_type_column="Type",
        )

    return graph


def load_full_graph(
    drop_disconnected: bool = True,
    homogenous: bool = True,
    use_go_terms: bool = False,
    n_go_terms: int = -1,
) -> StellarGraph:
    df_gen = _load_gen_df()
    df_phn = _load_phn_df(drop_disconnected)

    graph = _make_graph(
        df_gen,
        df_phn,
        homogenous=homogenous,
        use_go_terms=use_go_terms,
        n_go_terms=n_go_terms,
    )

    return graph


def _extract_graph_edges_labels(
    df: pd.DataFrame,
    positive_edges: pd.DataFrame,
    negative_edges: pd.DataFrame,
    genes: Optional[np.ndarray] = None,
    phenotypes: Optional[np.ndarray] = None,
    homogenous: bool = True,
    use_go_terms: bool = False,
    n_go_terms: int = -1,
) -> tuple:
    df_gen = df[df["Type"] == "G2G"]
    df_phn = df[df["Type"] == "P2G"]

    graph = _make_graph(
        df_gen,
        df_phn,
        genes,
        phenotypes,
        homogenous=homogenous,
        use_go_terms=use_go_terms,
        n_go_terms=n_go_terms,
    )
    edges = np.concatenate(
        [
            positive_edges[["Nod_A", "Nod_B"]].values,
            negative_edges[["Nod_A", "Nod_B"]].values,
        ]
    )
    labels = np.concatenate(
        [
            positive_edges["Label"].values,
            negative_edges["Label"].values,
        ]
    )

    return graph, edges, labels


def load_splits(
    n_splits: int = 5,
    train_edges_size: int = 0.2,
    sample_train_negatives: bool = True,
    sample_test_negatives: bool = False,
    homogenous: bool = True,
    use_go_terms: bool = False,
    n_go_terms: int = -1,
    drop_disconnected: bool = True,
    seed: int = 0,
) -> list:
    df_gen = _load_gen_df()
    df_phn = _load_phn_df(drop_disconnected)
    df = _merge_dfs(df_gen, df_phn)

    genes = _get_genes(df_gen, df_phn)
    phenotypes = _get_phenotypes(df_phn)

    positive_edges, negative_edges = _get_positive_negative_edges(df, genes, phenotypes)
    positive_edges["Label"] = 1
    negative_edges["Label"] = 0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train_edges = [[] for _ in range(n_splits)]
    test_edges = [[] for _ in range(n_splits)]

    for i, (train_index, test_index) in enumerate(kf.split(positive_edges)):
        train_edges[i].append(
            train_test_split(
                positive_edges.iloc[train_index].copy(),
                test_size=train_edges_size,
                random_state=seed,
            )[1]
        )
        test_edges[i].append(positive_edges.iloc[test_index].copy())

    for i, (train_index, test_index) in enumerate(kf.split(negative_edges)):
        train_edges[i].append(negative_edges.iloc[train_index].copy())
        test_edges[i].append(negative_edges.iloc[test_index].copy())

        if sample_train_negatives:
            train_edges[i][1] = train_edges[i][1].sample(
                len(train_edges[i][0]), random_state=seed
            )
        else:
            train_edges[i][1] = train_test_split(
                train_edges[i][1],
                test_size=train_edges_size,
                random_state=seed,
            )[1]

        if sample_test_negatives:
            test_edges[i][1] = test_edges[i][1].sample(
                len(test_edges[i][0]), random_state=seed
            )

    splits = []

    for i in range(n_splits):
        positive_test_edges = test_edges[i][0]
        negative_test_edges = test_edges[i][1]

        test_df = pd.concat(
            [df, positive_test_edges.drop("Label", axis=1)]
        ).drop_duplicates(keep=False)

        test_graph, test_examples, test_labels = _extract_graph_edges_labels(
            test_df,
            positive_test_edges,
            negative_test_edges,
            genes=genes,
            phenotypes=phenotypes,
            homogenous=homogenous,
            use_go_terms=use_go_terms,
            n_go_terms=n_go_terms,
        )

        positive_train_edges = train_edges[i][0]
        negative_train_edges = train_edges[i][1]

        train_df = pd.concat(
            [test_df, positive_train_edges.drop("Label", axis=1)]
        ).drop_duplicates(keep=False)

        train_graph, train_examples, train_labels = _extract_graph_edges_labels(
            train_df,
            positive_train_edges,
            negative_train_edges,
            genes=genes,
            phenotypes=phenotypes,
            homogenous=homogenous,
            use_go_terms=use_go_terms,
            n_go_terms=n_go_terms,
        )

        splits.append(
            (
                (train_graph, train_examples, train_labels),
                (test_graph, test_examples, test_labels),
            )
        )

    return splits


if __name__ == "__main__":
    print(load_full_graph().info())

    print("-----")

    (train_graph, train_examples, train_labels), (
        test_graph,
        test_examples,
        test_labels,
    ) = load_splits()[0]

    print("Test graph:")
    print(f"\t# edges: {len(test_examples)}.")
    print(f"\tLabel distribution: {Counter(test_labels)}.")
    print(test_graph.info())

    print("-----")

    print("Train graph:")
    print(f"\t# edges: {len(train_examples)}.")
    print(f"\tLabel distribution: {Counter(train_labels)}.")
    print(train_graph.info())
