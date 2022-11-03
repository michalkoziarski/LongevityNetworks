import argparse
from pathlib import Path

import pandas as pd
from stellargraph.layer import GCN, LinkEmbedding
from stellargraph.mapper import FullBatchLinkGenerator
from tensorflow import keras

from graph import load_splits


N_RUNS = 100

OUT_PATH = Path("results_multiseed")


def train_predict(split, epochs=5000):
    (G_train, edge_ids_train, edge_labels_train), (
        G_test,
        edge_ids_test,
        edge_labels_test,
    ) = split

    train_gen = FullBatchLinkGenerator(G_train, method="gcn")
    train_flow = train_gen.flow(edge_ids_train, edge_labels_train)
    test_gen = FullBatchLinkGenerator(G_test, method="gcn")
    test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

    gcn = GCN(
        layer_sizes=[256, 256, 256, 256],
        activations=["relu", "relu", "relu", "relu"],
        generator=train_gen,
        dropout=0.25,
    )

    x_inp, x_out = gcn.in_out_tensors()

    prediction = LinkEmbedding(activation="tanh", method="ip")(x_out)
    prediction = keras.layers.Reshape((-1,))(prediction)

    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.binary_crossentropy,
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.AUC(),
        ],
    )

    model.fit(
        train_flow, epochs=epochs, validation_data=test_flow, verbose=0, shuffle=True
    )

    y_prob = model.predict(test_flow)[0]

    rows = []

    for edge, p in zip(test_flow[0][0][1][0], y_prob):
        phn, gen = edge
        rows.append([G_test.nodes()[gen], G_test.nodes()[phn], p])

    df = pd.DataFrame(rows, columns=["Gen", "Phn", "p"])

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-seed", type=int, required=True)

    args = parser.parse_args()

    OUT_PATH.mkdir(exist_ok=True, parents=True)

    dfs = []

    splits = load_splits(sample_test_negatives=False, seed=args.seed)

    for split in splits:
        dfs.append(train_predict(split))

    df = pd.concat(dfs)
    df.to_csv(OUT_PATH / f"results_{args.seed}.csv", index=False)
