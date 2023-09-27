import argparse
import os
import pickle

from concept_erasure import OracleFitter, QuadraticFitter
from datasets import load_dataset

from mdl import MlpProbe, Sweep


def main(args):
    embeddings_seed = args.embeddings_seed  # None means not random
    ds_name = "atmallen/amazon_polarity_embeddings" + (
        f"_random{embeddings_seed}" if embeddings_seed is not None else ""
    )
    print(ds_name)
    ds_dict = load_dataset(ds_name)
    device = args.device
    n_train = 2**17
    seed = args.seed

    ds_dict = ds_dict.with_format("torch", columns=["embedding", "label"]).shuffle(
        seed=seed
    )
    num_classes = ds_dict["train"].features["label"].num_classes
    X_train = ds_dict["train"]["embedding"][:n_train]
    X_train = X_train / X_train.norm(dim=-1, keepdim=True)
    Y_train = ds_dict["train"]["label"][:n_train]

    for erasure_method in ["Q-LEACE", "None", "Linear"]:
        print(f"Erasure: {erasure_method}")

        if erasure_method != "None":
            fitter_class = (
                QuadraticFitter if erasure_method == "Q-LEACE" else OracleFitter
            )
            fitter = fitter_class.fit(X_train, Y_train)
            eraser = fitter.eraser
            X_train_ = eraser(X_train, Y_train)
        else:
            X_train_ = X_train.clone()

        sweep = Sweep(
            num_features=X_train_.shape[1],
            num_classes=num_classes,
            num_chunks=10,
            probe_cls=MlpProbe,
            val_frac=0.2,
            device=device,
            probe_kwargs=dict(
                num_layers=2,
            ),
        )
        result = sweep.run(X_train_.to(device), Y_train.to(device).to(float), seed=seed)
        print(result)
        out_path = os.path.join(
            args.out_dir, f"{erasure_method}_seed{seed}_on_{ds_name.split('/')[-1]}.pkl"
        )
        # pickle result
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved result to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embeddings-seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default="../data")
    args = parser.parse_args()
    main(args)
