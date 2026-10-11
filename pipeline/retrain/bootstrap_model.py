import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from pipeline.db import LABELED_COLUMNS
from pipeline.retrain.fit_catboost import prepare_features
from pipeline.settings import CAT_FEATURES, FRAUD_SRC, LABEL


def bootstrap_model(csv_path, output_path):
    data = pd.read_csv(csv_path, usecols=LABELED_COLUMNS)
    data = data[LABELED_COLUMNS].copy()
    data["transaction_time"] = pd.to_datetime(
        data["transaction_time"], format="mixed"
    )
    data = data.loc[
        (data["transaction_time"] >= "2019-01-01") & (data["transaction_time"] < "2019-03-01")
    ].sort_values("transaction_time").reset_index(drop=True)
    data[LABEL] = pd.to_numeric(data[LABEL]).astype(int)

    if FRAUD_SRC not in sys.path:
        sys.path.insert(0, FRAUD_SRC)
    from preprocessing import run_preproc

    features, target = prepare_features(run_preproc(data))
    model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        thread_count=2,
        allow_writing_files=False,
        verbose=20,
    )
    model.fit(features, target, cat_features=CAT_FEATURES)

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as temporary_dir:
        temporary_model = Path(temporary_dir) / output_path.name
        model.save_model(str(temporary_model))
        temporary_model.replace(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = bootstrap_model(args.csv, args.output)
    print(f"Путь модели: {output_path}")


if __name__ == "__main__":
    main()
