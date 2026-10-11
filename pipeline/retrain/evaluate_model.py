import logging
import math
import os
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score

from pipeline.retrain.fit_catboost import prepare_features
from pipeline.settings import MODELS_DIR

logger = logging.getLogger(__name__)


def model_metrics(model, features, target):
    columns = model.feature_names_
    data = features[columns].copy()
    for index in model.get_cat_feature_indices():
        column = columns[index]
        data[column] = data[column].astype(str)
    probabilities = model.predict_proba(data, thread_count=2)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "f1": float(f1_score(target, probabilities > 0.5, zero_division=0)),
    }


def metrics_improved(new_metrics, current_metrics):
    for metrics in (new_metrics, current_metrics):
        for name in ("roc_auc", "f1"):
            value = metrics[name]
            if value is None or not math.isfinite(value):
                raise ValueError("Не удалось рассчитать ROC-AUC или F1 для сравнения моделей")
    return (
        new_metrics["roc_auc"] >= current_metrics["roc_auc"]
        and new_metrics["f1"] >= current_metrics["f1"]
    )


def evaluate_model(train_info: dict) -> dict:
    test = pd.read_parquet(train_info["test_proc_path"])
    features, target = prepare_features(test)

    new_model = CatBoostClassifier()
    new_model.load_model(train_info["staging_model"])

    latest = MODELS_DIR / "latest.txt"
    previous_version = "initial"
    if latest.exists():
        previous_version = latest.read_text(encoding="utf-8").strip()

    current_model_path = MODELS_DIR / f"my_catboost_{previous_version}.cbm"
    if previous_version == "initial" and not current_model_path.exists():
        current_model_path = Path(os.getenv("BUNDLED_MODEL_PATH", "/opt/airflow/initial_model.cbm"))
    current_model = CatBoostClassifier()
    current_model.load_model(str(current_model_path))

    new_metrics = model_metrics(new_model, features, target)
    current_metrics = model_metrics(current_model, features, target)
    promoted = metrics_improved(new_metrics, current_metrics)
    logger.info(
        "Новая модель: %s; текущая модель: %s; можно публиковать: %s",
        new_metrics, current_metrics, promoted,
    )
    return {
        **train_info,
        **new_metrics,
        "previous_version": previous_version,
        "champion_metrics": current_metrics,
        "promoted": promoted,
    }
