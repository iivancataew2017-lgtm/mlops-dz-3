"""Сдвиг трёхмесячного окна обучения на один месяц вперёд."""

import logging
from datetime import date

import pandas as pd

from pipeline.settings import CURRENT_DATASET, PENDING_WINDOW

logger = logging.getLogger(__name__)


def slide_training_window(batch_info: dict) -> dict:
    """Добавляет новый месяц к текущему окну и выбрасывает самый старый.

    Читает current.parquet и parquet батча из batch_info. Новое окно пишет
    в pending_window.parquet и не заменяет current.parquet: замена происходит
    только после публикации модели, чтобы повтор задачи не сдвинул окно дважды.
    Возвращает путь к новому окну и месяц добавленного батча.
    """
    if not CURRENT_DATASET.exists():
        raise FileNotFoundError(f"Нет исходного датасета {CURRENT_DATASET}")

    batch_month = date.fromisoformat(batch_info["batch_month"])
    current = pd.read_parquet(CURRENT_DATASET)
    batch = pd.read_parquet(batch_info["batch_path"])
    current["transaction_time"] = pd.to_datetime(current["transaction_time"])
    batch["transaction_time"] = pd.to_datetime(batch["transaction_time"])

    month = pd.Period(batch_month, freq="M")
    batch_months = batch["transaction_time"].dt.to_period("M")
    if batch.empty or not (batch_months == month).all():
        raise ValueError("Батч должен содержать только указанный календарный месяц")

    current_months = current["transaction_time"].dt.to_period("M")
    current = current.loc[current_months.isin([month - 2, month - 1])]
    combined = pd.concat([current, batch], ignore_index=True)
    months = combined["transaction_time"].dt.to_period("M")
    expected_months = {month - 2, month - 1, month}
    if set(months.unique()) != expected_months:
        raise ValueError("Окно должно содержать три последовательных календарных месяца")

    oldest = month - 3
    updated = combined.sort_values("transaction_time")
    updated.to_parquet(PENDING_WINDOW, index=False)

    logger.info(
        "Окно после сдвига: %s строк, %s — %s. Удалён месяц %s.",
        len(updated),
        updated["transaction_time"].min(),
        updated["transaction_time"].max(),
        oldest,
    )
    return {
        "dataset_path": str(PENDING_WINDOW),
        "batch_month": batch_month.isoformat(),
    }
