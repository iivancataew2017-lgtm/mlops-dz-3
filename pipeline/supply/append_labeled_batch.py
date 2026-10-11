"""Дозапись подготовленного месячного батча в размеченную витрину."""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from pipeline.db import copy_labeled_frame, get_conn, month_end_exclusive

logger = logging.getLogger(__name__)


def append_labeled_batch(batch_info: dict) -> dict:
    """Дописывает parquet батча в конец labeled_transactions.

    Читает путь из batch_info. Существующие строки не обновляет и не удаляет.
    Возвращает тот же словарь с числом записанных строк, чтобы следующая
    задача знала, какой месяц только что появился в витрине.
    """
    batch_path = Path(batch_info["batch_path"])
    if not batch_path.exists():
        raise FileNotFoundError(f"Нет файла батча для загрузки: {batch_path}")

    frame = pd.read_parquet(batch_path)
    month = date.fromisoformat(batch_info["batch_month"])
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM labeled_transactions
                WHERE transaction_time >= %s AND transaction_time < %s
                """,
                (month, month_end_exclusive(month)),
            )
            rows_in_db = cur.fetchone()[0]

        if rows_in_db == 0:
            copy_labeled_frame(conn, frame)
        elif rows_in_db != len(frame):
            raise RuntimeError("Месяц уже существует с другим числом строк")
    finally:
        conn.close()
    logger.info(
        "Месяц %s в витрине, строк: %s",
        batch_info["batch_month"],
        len(frame),
    )
    return {
        "batch_path": batch_info["batch_path"],
        "batch_month": batch_info["batch_month"],
        "rows": int(len(frame)),
    }
