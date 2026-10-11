import html
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from airflow.exceptions import AirflowFailException

from pipeline.db import LABELED_COLUMNS
from pipeline.settings import CAT_FEATURES, CURRENT_DATASET, LABEL, MODELS_DIR

logger = logging.getLogger(__name__)


def _dataset_checks(data):
    checks = []

    def add_check(name, passed, details):
        checks.append({
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "details": str(details),
        })

    missing = [column for column in LABELED_COLUMNS if column not in data.columns]
    unexpected = [column for column in data.columns if column not in LABELED_COLUMNS]
    columns_ok = not missing and not unexpected and data.columns.is_unique
    add_check(
        "display колонок",
        columns_ok,
        f"отсутствуют: {missing}; лишние: {unexpected}; повторы: {data.columns.duplicated().sum()}",
    )
    add_check("Датасет не пустой", not data.empty, f"Строк: {len(data)}")
    if not columns_ok or data.empty:
        return checks

    null_count = int(data.isna().sum().sum())
    add_check("Нет пропусков", null_count == 0, f"Количество пропущенных значений: {null_count}")

    ranges = {
        "amount": (0, np.inf),
        "post_code": (0, 99999),
        "lat": (-90, 90),
        "lon": (-180, 180),
        "population_city": (0, np.inf),
        "merchant_lat": (-90, 90),
        "merchant_lon": (-180, 180),
    }

    numbers = data[list(ranges) + [LABEL]].apply(pd.to_numeric, errors="coerce").astype(float)
    invalid_numbers = int((~np.isfinite(numbers)).sum().sum())
    add_check("Корректные числовые значения", invalid_numbers == 0, f"Некорректных значений: {invalid_numbers}")

    invalid_ranges = []
    for column, (lower, upper) in ranges.items():
        if not numbers[column].between(lower, upper).all():
            invalid_ranges.append(column)
    add_check("Числа в допустимых диапазонах", not invalid_ranges, f"Числа с недопустимыми значениями: {invalid_ranges}")

    blank_columns = []
    for column in CAT_FEATURES + ["name_1", "name_2", "street"]:
        values = data[column].astype("string").fillna("").str.strip()
        if values.eq("").any():
            blank_columns.append(column)
    add_check("Поля заполнены", not blank_columns, f"Колонки с пустыми значениями: {blank_columns}")

    target = numbers[LABEL]
    add_check(
        "target ok",
        target.isin([0, 1]).all() and target.nunique() == 2,
        f"Количество строк по классам: {target.value_counts(dropna=False).to_dict()}",
    )

    times = pd.to_datetime(data["transaction_time"], errors="coerce", format="mixed", utc=True)
    invalid_times = int(times.isna().sum())
    add_check("Корректное время транзакций", invalid_times == 0, f"Количество некорректных дат: {invalid_times}")
    return checks


def check_dataset(window_info=None) -> dict:
    info = dict(window_info or {})
    dataset_path = Path(info.get("dataset_path", CURRENT_DATASET))
    report_dir = MODELS_DIR.parent / "dq_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_label = str(info.get("batch_month", "current")).replace("/", "_").replace("\\", "_")
    report_path = report_dir / f"data_quality_{report_label}.html"
    row_count = 0

    operation = "Чтение датасета"
    try:
        data = pd.read_parquet(dataset_path)
        row_count = len(data)
        operation = "Проверка датасета"
        checks = _dataset_checks(data)
    except Exception as exc:
        checks = [{
            "name": operation,
            "status": "FAIL",
            "details": f"{type(exc).__name__}: {exc}",
        }]

    failed = [check["name"] for check in checks if check["status"] == "FAIL"]
    status = "FAIL" if failed else "PASS"
    table = pd.DataFrame(checks)
    table.columns = ["Проверка", "Результат", "Подробности"]

    document = f"""
    <meta charset="utf-8">
    <h1>Качество данных: {status}</h1>
    <p>Датасет: {html.escape(str(dataset_path))}</p>
    <p>Строк: {row_count}; проверок: {len(checks)}</p>
    {table.to_html(index=False, escape=True)}
    """
    temporary_path = report_path.with_suffix(".tmp")
    temporary_path.write_text(document, encoding="utf-8")
    temporary_path.replace(report_path)

    logger.info("Качество данных %s: строк %s, отчёт %s", status, row_count, report_path)
    if failed:
        raise AirflowFailException(f"Проверки качества данных не пройдены: {', '.join(failed)}. Отчёт: {report_path}")
    return {
        **info,
        "dataset_path": str(dataset_path),
        "dq_report_path": str(report_path),
        "dq_status": status,
        "rows": int(row_count),
        "dq_checks": int(len(checks)),
    }
