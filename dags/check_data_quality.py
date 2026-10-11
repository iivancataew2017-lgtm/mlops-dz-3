from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.retrain.check_data_quality import check_dataset


@dag(
    dag_id="check_data_quality",
    description="Проверяет качество текущего трёхмесячного датасета и сохраняет HTML-отчёт",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=5),
    default_args={"retries": 0},
    tags=["mlops", "data_quality"],
)
def check_data_quality():
    task(check_dataset, task_id="check_dataset")()


check_data_quality()
