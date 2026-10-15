# Домашнее задание 3

## Что сделано в работе (по пунктам задания)

1) `dags/retrain_catboost.py` запускает переобучение каждые 5 минут. Код шагов переобучения находится в `pipeline/retrain/`. Следующий запуск DAG начинается после завершения предыдущего.

2) `data_loader/app.py` загружает CSV в `labeled_transactions` и сохраняет исходное окно (января–марта 2019 года). Каждый следующий прогон добавляет новый месяц и убирает самый старый: два ранних месяца идут в train, последний в test.

3) В `fraud_detector/src/preprocessing.py` добавлены признаки времени суток, дня недели, ночных покупок, логарифмы суммы и населения. Обучение и скоринг используют одну функцию `run_preproc`.

4) `evaluate_model.py` сравнивает кандидата и действующую модель на одном проверочном месяце. Для публикации нужны одновременно `ROC-AUC кандидата >= ROC-AUC действующей модели` и `F1 кандидата >= F1 действующей модели`. Порог для F1 — `score > 0.5`.

5) `publish_model.py` сохраняет допущенную модель в общий том `model_registry` и обновляет `latest.txt`. Скорер загружает новую версию без перезапуска, а `scoring_writer/app.py` записывает её в `scores.model_version` вместе со скором.

6) `dags/supply_labeled_batch.py` берёт до 100000 случайных строк витрины, переносит время в следующий календарный месяц и дописывает батч. Повторная загрузка того же батча не добавляет его второй раз. Данные старше трёх лет относительно самого свежего месяца удаляются.

7) `dags/check_data_quality.py` отдельно проверяет текущий датасет, эти же проверки выполняются перед обучением. Проверяются состав колонок, непустой набор, пропуски, конечность чисел, допустимые диапазоны, непустые текстовые поля, бинарный target с обоими классами и корректное время. HTML сохраняется в `/opt/airflow/dq_reports`, рядом с каталогом моделей, в том числе при ошибках данных.

## Запуск

Нужны Docker Desktop с запущенным движком и Docker Compose v2. Репозиторий приватный: для клонирования нужна авторизация GitHub с доступом к нему. Все команды выполняются из каталога с `docker-compose.yaml`.

```bash
git clone https://github.com/iivancataew2017-lgtm/mlops-dz-3.git
cd mlops-dz-3
cp .env.example .env
```

Скачайте [датасет](https://www.kaggle.com/competitions/teta-ml-1-2025/data) и укажите в `.env` абсолютный путь к существующему `train.csv`. CSV не включён в репозиторий. Нужны исходные 18 колонок, данные с января 2019 и следующие месяцы без разрывов; в обучающей и проверочной выборках должны присутствовать оба класса `target`. Вид пути:

```dotenv
TRAIN_DATA_CSV="/Users/your_name/data/train.csv"
```
Запускаем докер:

```bash
docker compose up -d --build
```

Airflow доступен на [localhost:8081](http://localhost:8081), логин и пароль — `admin` / `admin`. Kafka UI — на [localhost:8080](http://localhost:8080). PostgreSQL доступен на `localhost:5432`. База, пользователь и пароль задаются в `.env`, в шаблоне — `fraud_db`, `fraud_user`, `fraud_pass`.

## Проверка и результат

В Airflow каждый DAG можно запустить кнопкой Trigger DAG. Включённые DAG также запускаются автоматически каждые 5 минут.

1) `check_data_quality` проверяет текущий датасет и сохраняет HTML-отчёт. В Graph у него одна задача `check_dataset`, внутри которой выполняются восемь проверок данных.

2) `retrain_catboost` сдвигает окно данных, обучает кандидата и сравнивает его с действующей моделью. В Graph у него восемь задач. В логе задачи `evaluate_model` показаны метрики обеих моделей и решение о публикации; при допуске `publish_model` переключает скорер на новую версию.

3) `supply_labeled_batch` добавляет до 100000 случайных строк витрины с датами следующего месяца и удаляет данные старше трёх лет. В Graph у него три задачи: подготовка батча, дозапись и очистка старых данных.

После завершения `check_data_quality` внутри контейнера Airflow появляется файл `/opt/airflow/dq_reports/data_quality_current.html`. В нём находится таблица проверок текущего датасета с результатами `PASS` или `FAIL`. Этот файл обновляется при каждом запуске `check_data_quality`.

Переобучение сохраняет отдельные отчёты для новых окон данных. Дата в имени обозначает последний месяц окна: например, `data_quality_2019-04-01.html` относится к февралю–апрелю 2019. Поэтому после нескольких запусков файлов может быть несколько. Это история проверок качества данных.

Для просмотра последней проверки текущего датасета скопируйте только `data_quality_current.html`. Выполните из каталога репозитория:

```bash
mkdir -p dq_reports
docker compose cp airflow:/opt/airflow/dq_reports/data_quality_current.html ./dq_reports/data_quality_current.html
```

Первая команда создаёт локальную папку, если её ещё нет. Вторая копирует один HTML-файл из сервиса `airflow` в эту папку. При повторном копировании обновляется тот же файл. Локальный путь — `dq_reports/data_quality_current.html`.

Для удобства можно открыть в ghrome:

```bash
open -a "Google Chrome" ./dq_reports/data_quality_current.html
```

Отчёты внутри контейнера теряются при его пересоздании, а локальные копии сохраняются. Для обновления локального отчёта надо повторить копирование после следующего запуска DAG.

Чтобы посмотреть окно обучения и версии, которыми скорились сообщения, можно подключится к PostgreSQL:

```bash
docker compose exec postgres psql -U fraud_user -d fraud_db
```

```sql
SELECT window_start, window_end, next_batch_month FROM training_state;

SELECT model_version, COUNT(*) AS transactions, MAX(created_at) AS last_score
FROM scores
GROUP BY model_version
ORDER BY last_score DESC;
```

## Как устроено переобучение

```text
extract_month_batch → slide_training_window → check_data_quality → time_train_test_split
→ preprocess_dataset → fit_catboost → evaluate_model → publish_model
```

Если следующего месяца ещё нет в витрине, переобучение пропускается. Новое окно сначала сохраняется отдельно, а `current.parquet` и `training_state` обновляются после оценки кандидата. При отклонении кандидата действующая модель остаётся прежней, но окно продвигается к следующему месяцу. Метрики сохраняются в `my_catboost_<version>.metrics.json` или `rejected_<version>.metrics.json`.

Начальная модель в репозитории обучена на январе–феврале 2019 и соответствует текущим 29 признакам. Для её воспроизведения после сборки образа замените путь к CSV и выполните (то есть можно любую csv-шку вставить с соответствующими колонками):

```bash
docker compose run --rm --no-deps --entrypoint python \
  -v "/absolute/path/train.csv:/data/train.csv:ro" \
  -v "./fraud_detector/models:/bootstrap" \
  airflow -m pipeline.retrain.bootstrap_model \
  --csv /data/train.csv --output /bootstrap/my_catboost.cbm
```

Команда перезаписывает `fraud_detector/models/my_catboost.cbm`. Пересборка образа автоматически не обновляет уже существующий том `model_registry`.

## Структура ДЗ-3

```text
./
├── docker-compose.yaml
├── .env.example
├── airflow/
├── dags/
│   ├── retrain_catboost.py
│   ├── supply_labeled_batch.py
│   └── check_data_quality.py
├── pipeline/
│   ├── db.py
│   ├── settings.py
│   ├── retrain/
│   └── supply/
├── fraud_detector/
│   ├── app/
│   ├── models/
│   └── src/
├── transaction_simulator/
├── scoring_writer/
├── data_loader/
└── README.md
```
