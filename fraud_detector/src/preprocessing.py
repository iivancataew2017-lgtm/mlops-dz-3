import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_train_data():
    logger.info('Loading training data...')

    # Import Train dataset
    train = pd.read_csv('./train_data/train.csv')
    logger.info('Raw train data imported. Shape: %s', train.shape)

    # Add some features
    train = add_features(train)
    logger.info('Train data processed. Shape: %s', train.shape)

    return train


def haversine(lat1, lon1, lat2, lon2):
    rad = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0, 1)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return rad * c


def add_features(df):
    # Работаем с копией, чтобы не изменять исходный DataFrame.
    df_ = df.copy()
    # Убираем пробелы по краям категорий, пустые строки и пропуски заменяем на unknown.
    for column in ['merch', 'cat_id', 'gender', 'one_city', 'us_state', 'jobs']:
        values = df_[column].astype('string').str.strip()
        df_[column] = values.mask(values.eq(''), 'unknown').fillna('unknown').astype(str)
    # Приводим значения к числам; ошибки преобразования и бесконечности становятся NaN.
    for column in ['amount', 'post_code', 'lat', 'lon', 'population_city',
                   'merchant_lat', 'merchant_lon']:
        df_[column] = pd.to_numeric(df_[column], errors='coerce').replace(
            [np.inf, -np.inf], np.nan,
        )
    # Координаты за пределами допустимой широты и долготы считаем пропусками.
    for column, limit in [('lat', 90), ('lon', 180),
                          ('merchant_lat', 90), ('merchant_lon', 180)]:
        df_[column] = df_[column].where(df_[column].between(-limit, limit))
    # Запоминаем отсутствие хотя бы одной координаты до заполнения пропусков нулями.
    df_['location_missing'] = df_[['lat', 'lon', 'merchant_lat', 'merchant_lon']].isna().any(axis=1).astype(int)
    # Рассчитываем расстояние между клиентом и продавцом в километрах.
    df_['haversine'] = haversine(df_['lat'], df_['lon'],
                                 df_['merchant_lat'], df_['merchant_lon'])
    # Разбираем время в UTC и извлекаем календарные признаки; неверные даты становятся NaT.
    df_["transaction_time"] = pd.to_datetime(df_["transaction_time"], errors='coerce', format='mixed', utc=True)
    df_["year"] = df_["transaction_time"].dt.year
    df_["month"] = df_["transaction_time"].dt.month
    df_["day"] = df_["transaction_time"].dt.day
    df_["hour"] = df_["transaction_time"].dt.hour
    df_["minute"] = df_["transaction_time"].dt.minute
    df_["weekend"] = (df_["transaction_time"].dt.dayofweek > 4).astype(int)
    # Номер дня недели: понедельник — 0, воскресенье — 6.
    df_['weekday'] = df_['transaction_time'].dt.dayofweek
    # Синус и косинус сохраняют близость соседних часов и дней на границе цикла.
    hour = df_['hour'] + df_['minute'] / 60
    df_['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df_['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df_['weekday_sin'] = np.sin(2 * np.pi * df_['weekday'] / 7)
    df_['weekday_cos'] = np.cos(2 * np.pi * df_['weekday'] / 7)
    # Ночными считаем операции с 22:00 включительно до 06:00.
    df_['is_night'] = ((df_['hour'] < 6) | (df_['hour'] >= 22)).astype(int)
    # log(1 + x) сжимает масштаб больших сумм и населения; отрицательные значения обрезаем до нуля.
    df_['amount_log1p'] = np.log1p(df_['amount'].clip(lower=0))
    df_['population_log1p'] = np.log1p(df_['population_city'].clip(lower=0))
    # Для отсутствующей или неверной даты помечаем календарные признаки значением -1.
    calendar = ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'weekend', 'is_night']
    df_.loc[df_['transaction_time'].isna(), calendar] = -1
    # Оставшиеся числовые пропуски и бесконечности заменяем нулём, не затрагивая целевую метку target.
    for column in df_.select_dtypes(include='number').columns:
        if column != 'target':
            df_[column] = df_[column].replace([np.inf, -np.inf], np.nan).fillna(0)
    # После извлечения признаков удаляем исходное время, имена и адрес; отсутствие столбцов допустимо.
    return df_.drop(columns=["transaction_time", "name_1", "name_2", "street"], errors='ignore')


# Main preprocessing function
def run_preproc(input_df):
    # Add features
    output_df = add_features(input_df)
    logger.info('Added new features. Output shape: %s', output_df.shape)

    # Return resulting dataset
    return output_df
