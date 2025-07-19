# 데이터 전처리 및 sentiment 계산

import pandas as pd

def load_data(path='data/dataset.json'):
    df = pd.read_json(path)

    def safe_get_sentiment(value, key):
        if isinstance(value, dict) and key in value:
            return value[key]
        return 0  # 또는 None / np.nan

    df['positive'] = df['sentiment'].apply(lambda x: safe_get_sentiment(x, 'positive'))
    df['negative'] = df['sentiment'].apply(lambda x: safe_get_sentiment(x, 'negative'))

    return df
