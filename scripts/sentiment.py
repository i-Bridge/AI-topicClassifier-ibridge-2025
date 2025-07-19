# category별로 sentiment 평균 계산 및 출력
import pandas as pd
from scripts.preprocess import load_data

def calculate_sentiment():
    # 데이터 로드
    df = load_data()

    # category별 sentiment 평균 계산
    category_sentiment = df.groupby('category')[['positive', 'negative']].mean()
    
    return category_sentiment

def display_sentiment():
    # sentiment 계산
    sentiment = calculate_sentiment()
    print("Sentiment Average by Category:")
    print(sentiment)

# 프로그램 실행 시 실행
if __name__ == "__main__":
    display_sentiment()
