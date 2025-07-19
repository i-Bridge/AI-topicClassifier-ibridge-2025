from scripts.sentiment import display_sentiment
from scripts.model import train_model, predict_category

if __name__ == "__main__":
    # sentiment 계산 및 출력
    display_sentiment()

    # 텍스트 분류 모델 학습
    train_model()

    # 예시 예측
    text = "가족 관계에 대해 이야기해 주세요."
    category = predict_category(text)
    print(f"Predicted Category: {category}")
