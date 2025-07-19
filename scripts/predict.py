# scripts/predict.py

import joblib
import json

from scripts.preprocess import preprocess_text

# 학습된 모델 불러오기
model = joblib.load('models/category_classifier.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')

samples = [
    {"topic": "가족", "reaction": "부모님의 관심 부족에 대한 외로움과 슬픔."},
    {"topic": "가족", "reaction": "엄마에게 인정받고 싶은 마음, 미안함과 두려움이 섞인 감정"}
]

for sample in samples:
    text = preprocess_text(sample['topic'] + " " + sample['reaction'])
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    print(f"입력: {sample}")
    print(f"예측된 카테고리: {pred}\n")
