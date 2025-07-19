from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from scripts.preprocess import load_data

def train_model():
    # 데이터 로드
    df = load_data()
    
    # topic과 reaction을 합친 텍스트로 category 예측
    df['text'] = df['topic'] + " " + df['reaction']
    
    # X, y 준비
    X = df['text']
    y = df['category']
    
    # 텍스트 벡터화
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)
    
    # 훈련/테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    # 분류 모델 훈련
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 정확도 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # 모델 저장 (선택 사항)
    # import joblib
    # joblib.dump(model, 'category_predictor.pkl')

def predict_category(text):
    # 모델 로드 및 예측 (모델이 저장되었으면)
    # model = joblib.load('category_predictor.pkl')
    
    # 예시로 다시 모델 훈련
    df = load_data()
    df['text'] = df['topic'] + " " + df['reaction']
    X = df['text']
    y = df['category']
    
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_vec, y)
    
    # 예측
    input_vec = vectorizer.transform([text])
    category = model.predict(input_vec)
    
    return category[0]

if __name__ == "__main__":
    train_model()
    text = "가족 관계에 대해 이야기해 주세요."
    category = predict_category(text)
    print(f"Predicted Category: {category}")
