import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def load_and_preprocess_data():
    # 데이터 로드
    pitches = pd.read_csv('data/pitches.csv')
    atbats = pd.read_csv('data/atbats.csv')
    games = pd.read_csv('data/games.csv')

    # 데이터 병합
    data = pd.merge(pitches, atbats, on='ab_id')  # 'ab_id'는 예시, 실제 열 이름 확인 필요
    data = pd.merge(data, games, on='g_id')  # 'g_id'는 예시, 실제 열 이름 확인 필요

    # 필요 없는 열 제거
    data.drop(['g_id', 'ab_id'], axis=1, inplace=True)  # 'g_id', 'ab_id'는 예시, 실제 열 이름 확인 필요

    # 결측치 처리
    data.fillna(method='ffill', inplace=True)

    # 문자열 열 인코딩
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])

    # 특성과 레이블 분리
    X = data.drop('pitch_type', axis=1)
    y = data['pitch_type']

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, le

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    logging.info("모델 초기화 및 학습 시작")
    
    # 모델 구성
    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(set(y_train)), activation='softmax')  # 클래스 수에 맞게 출력 레이어 설정
    ])
    
    # 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # 모델 학습
    logging.info("모델 학습 진행")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    
    # 예측
    logging.info("예측 시작")
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    
    # 평가
    logging.info("모델 평가 시작")
    accuracy = accuracy_score(y_test, y_pred_classes)
    report = classification_report(y_test, y_pred_classes)
    
    return model, accuracy, report