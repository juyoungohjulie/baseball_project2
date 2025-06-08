import logging
from data_preprocessing import load_and_preprocess_data
from feature_engineering import feature_engineering
from training import train_and_evaluate_model, preprocess_data, preprocess_data_LSTM, train_and_evaluate_model_LSTM, train_and_evaluate_model_CNN, preprocess_data_transformer, train_and_evaluate_model_transformer
from training import (
    train_and_evaluate_model_CNN, 
    train_and_evaluate_model_LSTM,
    train_and_evaluate_model_transformer 
)
from visualization import plot_confusion_matrix , plot_training_history #, plot_class_distribution
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("데이터 로드 및 전처리 시작")
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data()

    logging.info("특성 엔지니어링 시작")
    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)

    logging.info("데이터 전처리 시작")
    X_train, X_test = preprocess_data(X_train, X_test)
    # LSTM
    # X_train, X_test = preprocess_data_LSTM(X_train, X_test)
    # CNN
    # X_train, X_test = preprocess_data_CNN(X_train, X_test)
    # Transformer
    # X_train, X_test = preprocess_data_transformer(X_train, X_test)

    logging.info("모델 학습 및 평가 시작")
    model, accuracy, report, history = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    # LSTM
    # model, accuracy, report, history = train_and_evaluate_model_LSTM(X_train, X_test, y_train, y_test)
    # Transformer
    # model, accuracy, report, history = train_and_evaluate_model_transformer(X_train, X_test, y_train, y_test)

    # CNN
    # model, accuracy, report, history = train_and_evaluate_model_CNN(X_train, X_test, y_train, y_test)

    logging.info(f"모델 정확도: {accuracy}")
    logging.info(f"분류 보고서:\n{report}")

    logging.info("결과 시각화 시작")
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)


    # # 1. Confusion Matrix 저장
    # logging.info("시각화 결과가 results/ 폴더에 저장되었습니다.")
    # plot_confusion_matrix(y_test, y_pred_classes)

    # # 2. Training History 저장
    # plot_training_history(history, model_name='Baseline model', save_dir='results')

    ####For LSTM
    # 1. Confusion Matrix 저장
    logging.info("시각화 결과가 results/ 폴더에 저장되었습니다.")
    plot_confusion_matrix(y_test, y_pred_classes, save_path='results/confusion_matrix_baseline.png')

    # plot_confusion_matrix(y_test, y_pred_classes, save_path='results/confusion_matrix_LSTM.png')
    # plot_confusion_matrix(y_test, y_pred_classes, save_path='results/confusion_matrix_CNN.png')
    # plot_confusion_matrix(y_test, y_pred_classes, save_path='results/confusion_matrix_transformer.png')

    # 2. Training History 저장
    plot_training_history(history, model_name='baseline', save_dir='results/baseline')

    # plot_training_history(history, model_name='LSTM', save_dir='results/LSTM')
    # plot_training_history(history, model_name='CNN', save_dir='results/CNN')
    # plot_training_history(history, model_name='Transformer', save_dir='results/transformer')

if __name__ == "__main__":
    main()