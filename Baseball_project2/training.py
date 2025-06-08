import logging
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Reshape, BatchNormalization, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from tensorflow.keras.layers import GRU, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def preprocess_data_transformer(X_train, X_test):
    """Transformer 모델을 위한 데이터 전처리"""
    selector = VarianceThreshold(threshold=0.01)
    X_train = selector.fit_transform(X_train)
    X_test = selector.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = X_train.reshape(X_train.shape[0], -1, 1)
    X_test = X_test.reshape(X_test.shape[0], -1, 1)
    
    return X_train, X_test

def train_and_evaluate_model_transformer(X_train, X_test, y_train, y_test):
    logging.info("모델 초기화 및 학습 시작")
    
    from collections import Counter
    class_counts = Counter(y_train)
    print("Original class distribution:", dict(class_counts))
    
    min_samples = 30 
    augmented_X = list(X_train)
    augmented_y = list(y_train)
    
    for class_label, count in class_counts.items():
        if count <= 2:  
            samples_to_generate = min_samples - count
            synthetic_X, synthetic_y = create_synthetic_samples_for_lstm(
                X_train, y_train, 
                target_class=class_label, 
                num_samples=samples_to_generate
            )
            if len(synthetic_X) > 0:
                augmented_X.extend(synthetic_X)
                augmented_y.extend(synthetic_y)
    
    X_train_augmented = np.array(augmented_X)
    y_train_augmented = np.array(augmented_y)
    
    print("Augmented class distribution:", dict(Counter(y_train_augmented)))
    
    input_shape = (X_train_augmented.shape[1], X_train_augmented.shape[2])
    
    inputs = Input(shape=input_shape)
    
    attention_output = MultiHeadAttention(
        num_heads=2, 
        key_dim=32
    )(inputs, inputs, inputs)  
    
    x = LayerNormalization()(attention_output)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    
    # Global Average Pooling
    x = GlobalAveragePooling1D()(x)
    

    outputs = Dense(len(set(y_train)), activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logging.info("모델 학습 진행")
    history = model.fit(
        X_train_augmented, 
        y_train_augmented,
        epochs=15,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    
    logging.info("예측 시작")
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    
    logging.info("모델 평가 시작")
    accuracy = accuracy_score(y_test, y_pred_classes)
    report = classification_report(y_test, y_pred_classes)
    
    return model, accuracy, report, history
def create_synthetic_samples_for_lstm(X_train, y_train, target_class, num_samples=30):
   
    class_mask = y_train == target_class
    class_samples = X_train[class_mask]
    
    if len(class_samples) == 0:
        return np.array([]), np.array([])
    
    synthetic_samples = []
    synthetic_labels = []
    
    for _ in range(num_samples):
        base_sample = class_samples[np.random.choice(len(class_samples))]
        noise = np.random.normal(0, 0.02, base_sample.shape)
        synthetic_sample = base_sample + noise
        
        synthetic_samples.append(synthetic_sample)
        synthetic_labels.append(target_class)
    
    return np.array(synthetic_samples), np.array(synthetic_labels)

############### CNN 기반

def preprocess_data_CNN(X_train, X_test):

    selector = VarianceThreshold(threshold=0.01)
    X_train = selector.fit_transform(X_train)
    X_test = selector.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, X_test

def train_and_evaluate_model_CNN(X_train, X_test, y_train, y_test):
    logging.info("모델 초기화 및 학습 시작")
    
    from collections import Counter
    class_counts = Counter(y_train)
    print("Original class distribution:", dict(class_counts))
    
    min_samples = 30
    augmented_X = list(X_train)
    augmented_y = list(y_train)
    
    for class_label, count in class_counts.items():
        if count <= 2:
            samples_to_generate = min_samples - count
            synthetic_X, synthetic_y = create_synthetic_samples_for_lstm(
                X_train, y_train, 
                target_class=class_label, 
                num_samples=samples_to_generate
            )
            if len(synthetic_X) > 0:
                augmented_X.extend(synthetic_X)
                augmented_y.extend(synthetic_y)
    
    X_train_augmented = np.array(augmented_X)
    y_train_augmented = np.array(augmented_y)
    
    print("Augmented class distribution:", dict(Counter(y_train_augmented)))
    
    timesteps = X_train_augmented.shape[1]
    features = X_train_augmented.shape[2]
    
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(timesteps, features)),
        BatchNormalization(),
        Conv1D(filters=128, kernel_size=2, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Conv1D(filters=64, kernel_size=2, activation='relu'),
        BatchNormalization(),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(len(set(y_train)), activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logging.info("모델 학습 진행")
    history = model.fit(
        X_train_augmented,
        y_train_augmented,
        epochs=15,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    
    logging.info("예측 시작")
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    
    accuracy = accuracy_score(y_test, y_pred_classes)
    report = classification_report(y_test, y_pred_classes)
    
    return model, accuracy, report, history

def augment_minority_classes(X_train, y_train, min_samples=50):

    from collections import Counter
    import gc  
    class_counts = Counter(y_train)
    
    minority_classes = [cls for cls, count in class_counts.items() if count < min_samples]
    
    if minority_classes:
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        
        gc.collect()
        
        try:
            smote = SMOTE(random_state=42, 
                         k_neighbors=min(3, min(class_counts.values())-1),
                         sampling_strategy={cls: min(min_samples, max(class_counts.values()))
                                         for cls in minority_classes})
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_2d, y_train)
            
            X_train_balanced = X_train_balanced.reshape(-1, X_train.shape[1], X_train.shape[2])
            
            gc.collect()
            
            return X_train_balanced, y_train_balanced
            
        except MemoryError:
            print("메모리 부족으로 SMOTE를 적용할 수 없습니다. 원본 데이터를 반환합니다.")
            return X_train, y_train
    
    return X_train, y_train

def preprocess_data(X_train, X_test):

    selector = VarianceThreshold(threshold=0.01)
    X_train = selector.fit_transform(X_train)
    X_test = selector.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def preprocess_data_LSTM(X_train, X_test):

    selector = VarianceThreshold(threshold=0.01)
    X_train = selector.fit_transform(X_train)
    X_test = selector.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    seq_length = 3
    X_train_reshaped = []
    X_test_reshaped = []
    
    for i in range(0, X_train.shape[1] - seq_length + 1, seq_length):
        X_train_reshaped.append(X_train[:, i:i+seq_length])
    
    for i in range(0, X_test.shape[1] - seq_length + 1, seq_length):
        X_test_reshaped.append(X_test[:, i:i+seq_length])
    
    X_train_reshaped = np.stack(X_train_reshaped, axis=2)
    X_test_reshaped = np.stack(X_test_reshaped, axis=2)
    
    return X_train_reshaped, X_test_reshaped

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    logging.info("모델 초기화 및 학습 시작")
    

    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(len(set(y_train)), activation='softmax')
    ])
    

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    

    logging.info("모델 학습 진행")
    history = model.fit(
        X_train, 
        y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        verbose=1
    )
    

    logging.info("예측 시작")
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    

    logging.info("모델 평가 시작")
    accuracy = accuracy_score(y_test, y_pred_classes)
    report = classification_report(y_test, y_pred_classes)
    
    return model, accuracy, report, history

def train_and_evaluate_model_LSTM(X_train, X_test, y_train, y_test):
    logging.info("모델 초기화 및 학습 시작")
    
    # # 입력 형태 계산
    # timesteps = X_train.shape[1]  # 시퀀스 길이 (3)
    # features = X_train.shape[2]   # 특성 시퀀스 개수

    # #########SMOTE 적용시###############
    # X_train_augmented, y_train_augmented = augment_minority_classes(X_train, y_train, min_samples=50)
    # # 입력 형태 계산
    # timesteps = X_train_augmented.shape[1]
    # features = X_train_augmented.shape[2]
    
    # ##################################

    ######### 노이즈 기반 샘플 생성시###############


    from collections import Counter
    class_counts = Counter(y_train)
    print("Original class distribution:", dict(class_counts))
    

    min_samples = 30 
    augmented_X = list(X_train)
    augmented_y = list(y_train)
    
    for class_label, count in class_counts.items():
        if count <= 2:  

            samples_to_generate = min_samples - count

            synthetic_X, synthetic_y = create_synthetic_samples_for_lstm(
                X_train, y_train, 
                target_class=class_label, 
                num_samples=samples_to_generate
            )

            if len(synthetic_X) > 0:
                augmented_X.extend(synthetic_X)
                augmented_y.extend(synthetic_y)
    

    X_train_augmented = np.array(augmented_X)
    y_train_augmented = np.array(augmented_y)
    
    print("Augmented class distribution:", dict(Counter(y_train_augmented)))
    

    timesteps = X_train_augmented.shape[1]
    features = X_train_augmented.shape[2]
    #########################################################


    model = Sequential([
        LSTM(128, input_shape=(timesteps, features), return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(len(set(y_train)), activation='softmax')
    ])
    

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    

    logging.info("모델 학습 진행")
    history = model.fit(
        X_train, 
        y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    
        
    logging.info("예측 시작")
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    
    logging.info("모델 평가 시작")
    accuracy = accuracy_score(y_test, y_pred_classes)
    report = classification_report(y_test, y_pred_classes)
    
    return model, accuracy, report, history

    #네트워크 구조 개선:
    #첫 번째 레이어의 유닛 수를 128에서 256으로 증가
    #중간에 128 유닛을 가진 레이어 추가
    #과적합 방지를 위해 드롭아웃 레이어 강화 (0.3)
    #학습 파라미터 최적화:
    #에포크 수를 10에서 15로 증가
    #배치 사이즈를 128에서 64로 감소 (더 세밀한 학습)

