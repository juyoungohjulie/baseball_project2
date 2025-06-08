import pandas as pd

def feature_engineering(X):
    # 예시: 투구 속도와 회전수의 상호작용 특성 추가
    if 'velocity' in X.columns and 'spin_rate' in X.columns:
        X['velocity_spin_interaction'] = X['velocity'] * X['spin_rate']

    # 예시: 투구 위치에 따른 특성 추가
    if 'px' in X.columns and 'pz' in X.columns:
        X['pitch_location'] = (X['px']**2 + X['pz']**2)**0.5

    return X