"""
Гибридная модель: статистические фичи + RandomForest.

Логика работы:
1. Из окна [8 x 60] считаем агрегированные статистики (среднее, std, перцентили и т.д.)
2. RandomForest выдаёт вероятность читерства (0.0 – 1.0)

При первом запуске: модель обучается на синтетических данных (heuristic-based).
После того как соберёшь реальные CSV через /oasis datacollect — запусти retrain.py
и модель заменится на обученную по настоящим данным.
"""

import os
import pickle
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger("oasis-model")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/data/model.pkl"))
_model = None  # глобальный объект модели


# ── Извлечение фич из окна [8 x 60] ───────────────────────────────────────

FEATURE_NAMES = [
    "delta_yaw", "delta_pitch",
    "accel_yaw", "accel_pitch",
    "jerk_yaw",  "jerk_pitch",
    "gcd_error_yaw", "gcd_error_pitch",
]

def extract_features(matrix: np.ndarray) -> np.ndarray:
    """
    matrix: float32 [8, 60]
    Возвращает вектор признаков для sklearn-модели.
    """
    feats = []
    for i in range(8):
        row = matrix[i]
        abs_row = np.abs(row)
        feats += [
            float(np.mean(abs_row)),
            float(np.std(abs_row)),
            float(np.percentile(abs_row, 25)),
            float(np.percentile(abs_row, 75)),
            float(np.percentile(abs_row, 95)),
            float(np.max(abs_row)),
        ]

    # Специальные фичи для GCD error (индексы 6 и 7)
    gcd_yaw   = matrix[6]
    gcd_pitch = matrix[7]

    # Низкий GCD error = подозрительно равномерные движения (aim assist)
    feats.append(float(np.mean(gcd_yaw < 0.05)))    # доля "идеальных" тиков по yaw
    feats.append(float(np.mean(gcd_pitch < 0.05)))  # то же по pitch

    # Аномально равномерное ускорение (bot-like)
    accel_yaw   = np.abs(matrix[2])
    accel_pitch = np.abs(matrix[3])
    feats.append(float(np.std(accel_yaw)))
    feats.append(float(np.std(accel_pitch)))

    # Соотношение jerk к accel (у читов почти постоянное)
    jerk_yaw = np.abs(matrix[4]) + 1e-7
    jerk_p   = np.abs(matrix[5]) + 1e-7
    acc_y    = np.abs(matrix[2]) + 1e-7
    acc_p    = np.abs(matrix[3]) + 1e-7
    feats.append(float(np.mean(jerk_yaw / acc_y)))
    feats.append(float(np.mean(jerk_p   / acc_p)))

    return np.array(feats, dtype=np.float32)


# ── Синтетическая генерация обучающих данных ──────────────────────────────

def _generate_synthetic_data(n_samples: int = 5000):
    """
    Генерирует синтетические данные для начального обучения.

    Легитимный игрок:
    - delta_yaw/pitch: случайное, std высокое
    - gcd_error: близок к равномерному 0.3–0.8
    - accel: хаотичный

    Читер (aim assist / killaura):
    - delta_yaw: равномерные маленькие движения с низким std
    - gcd_error_yaw: очень низкий (< 0.05) — движения кратны одному шагу
    - accel: очень маленький и стабильный
    """
    rng = np.random.default_rng(42)
    X_list, y_list = [], []

    half = n_samples // 2

    # Легитимные игроки
    for _ in range(half):
        m = np.zeros((8, 60), dtype=np.float32)
        m[0] = rng.normal(0, rng.uniform(1.0, 8.0), 60)       # delta_yaw
        m[1] = rng.normal(0, rng.uniform(0.5, 4.0), 60)       # delta_pitch
        m[2] = rng.normal(0, rng.uniform(0.5, 3.0), 60)       # accel_yaw
        m[3] = rng.normal(0, rng.uniform(0.3, 2.0), 60)       # accel_pitch
        m[4] = rng.normal(0, rng.uniform(0.3, 2.0), 60)       # jerk_yaw
        m[5] = rng.normal(0, rng.uniform(0.2, 1.5), 60)       # jerk_pitch
        m[6] = rng.uniform(0.1, 0.9, 60)                       # gcd_error_yaw (разброс)
        m[7] = rng.uniform(0.1, 0.9, 60)                       # gcd_error_pitch
        X_list.append(extract_features(m))
        y_list.append(0)

    # Читеры (aim-assist / smooth rotation hack)
    for _ in range(half):
        cheat_type = rng.integers(0, 3)
        m = np.zeros((8, 60), dtype=np.float32)

        if cheat_type == 0:
            # Aim assist — равномерные микро-движения, почти нулевой GCD error
            base = rng.uniform(0.1, 0.5)
            m[0] = rng.normal(base, 0.01, 60)
            m[1] = rng.normal(base * 0.5, 0.005, 60)
            m[2] = rng.normal(0, 0.005, 60)
            m[3] = rng.normal(0, 0.003, 60)
            m[4] = rng.normal(0, 0.002, 60)
            m[5] = rng.normal(0, 0.002, 60)
            m[6] = rng.uniform(0.0, 0.03, 60)   # почти нулевой GCD error
            m[7] = rng.uniform(0.0, 0.03, 60)

        elif cheat_type == 1:
            # KillAura — резкие snap-повороты с паузами
            m[0] = rng.choice([0.0, rng.uniform(5, 15)], 60, p=[0.7, 0.3])
            m[1] = rng.choice([0.0, rng.uniform(2, 8)],  60, p=[0.7, 0.3])
            m[2] = np.diff(np.concatenate([[0], m[0]]))
            m[3] = np.diff(np.concatenate([[0], m[1]]))
            m[4] = np.diff(np.concatenate([[0], m[2]]))
            m[5] = np.diff(np.concatenate([[0], m[3]]))
            m[6] = rng.uniform(0.0, 0.02, 60)
            m[7] = rng.uniform(0.0, 0.02, 60)

        else:
            # Smooth aimbot — очень плавный трекинг цели
            t = np.linspace(0, 2 * np.pi, 60)
            m[0] = (np.sin(t) * rng.uniform(0.5, 2.0)).astype(np.float32)
            m[1] = (np.cos(t) * rng.uniform(0.3, 1.0)).astype(np.float32)
            m[2] = np.gradient(m[0]).astype(np.float32)
            m[3] = np.gradient(m[1]).astype(np.float32)
            m[4] = np.gradient(m[2]).astype(np.float32)
            m[5] = np.gradient(m[3]).astype(np.float32)
            m[6] = rng.uniform(0.0, 0.04, 60)
            m[7] = rng.uniform(0.0, 0.04, 60)

        X_list.append(extract_features(m))
        y_list.append(1)

    X = np.stack(X_list)
    y = np.array(y_list)

    # Перемешать
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ── Обучение / загрузка модели ─────────────────────────────────────────────

def _train_and_save():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    logger.info("Training model on synthetic data...")
    X, y = _generate_synthetic_data(n_samples=8000)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])

    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    logger.info(f"Cross-val ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model saved to {MODEL_PATH}")
    return model


def load_model():
    global _model
    if MODEL_PATH.exists():
        logger.info(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    else:
        logger.info("No saved model found, training from scratch...")
        _model = _train_and_save()


def predict(matrix: np.ndarray) -> float:
    """
    matrix: float32 [8, 60]
    Возвращает confidence в диапазоне 0.0 – 1.0
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    feats = extract_features(matrix).reshape(1, -1)
    prob = _model.predict_proba(feats)[0][1]  # вероятность класса 1 (читер)
    return float(np.clip(prob, 0.0, 1.0))
