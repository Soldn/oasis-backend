#!/usr/bin/env python3
"""
retrain.py — дообучение модели на реальных CSV-данных с сервера.

Как использовать:
1. Собери данные через плагин:
   /oasis datacollect start legit <игрок>    — для честных игроков
   /oasis datacollect start cheat <игрок>    — для читеров
2. Скопируй CSV из plugins/oasisAC/datacollect/ в папку /data/datacollect/
3. Запусти: docker exec oasis-backend python retrain.py
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Добавляем app в path
sys.path.insert(0, "/app")
from app.model import extract_features, MODEL_PATH, _generate_synthetic_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("retrain")

DATACOLLECT_DIR = Path(os.getenv("DATACOLLECT_DIR", "/data/datacollect"))
FEATURE_COLS = 8 * 60  # 480 колонок фич


def load_csv_data():
    """Загружает все CSV из папки datacollect."""
    csv_files = list(DATACOLLECT_DIR.glob("*.csv"))
    if not csv_files:
        return None, None

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            logger.info(f"  Loaded {f.name}: {len(df)} rows, label={df['label'].value_counts().to_dict()}")
        except Exception as e:
            logger.warning(f"  Skipping {f.name}: {e}")

    if not dfs:
        return None, None

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows: {len(combined)} | cheats: {(combined['label']==1).sum()} | legit: {(combined['label']==0).sum()}")

    # Парсим матрицы [8, 60] из 480 колонок
    feat_cols = [c for c in combined.columns if c != "label"][:FEATURE_COLS]
    raw = combined[feat_cols].values.astype(np.float32)

    X_list = []
    for row in raw:
        matrix = row.reshape(8, 60)
        X_list.append(extract_features(matrix))

    X = np.stack(X_list)
    y = combined["label"].values.astype(int)
    return X, y


def main():
    logger.info("=== OasisAC Model Retraining ===")

    # Попытка загрузить реальные данные
    X_real, y_real = None, None
    if DATACOLLECT_DIR.exists():
        logger.info(f"Looking for CSV files in {DATACOLLECT_DIR}...")
        X_real, y_real = load_csv_data()

    if X_real is not None:
        n_real = len(y_real)
        logger.info(f"Real data: {n_real} samples")

        # Дополняем синтетикой если реальных данных мало
        if n_real < 500:
            synth_n = max(1000, 2000 - n_real)
            logger.info(f"Too few real samples, adding {synth_n} synthetic...")
            X_syn, y_syn = _generate_synthetic_data(synth_n)
            X = np.concatenate([X_real, X_syn])
            y = np.concatenate([y_real, y_syn])
        else:
            X, y = X_real, y_real
    else:
        logger.info("No real data found, training on synthetic data only.")
        X, y = _generate_synthetic_data(8000)

    # Разбивка на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(y_train)} | Test: {len(y_test)}")

    # Обучение
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )),
    ])

    logger.info("Training...")
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    logger.info(f"Cross-val ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    model.fit(X_train, y_train)

    # Оценка на тесте
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    logger.info(f"Test ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["legit", "cheat"]))

    # Сохранение
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {MODEL_PATH}")
    logger.info("Restart the server to apply: docker restart oasis-backend")


if __name__ == "__main__":
    main()
