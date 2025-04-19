import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from core.signature_normalizer import SignatureNormalizer
from core.signature_features import SignatureFeaturesExtractor
from core.ocs_verifier import OCSVMVerifier
from utils.results_logger import ResultsLogger

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
originals_path = os.path.join(BASE_DIR, "storage", "data", "CEDAR", "originals")
forgeries_path = os.path.join(BASE_DIR, "storage", "data", "CEDAR", "forgeries")
models_path = os.path.join(BASE_DIR, "storage", "models")

logger = ResultsLogger()

def train_and_log_model(user_id: int):
    original_files = sorted([
        f for f in os.listdir(originals_path)
        if f.startswith(f"original_{user_id}_")
    ])
    forgery_files = sorted([
        f for f in os.listdir(forgeries_path)
        if f.startswith(f"forgeries_{user_id}_")
    ])

    train_files = original_files[:18]
    test_originals = original_files[18:]
    test_forgeries = forgery_files

    # Обучение модели
    features_train = []
    for fname in train_files:
        img = cv2.imread(os.path.join(originals_path, fname), 0)
        norm = SignatureNormalizer(img, target_size=(300, 150)).normalize()
        feat = SignatureFeaturesExtractor(norm).extract_all()
        features_train.append(feat)

    verifier = OCSVMVerifier()
    verifier.fit(features_train)
    verifier.save_model(user_id)

    # Тестирование
    X_test = []
    y_true = []

    for fname in test_originals:
        img = cv2.imread(os.path.join(originals_path, fname), 0)
        norm = SignatureNormalizer(img, target_size=(300, 150)).normalize()
        feat = SignatureFeaturesExtractor(norm).extract_all()
        X_test.append(feat)
        y_true.append(1)

    for fname in test_forgeries:
        img = cv2.imread(os.path.join(forgeries_path, fname), 0)
        norm = SignatureNormalizer(img, target_size=(300, 150)).normalize()
        feat = SignatureFeaturesExtractor(norm).extract_all()
        X_test.append(feat)
        y_true.append(-1)

    y_pred = [verifier.predict(x) for x in X_test]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == -1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == -1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == -1 and yp == -1) for yt, yp in zip(y_true, y_pred))

    print(f"\n📊 Результаты тестирования модели OC-SVM для пользователя {user_id}")
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ Recall:    {rec:.4f}")
    print(f"✅ F1-score:  {f1:.4f}")
    print(f"📌 TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")

    # Логирование
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }

    confusion = {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

    logger.log(user_id, metrics, confusion)
    print("📄 Результаты сохранены в базу данных.")


# 🔁 Запуск обучения для всех 20 пользователей
for user_id in range(1, 21):
    train_and_log_model(user_id)
