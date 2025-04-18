import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from core.signature_normalizer import SignatureNormalizer
from core.signature_features import SignatureFeaturesExtractor
from core.ocs_verifier import OCSVMVerifier
from utils.results_logger import ResultsLogger

from core.image_preprocessor import ImagePreprocessor

# Автоматически определим абсолютный путь к корню проекта
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# Формируем абсолютные пути
originals_path = os.path.join(BASE_DIR, "storage", "data", "CEDAR", "originals")
forgeries_path = os.path.join(BASE_DIR, "storage", "data", "CEDAR", "forgeries")
models_path = os.path.join(BASE_DIR, "storage", "models")
logs_path = os.path.join(BASE_DIR, "storage", "logs")

# Настройки
user_id = "1"
# base_path = "storage/data/CEDAR"
# originals_path = os.path.join(base_path, "originals")
# forgeries_path = os.path.join(base_path, "forgeries")

# Разбиваем оригинальные подписи: 18 для обучения, 6 для теста
original_files = sorted([
    f for f in os.listdir(originals_path)
    if f.startswith("original_1_")
])

forgery_files = sorted([
    f for f in os.listdir(forgeries_path)
    if f.startswith("forgeries_1_")
])


train_files = original_files[:18]
test_originals = original_files[18:]  # 6 штук
test_forgeries = forgery_files  # все 24

# Обучение модели
features_train = []

for fname in train_files:
    img = cv2.imread(os.path.join(originals_path, fname), 0)
    norm = SignatureNormalizer(img).normalize()

    # pre = ImagePreprocessor(norm)
    # pre.apply_threshold()
    # pre.remove_noise()
    # processed_img = pre.get_result()

    feat = SignatureFeaturesExtractor(norm).extract_all()
    features_train.append(feat)

verifier = OCSVMVerifier()
verifier.fit(features_train)
verifier.save_model(user_id)

# Проверка модели
X_test = []
y_true = []

# Проверка на оставшихся оригиналах (должны быть “свои” → 1)
for fname in test_originals:
    img = cv2.imread(os.path.join(originals_path, fname), 0)
    norm = SignatureNormalizer(img).normalize()
    feat = SignatureFeaturesExtractor(norm).extract_all()
    X_test.append(feat)
    y_true.append(1)

# Проверка на подделках (должны быть “чужие” → -1)
for fname in test_forgeries:
    img = cv2.imread(os.path.join(forgeries_path, fname), 0)
    norm = SignatureNormalizer(img).normalize()
    feat = SignatureFeaturesExtractor(norm).extract_all()
    X_test.append(feat)
    y_true.append(-1)

# Предсказания
y_pred = verifier.model.predict(X_test)

# Подсчёт метрик
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, pos_label=1)
rec = recall_score(y_true, y_pred, pos_label=1)
f1 = f1_score(y_true, y_pred, pos_label=1)

# Вывод
print("\n📊 Результаты тестирования модели OC-SVM для",user_id,"человека")
print(f"✅ Accuracy:  {acc:.4f}")
print(f"✅ Precision: {prec:.4f}")
print(f"✅ Recall:    {rec:.4f}")
print(f"✅ F1-score:  {f1:.4f}")

# Дополнительно — вывод TP/FP/FN/TN
tp = sum((np.array(y_pred) == 1) & (np.array(y_true) == 1))
fp = sum((np.array(y_pred) == 1) & (np.array(y_true) == -1))
fn = sum((np.array(y_pred) == -1) & (np.array(y_true) == 1))
tn = sum((np.array(y_pred) == -1) & (np.array(y_true) == -1))

print(f"\n📌 TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")

# 📦 Логирование результатов в базу данных
logger = ResultsLogger()

metrics = {
    'accuracy': acc,
    'precision': prec,
    'recall': rec,
    'f1_score': f1
}

conf_matrix = {
    'tp': tp,
    'fp': fp,
    'fn': fn,
    'tn': tn
}

logger.log(user_id, metrics, conf_matrix)
print("📄 Результаты сохранены в базу данных.")

