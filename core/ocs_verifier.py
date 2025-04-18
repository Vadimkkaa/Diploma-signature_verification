import os
import joblib  # для сохранения и загрузки модели
from sklearn.svm import OneClassSVM
import numpy as np
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class OCSVMVerifier:
    """
    Класс для обучения и верификации с помощью One-Class SVM.
    """

    def __init__(self, model_dir="storage/models"):
        """
        Инициализация: создаём или указываем папку для хранения моделей.
        """
        self.model = None
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def fit(self, features_list):
        """
        Обучает модель на списке векторов признаков (только подлинные подписи).
        """
        self.model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        self.model.fit(features_list)

    def predict(self, feature):
        """
        Проверяет один вектор признаков.
        Возвращает: 1 (своя подпись) или -1 (подделка)
        """
        if self.model is None:
            raise ValueError("Модель не обучена или не загружена.")

        return self.model.predict([feature])[0]  # возвращает 1 или -1

    def save_model(self, user_id):
        """
        Сохраняет обученную модель в файл (по ID пользователя).
        """
        if self.model is None:
            raise ValueError("Нет модели для сохранения.")

        path = os.path.join(BASE_DIR, "storage", "models", f"user_{user_id}_model.pkl")
        joblib.dump(self.model, path)

    def load_model(self, user_id):
        """
        Загружает модель из файла по ID пользователя.
        """
        path = os.path.join(self.model_dir, f"user_{user_id}_model.pkl")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Модель пользователя user_{user_id} не найдена.")

        self.model = joblib.load(path)

    def evaluate(self, X_test, y_true):
        """
        Оценка точности на тестовой выборке.
        X_test — список векторов признаков
        y_true — список меток: 1 (своя), -1 (подделка)
        Возвращает точность, precision, recall, F1
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred, pos_label=1)

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
