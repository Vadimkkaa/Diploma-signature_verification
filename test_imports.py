import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import OneClassSVM
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

# Простой вывод подтверждений
print("✅ OpenCV:", cv2.__version__)
print("✅ NumPy:", np.__version__)
print("✅ scikit-image: OK")
print("✅ scikit-learn: OK")
print("✅ Matplotlib:", matplotlib.__version__)
print("✅ Pandas:", pd.__version__)
print("✅ SQLite3: встроен и работает!")

# Мини-тест: создание таблицы SQLite
conn = sqlite3.connect("test_log.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS log (id INTEGER PRIMARY KEY, note TEXT)")
cursor.execute("INSERT INTO log (note) VALUES ('Test passed')")
conn.commit()
conn.close()

print("✅ SQLite: запись прошла успешно")
