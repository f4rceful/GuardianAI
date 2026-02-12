import sys
import os
import time
import numpy as np

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.classifier import GuardianClassifier

def benchmark():
    print("=== Бенчмарк Модели ===")
    
    # Инициализация классификатора (должен авто-определить ONNX)
    clf = GuardianClassifier()
    if not clf.use_onnx:
        print("[ВНИМАНИЕ] ONNX не обнаружен! Бенчмарк будет запущен только на PyTorch.")
    
    # Принудительная загрузка
    print("Инициализация движка...")
    clf._init_bert()
    mode = "ONNX" if clf.use_onnx else "PyTorch"
    print(f"Активный режим: {mode}")

    # Подготовка данных
    texts = ["Тестовое сообщение для проверки скорости."] * 100
    print(f"Запуск инференса на {len(texts)} примерах...")
    
    start_time = time.time()
    # Прямой вызов эмбеддингов для замера чистой скорости модели
    embeddings = clf._get_bert_embeddings(texts)
    end_time = time.time()
    
    duration = end_time - start_time
    avg_time = (duration / len(texts)) * 1000 # мс
    
    print(f"\nОбщее время: {duration:.4f}s")
    print(f"Среднее время на пример: {avg_time:.2f}ms")
    
    if avg_time < 50:
         print(f"🚀 Производительность ОТЛИЧНАЯ (<50ms). {mode} работает хорошо.")
    elif avg_time < 150:
         print(f"✅ Производительность ХОРОШАЯ (<150ms).")
    else:
         print(f"⚠️ Производительность НИЗКАЯ (>150ms). Требуется оптимизация.")

if __name__ == "__main__":
    benchmark()
