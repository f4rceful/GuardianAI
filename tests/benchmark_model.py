import sys
import os
import time
import numpy as np

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.classifier import GuardianClassifier

def benchmark():
    print("=== Model Benchmark ===")
    
    # Init classifier (should auto-detect ONNX)
    clf = GuardianClassifier()
    if not clf.use_onnx:
        print("[WARNING] ONNX not detected! Benchmark will run on PyTorch only.")
    
    # Force loading
    print("Initializing engine...")
    clf._init_bert()
    mode = "ONNX" if clf.use_onnx else "PyTorch"
    print(f"Active Mode: {mode}")

    # Prepare data
    texts = ["Тестовое сообщение для проверки скорости."] * 100
    print(f"Running inference on {len(texts)} samples...")
    
    start_time = time.time()
    # Direct embedding call to measure purely model speed
    embeddings = clf._get_bert_embeddings(texts)
    end_time = time.time()
    
    duration = end_time - start_time
    avg_time = (duration / len(texts)) * 1000 # ms
    
    print(f"\nTotal Time: {duration:.4f}s")
    print(f"Avg Time per sample: {avg_time:.2f}ms")
    
    if avg_time < 50:
         print(f"🚀 Performance is GREAT (<50ms). {mode} is working well.")
    elif avg_time < 150:
         print(f"✅ Performance is GOOD (<150ms).")
    else:
         print(f"⚠️ Performance is SLOW (>150ms). Optimization might be needed.")

if __name__ == "__main__":
    benchmark()
