import sys
import os

# Добавляем корневую директорию в путь импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.classifier import GuardianClassifier
from src.utils.text_processing import normalize_homoglyphs

def test_homoglyphs():
    print("=== Тестирование защиты от гомоглифов ===")
    
    # 1. Проверка функции нормализации
    input_text = "Mам, скинь дeнeг" # M (lat), e (lat)
    expected = "Мам, скинь денег" # Кириллица
    normalized = normalize_homoglyphs(input_text)
    
    print(f"Input:    '{input_text}'")
    print(f"Result:   '{normalized}'")
    print(f"Expected: '{expected}'")
    
    if normalized == expected:
        print("[OK] Нормализация работает корректно.")
    else:
        print("[FAIL] Ошибка нормализации!")
        diff = []
        for i, (c1, c2) in enumerate(zip(normalized, expected)):
            if c1 != c2:
                diff.append(f"Pos {i}: '{c1}' (code {ord(c1)}) != '{c2}' (code {ord(c2)})")
        print("\n".join(diff))

    # 2. Проверка Классификатора
    print("\n--- Проверка классификации ---")
    try:
        clf = GuardianClassifier()
        if not clf.is_trained:
            print("[SKIP] Модель не обучена. Пропуск теста классификации.")
            return

        # Тест: Фраза, которая должна быть SCAM, но написана латиницей
        scam_phrase_lat = "Bаш aккaунт зaблoкиpoвaн." # B, a, o, p - latin
        scam_phrase_cyr = "Ваш аккаунт заблокирован."
        
        res_lat = clf.predict(scam_phrase_lat)
        res_cyr = clf.predict(scam_phrase_cyr)
        
        print(f"Latin phrase score: {res_lat['ml_score']:.4f} (Is Scam: {res_lat['is_scam']})")
        print(f"Cyril phrase score: {res_cyr['ml_score']:.4f} (Is Scam: {res_cyr['is_scam']})")
        
        #Scores should be very close
        if abs(res_lat['ml_score'] - res_cyr['ml_score']) < 0.05:
             print("[OK] Гомоглифы не влияют на ML скор.")
        else:
             print("[WARNING] Слишком большая разница в скорах! Нормализация может не применяться перед ML.")

        if res_lat['is_scam']:
            print("[OK] Атака гомоглифами успешно отбита (Detected as Scam).")
        else:
            print("[FAIL] Атака гомоглифами прошла (Detected as Safe).")

    except Exception as e:
        print(f"Ошибка при инициализации классификатора: {e}")

if __name__ == "__main__":
    test_homoglyphs()
