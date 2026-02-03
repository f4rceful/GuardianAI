import sys
import os
import time

# Add src to path
sys.path.append(os.getcwd())

from src.core.classifier import GuardianClassifier

def test_model_baseline():
    print("=== ЗАПУСК БАЗОВОГО ТЕСТИРОВАНИЯ МОДЕЛИ ===")
    
    # 1. Загрузка модели
    try:
        clf = GuardianClassifier()
        if not clf.is_trained:
            print("[ERROR] ОШИБКА: Модель не обучена! Сначала запустите train.py")
            return
        print(f"[OK] Модель успешно загружена. Порог: {clf.threshold}")
    except Exception as e:
        print(f"[ERROR] Критическая ошибка при загрузке: {e}")
        return

    # 2. Набор тестовых сценариев (Сложные случаи)
    test_cases = [
        # Группа 1: Очевидный скам (должен ловиться легко)
        ("Мам, скинь 5000 на карту срочно, потом объясню", True, "Classic Scam"),
        ("Ваша карта заблокирована. Позвоните в службу безопасности: +79001234567", True, "Bank Security fake"),
        
        # Группа 2: Легитимные сообщения (должны быть Safe)
        ("Купил хлеб и молоко, буду через 15 минут", False, "Normal chat"),
        ("Ваш код подтверждения для входа в Госуслуги: 8892. Никому не сообщайте.", False, "System Notification"),
        
        # Группа 3: Сложные случаи (Попытка обхода)
        ("Baшa кapтa зaблoкиpoвaнa (латиница вместо кириллицы)", True, "Homoglyphs"),
        ("Привет, это служба бeзoпacнocти, нам нужно сверить данные", True, "Social Engineering"),
        
        # Группа 4: Эмоциональное давление (То, что мы хотим улучшить)
        ("СРОЧНО!!! ПОМОГИ!!! Я В БЕДЕ!!!", True, "Panic/Urgency"),
        ("Поздравляем! Вы выиграли автомобиль! Заберите приз по ссылке", True, "Joy/Lottery")
    ]

    print(f"\nЗапуск проверки на {len(test_cases)} примерах...\n")
    
    correct = 0
    start_time = time.time()
    
    for text, expected, label in test_cases:
        print(f"[TEST] Тест: '{text}' ({label})")
        
        result = clf.predict(text)
        is_scam = result['is_scam']
        score = result['ml_score']
        triggers = result['triggers']
        reason = result['reason']
        
        status = "[PASS]" if is_scam == expected else "[FAIL]"
        
        if is_scam == expected: # Changed from `if status == "✅":`
            correct += 1
        
        print(f"   Результат: {status} | Вердикт: {'SCAN' if is_scam else 'SAFE'} (Ожидалось: {'SCAM' if expected else 'SAFE'})")
        print(f"   ML Score: {score:.4f} | Причина: {reason}")
        if triggers:
            print(f"   Триггеры: {triggers}")
        print("-" * 50)
        
    duration = time.time() - start_time
    accuracy = (correct / len(test_cases)) * 100
    
    print(f"\n=== ИТОГИ ===")
    print(f"Всего тестов: {len(test_cases)}")
    print(f"Пройдено успешно: {correct}")
    print(f"Точность: {accuracy:.1f}%")
    print(f"Время выполнения: {duration:.4f} сек")
    
    if accuracy < 100:
        print("\n[WARN] ВЫВОД: Модель работает, но есть ошибки. Улучшение Sentiment Analysis и Natasha поможет исправить промахи в Группе 3 и 4.")
    else:
        print("\n[SUCCESS] ВЫВОД: Текущая модель справляется отлично! Но улучшения сделают её еще надежнее.")

if __name__ == "__main__":
    test_model_baseline()
