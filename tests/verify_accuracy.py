from src.core.classifier import GuardianClassifier
import os

def check_examples():
    clf = GuardianClassifier()
    if not clf.is_trained:
        print("Модель не обучена!")
        return

    file_path = "exampels.txt"
    if not os.path.exists(file_path):
        print("Файл exampels.txt не найден!")
        return
        
    total = 0
    correct = 0
    errors = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"{'ОЖИДАЛОСЬ':<10} | {'ПОЛУЧЕНО':<10} | {'ТЕКСТ'}")
        print("-" * 80)
        
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Парсинг строки "Type: "Text""
            parts = line.split(':', 1)
            if len(parts) < 2: continue
            
            expected_type = parts[0].strip().upper() # SCAM или SAFE
            text = parts[1].strip().strip('"')
            
            # Предсказание
            result = clf.predict(text)
            # predict возвращает СЛОВАРЬ (DICT), а не кортеж
            is_scam = result.get('is_scam', False)
            predicted_type = "SCAM" if is_scam else "SAFE"
            
            total += 1
            if predicted_type == expected_type:
                correct += 1
            else:
                reason = result.get('reason', 'Неизвестно')
                errors.append(f"Ожидалось: {expected_type}, Получено: {predicted_type} | Причина: {reason} | Текст: {text}")

    print("-" * 80)
    print(f"Всего: {total}")
    print(f"Правильно: {correct}")
    if total > 0:
        print(f"Точность (Accuracy): {correct/total*100:.2f}%")
    
    if errors:
        print("\nОШИБКИ (Не прошли проверку):")
        for err in errors:
            print(err)
    else:
        print("\nВсе примеры прошли проверку успешно! 🎉")

if __name__ == "__main__":
    check_examples()
