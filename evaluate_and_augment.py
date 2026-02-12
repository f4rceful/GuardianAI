from src.core.classifier import GuardianClassifier
import os

def process_examples():
    input_file = "exampels.txt"
    scam_file = os.path.join("dataset", "scam_samples.txt")
    safe_file = os.path.join("dataset", "safe_samples.txt")
    
    clf = GuardianClassifier()
    # Принудительная загрузка существующей модели
    if not clf.is_trained:
        print("Модель не обучена или не найдена!")
        return

    new_scams = []
    new_safes = []
    
    print(f"Чтение {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            label_str, text = line.split(":", 1)
            text = text.strip().strip('"')
            
            if label_str.lower() == "scam":
                new_scams.append(text)
                expected = True
            elif label_str.lower() == "safe":
                new_safes.append(text)
                expected = False
            else:
                continue
                
            # Тестирование текущей модели
            result = clf.predict(text)
            is_scam = result['is_scam']
            
            status = "✅" if is_scam == expected else "❌"
            if status == "❌":
                print(f"{status} Ошибка: {text[:60]}... (Ожидалось: {expected}, Получено: {is_scam}, Score: {result['ml_score']:.2f})")

    print(f"\n--- Итоги аугментации ---")
    print(f"Новых примеров Scam: {len(new_scams)}")
    print(f"Новых примеров Safe: {len(new_safes)}")
    
    # Добавление в датасет
    with open(scam_file, "a", encoding="utf-8") as f:
        for s in new_scams:
            f.write(f"\n{s}")
            
    with open(safe_file, "a", encoding="utf-8") as f:
        for s in new_safes:
            f.write(f"\n{s}")
            
    print("Датасет успешно обновлен.")

if __name__ == "__main__":
    process_examples()
