import os
import sys
import time

# Добавляем текущую директорию в путь, чтобы импорты работали корректно
sys.path.append(os.getcwd())

from src.core.classifier import GuardianClassifier

def print_alert(result):
    print("\n" + "!" * 50)
    print("!!! ОБНАРУЖЕНА УГРОЗА МОШЕННИЧЕСТВА !!!")
    print(f"Текст: {result['text']}")
    print(f"Причина: {result['reason']}")
    if result['triggers']:
        print(f"Сработавшие триггеры: {', '.join(result['triggers'])}")
    print(f"ML Оценка уверенности: {result['ml_score']:.2%}")
    print("!" * 50 + "\n")
    print(">> ОТПРАВКА УВЕДОМЛЕНИЯ РОДСТВЕННИКУ...")
    time.sleep(1)
    print(">> УВЕДОМЛЕНИЕ ОТПРАВЛЕНО!")

def main():
    print("--- GuardianAI: Цифровой телохранитель (Прототип) ---")
    classifier = GuardianClassifier()
    
    dataset_path = os.path.join(os.getcwd(), 'dataset')
    print(f"Обучение модели на данных из {dataset_path}...")
    classifier.train(dataset_path)
    print("Система готова к работе!\n")
    
    print("Введите текст сообщения (или 'exit' для выхода):")
    print("Пример: 'Мам, срочно нужны деньги' или 'Привет, как дела?'")
    
    while True:
        try:
            user_input = input("\n[Входящее сообщение] > ")
            if user_input.lower() in ['exit', 'quit', 'выход']:
                break
            
            if not user_input.strip():
                continue
                
            result = classifier.predict(user_input)
            
            if result['is_scam']:
                print_alert(result)
            else:
                print(f"[OK] Сообщение безопасно (ML Score: {result['ml_score']:.2%})")
                
        except KeyboardInterrupt:
            break
            
    print("\nЗавершение работы GuardianAI.")

if __name__ == "__main__":
    main()
