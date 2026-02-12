from src.core.classifier import GuardianClassifier

def stress_test():
    print("Загрузка модели для стресс-теста...")
    clf = GuardianClassifier()
    clf.load_model()
    
    # [Text, Expected Is_Scam, Description]
    stress_cases = [
        # 1. Легитимные, но "страшные" сообщения (Проверка на False Positive)
        ("Сбербанк: Ваша заявка на ипотеку одобрена. Менеджер свяжется с вами.", 0, "Легитимная информация банка"),
        ("Внимание! Завтра плановое отключение горячей воды.", 0, "ЖКХ Уведомление"),
        ("Ваш пароль был успешно изменен. Если это были вы - игнорируйте.", 0, "Уведомление безопасности"),
        ("Верни долг 500 рублей, иначе я обижусь.", 0, "Долг/Агрессивный друг"),
        
        # 2. Хитрые скамы без ссылок (Социальная инженерия) (Проверка на False Negative)
        ("Привет! Проголосуй за меня, пожалуйста. Я участвую в конкурсе.", 1, "Фишинг с голосованием"),
        ("Слушай, мне тут код пришел случайно, скажи цифры? Я перепутал номер.", 1, "Кража кода из SMS"),
        ("Здравствуйте. Мы нашли ошибку в начислении пенсии. Нужен номер вашего СНИЛС.", 1, "Кража данных (Без ссылок)"),
        ("Мам, телефон сломался. Пишу с чужого. Переведи 3000 на карту 2202...", 1, "Схема 'Сломанный телефон'"),
        
        # 3. Транслит и обфускация
        ("Privet, mama. Srochno nuzhni dengi. Skin na kartu.", 1, "Скам на транслите"),
        ("Vash a k k a u n t budet udalen.", 1, "Обфускация пробелами"),
        
        # 4. Контекстные ловушки
        ("Да, я выиграл миллион! Ха-ха-ха, шутка.", 0, "Сарказм/Шутка"), 
        ("Смотри, какую машину я купил!", 0, "Легитимный контекст фото")
    ]
    
    print(f"\n--- СТРЕСС-ТЕСТ ({len(stress_cases)} кейсов) ---\n")
    
    passed = 0
    for text, expected, desc in stress_cases:
        result = clf.predict(text)
        pred = 1 if result['is_scam'] else 0
        score = result['ml_score']
        
        status = "[PASS]" if pred == expected else "[FAIL]"
        label = "SCAM" if pred == 1 else "Safe"
        expect_label = "SCAM" if expected == 1 else "Safe"
        
        print(f"{status} [{desc}]")
        print(f"   Text: '{text[:50]}...'")
        print(f"   Verdict: {label} ({score:.4f}) | Expected: {expect_label}")
        print("-" * 30)
        
        if pred == expected: passed += 1
        
    print(f"\nИтог: {passed}/{len(stress_cases)} пройдены.")

if __name__ == "__main__":
    stress_test()
