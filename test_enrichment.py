from src.core.classifier import GuardianClassifier

def test_enrichment():
    print("Загрузка модели...")
    clf = GuardianClassifier()
    clf.load_model()
    
    test_cases = [
        # --- ENRICHMENT TARGETS (Previously Unknown or Weak) ---
        ("Поздравляем! Вы выиграли автомобиль Toyota Camry. Оплатите пошлину.", 1),
        ("Ура! Вы выиграли машину! Mercedes-Benz C-Class ваш.", 1),
        ("Инвестиции в Газпром! Доходность от 50% в месяц.", 1),
        ("Арбитраж криптовалют. Связки с доходностью 3-5% за круг.", 1),
        ("Требуется менеджер Wildberries. Удаленная работа, 2-3 часа.", 1),
        ("Заработок на лайках и отзывах. До 5000 рублей в день.", 1),
        ("Привет, любимая. Я отправил тебе подарок из Сирии, но он застрял.", 1),
        ("Я американский генерал, хочу переехать в Россию.", 1),
        
        # --- REGRESSION TESTS (Should remain correct) ---
        ("Boxberry: Заказ 755175 прибыл в пункт выдачи.", 0),
        ("Встреча подтверждена: завтра в 11:00.", 0),
        ("Мам, скинь 5000 на карту.", 1),
        ("Привет! Это твои фото?", 1), 
    ]
    
    print("\n--- ЗАПУСК ТЕСТОВ ОБОГАЩЕНИЯ ---\n")
    passed = 0
    for text, expected in test_cases:
        result = clf.predict(text)
        pred = 1 if result['is_scam'] else 0
        proba = result['ml_score']
        status = "[PASS]" if pred == expected else "[FAIL]"
        if pred == expected: passed += 1
        
        lbl_str = "SCAM" if pred == 1 else "Safe"
        print(f"[{status}] Text: '{text[:40]}...' -> {lbl_str} ({proba:.4f})")
        
    acc = passed / len(test_cases) * 100
    print(f"\nРезультат: {acc:.1f}% ({passed}/{len(test_cases)})")

if __name__ == "__main__":
    test_enrichment()
