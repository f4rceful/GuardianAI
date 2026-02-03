from src.core.classifier import GuardianClassifier

def test_explainability():
    print("Загрузка модели...")
    clf = GuardianClassifier() # Loads existing model
    
    test_cases = [
        "Срочно переведи деньги на карту Тинькофф, мама в беде!",
        "Ваша карта заблокирована, сообщите код из смс сотруднику безопасности.",
        "Привет, как дела? Пойдешь гулять?",
        "Я выиграл миллион в лотерее! Переходи по ссылке winner.com"
    ]
    
    print("\n" + "="*60)
    print("             TESTING EXPLAINABILITY")
    print("="*60)
    
    for text in test_cases:
        res = clf.predict(text)
        vis = clf.explainer.visualize(text, res['explanation'])
        
        print(f"\nOriginal: {text}")
        print(f"Verdict:  {'SCAM' if res['is_scam'] else 'Safe'} ({res['ml_score']:.2f})")
        print(f"Explained: {vis}")
        
        if res['explanation']:
            print("Impact Factors:")
            for h in res['explanation']:
                print(f" - {h['word']} ({h['type']}): {h['impact']:.2f}")
        print("-" * 60)

if __name__ == "__main__":
    test_explainability()
