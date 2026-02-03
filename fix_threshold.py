from src.core.classifier import GuardianClassifier

def fix_threshold():
    print("Загрузка модели...")
    clf = GuardianClassifier()
    clf.load_model()
    
    print(f"Текущий порог: {clf.threshold}")
    
    # Force set to 0.5
    new_thresh = 0.5
    clf.threshold = new_thresh
    
    print(f"Установка нового порога: {new_thresh}")
    clf.save_model()
    print("Модель сохранена.")

if __name__ == "__main__":
    fix_threshold()
