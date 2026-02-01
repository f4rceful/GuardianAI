from src.core.classifier import GuardianClassifier
import os

def train():
    print("Запуск ОДНОРАЗОВОГО обучения...")
    clf = GuardianClassifier()
    dataset_path = os.path.join(os.getcwd(), 'dataset')
    clf.train(dataset_path)
    print("Готово! Модель сохранена как 'model_hybrid.joblib'.")

if __name__ == "__main__":
    train()
