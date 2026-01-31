from src.core.classifier import GuardianClassifier
import os

def train():
    print("Starting ONE-TIME training...")
    clf = GuardianClassifier()
    dataset_path = os.path.join(os.getcwd(), 'dataset')
    clf.train(dataset_path)
    print("Done! Model is saved as 'model_hybrid.joblib'.")

if __name__ == "__main__":
    train()
