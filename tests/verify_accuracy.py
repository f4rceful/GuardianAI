from src.core.classifier import GuardianClassifier
import os

def check_examples():
    clf = GuardianClassifier()
    if not clf.is_trained:
        print("Model not trained!")
        return

    file_path = "exampels.txt"
    if not os.path.exists(file_path):
        print("exampels.txt not found!")
        return
        
    total = 0
    correct = 0
    errors = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"{'EXPECTED':<10} | {'PREDICTED':<10} | {'TEXT'}")
        print("-" * 80)
        
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Parse line "Type: "Text""
            parts = line.split(':', 1)
            if len(parts) < 2: continue
            
            expected_type = parts[0].strip().upper() # SCAM or SAFE
            text = parts[1].strip().strip('"')
            
            # Predict
            result = clf.predict(text)
            # predict returns a DICT now, not tuple (is_scam, reason, ...)
            # Wait, let's check predict method signature again.
            # Looking at source code lines 323-338... It returns a DICT.
            is_scam = result.get('is_scam', False)
            predicted_type = "SCAM" if is_scam else "SAFE"
            
            total += 1
            if predicted_type == expected_type:
                correct += 1
            else:
                reason = result.get('reason', 'Unknown')
                errors.append(f"Expected: {expected_type}, Got: {predicted_type} | Reason: {reason} | Text: {text}")

    print("-" * 80)
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    if total > 0:
        print(f"Accuracy: {correct/total*100:.2f}%")
    
    if errors:
        print("\nERRORS (Не прошли проверку):")
        for err in errors:
            print(err)
    else:
        print("\nAll examples passed successfully! 🎉")

if __name__ == "__main__":
    check_examples()
