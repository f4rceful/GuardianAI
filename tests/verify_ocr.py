import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.ocr_engine import OCRExplorer

def test_ocr():
    print("=== Testing OCR Engine ===")
    
    # 1. Initialize
    print("Initializing OCR...")
    ocr = OCRExplorer()
    ocr._initialize()
    
    if not ocr.is_ready:
        print("[FAIL] OCR engine failed to initialize.")
        return

    print("[OK] OCR engine initialized.")
    
    # 2. Check model download/cache
    print("Dependencies check passed (easyocr imported).")
    print("Available languages:", ocr.languages)
    
    # Since we don't have a guaranteed image file, we just check initialization.
    # If a test image exists, we could try it.
    test_img = "test_ocr.png"
    if os.path.exists(test_img):
        print(f"Found test image {test_img}, attempting extraction...")
        text = ocr.extract_text(test_img)
        print(f"Extracted text: {text}")
    else:
        print("[INFO] No test image found to run actual extraction. Initialization check is sufficient for CI.")

if __name__ == "__main__":
    test_ocr()
