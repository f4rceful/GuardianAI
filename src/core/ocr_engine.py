import easyocr
import os
import logging

class OCRExplorer:
    def __init__(self, languages=['ru', 'en']):
        """
        Инициализация EasyOCR Reader.
        :param languages: Список языков для распознавания (по умолчанию ru, en)
        """
        self.languages = languages
        self.reader = None
        self.is_ready = False

    def _initialize(self):
        """Ленивая инициализация, чтобы не грузить память при старте приложения"""
        if self.reader is None:
            print("Инициализация OCR модели... (Это может занять время при первом запуске)")
            try:
                # gpu=True если есть CUDA, иначе False автоматически, но можно форсировать
                self.reader = easyocr.Reader(self.languages) 
                self.is_ready = True
                print("OCR модель готова.")
            except Exception as e:
                print(f"Ошибка инициализации OCR: {e}")
                self.is_ready = False

    def extract_text(self, image_path: str) -> str:
        """
        Извлекает текст из изображения.
        :param image_path: Путь к файлу изображения
        :return: Распознанный текст одной строкой
        """
        if not os.path.exists(image_path):
            return ""

        self._initialize()
        if not self.is_ready:
            return "Error: OCR engine not initialized."

        try:
            # detail=0 возвращает просто список строк
            result = self.reader.readtext(image_path, detail=0, paragraph=True)
            text = " ".join(result)
            return text
        except Exception as e:
            print(f"Ошибка OCR распознавания: {e}")
            return ""
