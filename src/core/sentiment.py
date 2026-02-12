import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "cointegrated/rubert-tiny-sentiment-balanced"
        self.tokenizer = None
        self.model = None
        self._is_loaded = False

    def _load_model(self):
        if self._is_loaded:
            return
        
        try:
            logging.info(f"Загрузка модели анализа эмоций: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._is_loaded = True
        except Exception as e:
            logging.error(f"Не удалось загрузить модель анализа эмоций: {e}")

    def analyze(self, text: str) -> dict:
        """
        Возвращает оценки тональности:
        - negative (Негатив: Страх, Гнев)
        - neutral (Нейтрально)
        - positive (Позитив: Радость, Доверие)
        """
        if not self._is_loaded:
            self._load_model()
            if not self._is_loaded:
                return {"negative": 0.0, "neutral": 0.0, "positive": 0.0}

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)[0].tolist()
            
            # Сопоставление меток для этой конкретной модели:
            # 0: negative, 1: neutral, 2: positive
            return {
                "negative": probs[0],
                "neutral": probs[1],
                "positive": probs[2]
            }
        except Exception as e:
            logging.error(f"Ошибка анализа тональности: {e}")
            return {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
