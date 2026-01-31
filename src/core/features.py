import re
import numpy as np

class FeatureExtractor:
    def extract(self, text: str) -> np.ndarray:
        """
        Извлекает статистические признаки из текста.
        Возвращает numpy массив формы (n_features,).
        """
        text_len = len(text)
        if text_len == 0:
            return np.array([0, 0, 0, 0, 0])

        # 1. Соотношение CAPS (Агрессия)
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / text_len

        # 2. Количество пунктуации (Срочность)
        # Считаем !, ? и многоточия...
        punct_count = text.count('!') + text.count('?')
        
        # 3. Соотношение цифр (Коды, Деньги)
        digits_count = sum(1 for c in text if c.isdigit())
        digits_ratio = digits_count / text_len

        # 4. Наличие URL
        # Простая проверка, LinkHunter делает детальную проверку, а это для признаков ML
        has_url = 1 if re.search(r'(http|www\.|t\.me|tg:)', text, re.IGNORECASE) else 0
        
        # 5. Длина (очень короткие или очень длинные)
        # Нормализованная длина (например, до 500 символов)
        norm_len = min(text_len, 500) / 500.0

        return np.array([caps_ratio, punct_count, digits_ratio, has_url, norm_len])
