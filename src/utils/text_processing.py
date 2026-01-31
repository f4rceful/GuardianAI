import re
import string

def clean_text(text: str) -> str:
    """
    Очищает текст: удаляет лишние пробелы, приводит к нижнему регистру.
    В будущем можно добавить стемминг/лемматизацию.
    """
    if not text:
        return ""
    
    text = text.lower()
    # Удаляем пунктуацию (опционально, зависит от того, нужны ли нам знаки вопроса/восклицания)
    # text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text
