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

def normalize_homoglyphs(text: str) -> str:
    """
    Заменяет латинские символы, похожие на кириллицу, на их кириллические аналоги.
    Это помогает бороться с обходом фильтров (например, 'o' латинская -> 'о' кириллическая).
    """
    homoglyphs = {
        'a': 'а', 'A': 'А',
        'e': 'е', 'E': 'Е',
        'o': 'о', 'O': 'О',
        'p': 'р', 'P': 'Р',
        'c': 'с', 'C': 'С',
        'x': 'х', 'X': 'Х',
        'y': 'у', 'Y': 'У',
        'T': 'Т',
        'H': 'Н',
        'K': 'К',
        'M': 'М',
        'B': 'В'
    }
    
    # Строим таблицу перевода
    table = str.maketrans(homoglyphs)
    return text.translate(table)
