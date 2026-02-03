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

def generate_homoglyphs(text: str) -> str:
    """
    Создает "испорченный" текст, заменяя кириллицу на похожую латиницу.
    Используется для Adversarial Training.
    """
    # Обратное отображение: Кириллица -> Латиница
    spoof_map = {
        'а': 'a', 'А': 'A',
        'е': 'e', 'Е': 'E',
        'о': 'o', 'О': 'O',
        'р': 'p', 'Р': 'P',
        'с': 'c', 'С': 'C',
        'х': 'x', 'Х': 'X',
        'у': 'y', 'У': 'Y',
        'Т': 'T',
        'Н': 'H',
        'К': 'K',
        'М': 'M',
        'В': 'B'
    }
    
    chars = list(text)
    import random
    
    for i, char in enumerate(chars):
        if char in spoof_map and random.random() < 0.3: # Заменяем 30% подходящих букв
            chars[i] = spoof_map[char]
            
    return "".join(chars)
