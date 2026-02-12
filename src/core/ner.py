import re
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsNERTagger,
    Doc
)

class EntityExtractor:
    def __init__(self):
        # 1. Инициализация Natasha (Тяжеловесный ML)
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.ner_tagger = NewsNERTagger(self.emb)
        
        # 2. Пользовательские Regex (Легковесные правила для того, что Natasha не видит)
        self.custom_patterns = {
            "URGENCY": [
                r"срочно", r"быстро", r"немедленно", r"сейчас", 
                r"в течение", r"истекает", r"через час"
            ],
            "SENSITIVE": [
                r"код", r"пароль", r"cvv", r"cvc", r"пин", r"логин",
                r"паспорт", r"снилс", r"госуслуги", r"реквизиты"
            ],
            "RELATIVES": [
                 r"мам[а-я]*", r"пап[а-я]*", r"бабушк[а-я]*", r"сыно[к-я]*", 
                 r"доч[ь-я]*", r"внук", r"внучк[а-я]*"
            ]
        }
    
    def extract(self, text: str) -> dict:
        found = {
            "PER": [],
            "ORG": [],
            "LOC": [],
            "URGENCY": [],
            "SENSITIVE": [],
            "RELATIVES": []
        }
        
        # --- 1. Извлечение с помощью Natasha ---
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_ner(self.ner_tagger)
        
        for span in doc.spans:
            span.normalize(self.morph_vocab)
            if span.type in found:
                found[span.type].append(span.normal)

        # Удаление дубликатов
        for k in ["PER", "ORG", "LOC"]:
            found[k] = list(set(found[k]))
            
        # --- 2. Извлечение с помощью пользовательских Regex ---
        text_lower = text.lower()
        for entity_type, patterns in self.custom_patterns.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches.append(pattern.replace(r"[а-я]*", ""))
            
            if matches:
                found[entity_type] = list(set([m.replace('\\', '') for m in matches]))
                
        # Очистка пустых ключей
        return {k: v for k, v in found.items() if v}
