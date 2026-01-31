import re

class EntityExtractor:
    def __init__(self):
        self.entities = {
            "AUTHORITY": [
                r"мвд", r"фсб", r"полиция", r"следственный комитет", 
                r"центробанк", r"цб рф", r"госуслуги", r"налоговая", r"суд"
            ],
            "RELATIVE": [
                r"мам[а-я]*", r"пап[а-я]*", r"бабушк[а-я]*", r"сыно[к-я]*", 
                r"доч[ь-я]*", r"внук", r"внучк[а-я]*", r"друг", r"подруг[а-я]*"
            ],
            "FINANCE": [
                r"сбер", r"тинькофф", r"в тб", r"альфа", r"банк", 
                r"счет[а-у]", r"карт[а-ы]", r"кредит", r"займ"
            ],
            "URGENCY": [
                r"срочно", r"быстро", r"немедленно", r"сейчас", 
                r"в течение", r"истекает", r"через час"
            ],
            "SENSITIVE": [
                r"код", r"пароль", r"cvv", r"cvc", r"пин", r"логин",
                r"паспорт", r"снилс"
            ]
        }
    
    def extract(self, text: str) -> dict:
        text_lower = text.lower()
        found = {}
        
        for entity_type, patterns in self.entities.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches.append(pattern.replace(r"[а-я]*", "").replace(r"[а-у]", "")) 
                    # Упрощенное название совпадения для отображения
            
            if matches:
                # Удаляем дубликаты и берем уникальные базовые слова
                found[entity_type] = list(set([m.replace('\\', '') for m in matches]))
                
        return found
