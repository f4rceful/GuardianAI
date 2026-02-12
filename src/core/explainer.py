import re
import numpy as np

class ExplainabilityEngine:
    def __init__(self, classifier):
        self.clf = classifier

    def explain(self, text: str, initial_score: float, triggers: list, entities: dict) -> dict:
        """
        Генерирует объяснение для вердикта о мошенничестве.
        Возвращает список 'Highlights' (слово, тип, оценка влияния).
        """
        words = text.split()
        highlights = []
        
        # Статические выделения (Правила и NER)
        # Мы предпочитаем их, так как это "Твердые" доказательства
        
        # Выделение триггеров
        for pattern_str in triggers:
            # Перекомпилируем, чтобы найти спан (интервал)
            try:
                for match in re.finditer(pattern_str, text, re.IGNORECASE):
                    highlights.append({
                        "span": match.span(),
                        "word": match.group(),
                        "type": "TRIGGER",
                        "impact": 1.0 # Подразумевает максимальное влияние
                    })
            except:
                pass

        # Выделение сущностей (NER)
        for ent_type, vals in entities.items():
            for val in vals:
                # Простой поиск строки значения сущности в тексте
                # В реальном приложении лучше использовать span из NER модели
                start = text.lower().find(val.lower()) # Базовый поиск
                if start != -1:
                    highlights.append({
                        "span": (start, start + len(val)),
                        "word": text[start:start+len(val)],
                        "type": f"NER_{ent_type}",
                        "impact": 0.8
                    })

        # Динамический анализ влияния (Упрощенный LIME/Occlusion)
        # Маскируем каждое слово и проверяем падение ML score
        
        # Делаем это, только если ML Score достаточно высок
        if initial_score > 0.4: 
            base_score = initial_score
            
            # Оптимизация: для коротких сообщений пословный перебор приемлем
            clean_words = [w.strip(".,!?:") for w in words]
            
            for i, word in enumerate(clean_words):
                if len(word) < 3: continue 
                
                # Создаем возмущенный текст (удаляем слово)
                perturbed_text = text.replace(word, "", 1)
                
                # Предикт (быстрый режим)
                res = self.clf.predict(perturbed_text, strict_mode=True)
                new_score = res['ml_score']
                
                drop = base_score - new_score
                
                # Если удаление слова значительно снизило оценку (>0.1), оно важно
                if drop > 0.05:
                    highlights.append({
                        "span": None, # Сложно отследить спан после замены
                        "word": word,
                        "type": "ML_FACTOR",
                        "impact": round(drop, 3)
                    })

        # Форматирование вывода
        # Дедупликация и приоритезация
        unique_highlights = {}
        for h in highlights:
            key = h['word'].lower()
            if key not in unique_highlights or h['impact'] > unique_highlights[key]['impact']:
                unique_highlights[key] = h
                
        return list(unique_highlights.values())

    def visualize(self, text, highlights):
        """Возвращает визуализированный текст, удобный для консоли"""
        vis_text = text
        # Сортируем выделения по длине (по убыванию), чтобы избежать некорректной замены подстрок
        sorted_h = sorted(highlights, key=lambda x: len(x['word']), reverse=True)
        
        for h in sorted_h:
            word = h['word']
            tag = h['type']
            # Цветовые коды для консоли
            # КРАСНЫЙ для Триггера, ЖЕЛТЫЙ для ML, ГОЛУБОЙ для NER
            replacement = f"[{word}]"
            
            if "TRIGGER" in tag:
                replacement = f"🔴[{word}]"
            elif "NER" in tag:
                replacement = f"🔵[{word}]"
            elif "ML" in tag:
                 replacement = f"⚠️[{word}]"
                 
            vis_text = vis_text.replace(word, replacement)
            
        return vis_text
