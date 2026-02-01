import json
import os
from datetime import datetime
from src import config

class HistoryManager:
    def __init__(self, filepath=None):
        self.filepath = filepath if filepath else config.HISTORY_FILE
        self.history = self.load_history()

    def load_history(self):
        if not os.path.exists(self.filepath):
            return []
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def add_entry(self, entry: dict):
        """
        entry: словарь с ключами 'text', 'is_scam', 'ml_score', 'timestamp', и т.д.
        """
        # Добавить временную метку, если отсутствует
        if 'timestamp' not in entry:
            entry['timestamp'] = datetime.now().isoformat()
            
        self.history.insert(0, entry) # Добавляем в начало
        self.save_history()

    def save_history(self):
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)

    def get_stats(self):
        total = len(self.history)
        scam = sum(1 for e in self.history if e.get('is_scam', False))
        safe = total - scam
        return {
            "total": total,
            "scam": scam,
            "safe": safe
        }

    def get_recent_context(self, limit: int = 3) -> list[str]:
        """Возвращает список текстов последних 'limit' сообщений, от старых к новым."""
        # История хранится от новых к старым, поэтому берем первые 'limit' и разворачиваем
        recent = self.history[:limit]
        return [entry['text'] for entry in reversed(recent)]
