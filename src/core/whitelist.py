import os
import hashlib

class Whitelist:
    def __init__(self, filepath="whitelist.txt"):
        self.filepath = filepath
        self.trusted_hashes = set()
        self.load()

    def _get_hash(self, text: str) -> str:
        # Нормализация текста: нижний регистр, удаление пробелов
        normalized = text.lower().strip()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def add(self, text: str):
        """Добавить текст (или отправителя) в белый список"""
        h = self._get_hash(text)
        if h not in self.trusted_hashes:
            self.trusted_hashes.add(h)
            self._append_to_file(h)
            print(f"Добавлено в белый список: {text[:20]}...")

    def is_trusted(self, text: str) -> bool:
        """Проверить, находится ли текст в белом списке"""
        return self._get_hash(text) in self.trusted_hashes

    def load(self):
        if not os.path.exists(self.filepath):
            return
        
        with open(self.filepath, "r") as f:
            for line in f:
                self.trusted_hashes.add(line.strip())

    def _append_to_file(self, h):
        with open(self.filepath, "a") as f:
            f.write(h + "\n")
