import re
from urllib.parse import urlparse

class LinkHunter:
    def __init__(self):
        # Индикаторы подозрительного риска (TLD)
        self.suspicious_tlds = {'.xyz', '.top', '.club', '.online', '.site', '.work', '.info', '.click', '.gq', '.ml', '.cf', '.tk', '.ga'}
        self.ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
        
        # Кириллические символы, похожие на латиницу (спуфинг)
        self.cyrillic_chars = set("асеогхуі") 
        
    def extract_urls(self, text: str) -> list[str]:
        """Найти все ссылки в тексте"""
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.findall(text)

    def analyze(self, text: str) -> dict:
        """Анализ текста на фишинговые ссылки"""
        urls = self.extract_urls(text)
        if not urls:
            return {"has_links": False, "suspicious": [], "score": 0.0}

        suspicious_links = []
        score = 0.0

        for url in urls:
            risk = self._analyze_single_url(url)
            if risk['is_suspicious']:
                suspicious_links.append(risk)
                score = max(score, risk['score'])

        return {
            "has_links": True,
            "suspicious_links": suspicious_links,
            "score": score
        }

    def _analyze_single_url(self, url: str) -> dict:
        verdict = {
            "url": url,
            "is_suspicious": False,
            "reasons": [],
            "score": 0.0
        }
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().split(':')[0] # remove port if any
            
            # Проверка 1: IP-адрес вместо домена
            if self.ip_pattern.match(domain):
                verdict['is_suspicious'] = True
                verdict['reasons'].append("IP-адрес вместо домена")
                verdict['score'] = 1.0
                return verdict # Мгновенный красный флаг
                
            # Проверка 2: Подозрительные доменные зоны (TLD)
            for tld in self.suspicious_tlds:
                if domain.endswith(tld):
                    verdict['is_suspicious'] = True
                    verdict['reasons'].append(f"Подозрительная доменная зона ({tld})")
                    verdict['score'] = max(verdict['score'], 0.8)
            
            # Проверка 3: Смешанные скрипты (Кириллица + Латиница) - Homograph attack
            # Базовая проверка
            has_latin = bool(re.search(r'[a-z]', domain))
            has_cyrillic = bool(re.search(r'[а-я]', domain))
            if has_latin and has_cyrillic:
                 verdict['is_suspicious'] = True
                 verdict['reasons'].append("Смешанные символы (Атака гомографов)")
                 verdict['score'] = 1.0

        except Exception as e:
            print(f"Ошибка анализа URL {url}: {e}")
            
        return verdict
