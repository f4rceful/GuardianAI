import os
import glob
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import joblib

# Пробуем импортировать transformers
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers не найдены. Используем TF-IDF fallback.")

# Импорт ансамблевых моделей
from sklearn.ensemble import RandomForestClassifier

class GuardianClassifier:
    def __init__(self, model_path="model_hybrid.joblib"):
        self.regex_patterns = get_compiled_patterns()
        self.link_hunter = LinkHunter()
        self.whitelist = Whitelist()
        self.ner = EntityExtractor()
        self.feature_extractor = FeatureExtractor()
        self.model_path = model_path
        self.use_bert = TRANSFORMERS_AVAILABLE
        
        if self.use_bert:
            # Гибридная модель: RuBERT признаки + Статистические признаки
            # RandomForest достаточно надежен, чтобы обработать их объединение
            self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
            self.tokenizer = None
            self.bert_model = None
        else:
            self.ml_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='char_wb', max_features=5000)),
                ('clf', SGDClassifier(loss='log_loss', random_state=42)),
            ])
        
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            self.is_trained = False

    def _init_bert(self):
        if self.tokenizer is None:
            print("Загрузка RuBERT...")
            model_name = "cointegrated/rubert-tiny2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            # Замораживаем BERT, чтобы использовать как экстрактор признаков
            for param in self.bert_model.parameters():
                param.requires_grad = False
            print("RuBERT загружен.")

    def _get_bert_embeddings(self, texts):
        if not self.use_bert:
            return None
        self._init_bert()
        
        embeddings = []
        # Пакетная обработка была бы лучше, но пока простой цикл пойдет для небольшого датасета
        for text in texts:
            t = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
            with torch.no_grad():
                model_output = self.bert_model(**t)
            # Используем эмбеддинг токена CLS (индекс 0)
            emb = model_output.last_hidden_state[:, 0, :]
            embeddings.append(emb[0].numpy())
        return embeddings

    def save_model(self):
        if self.use_bert:
            joblib.dump(self.clf, self.model_path)
        else:
            joblib.dump(self.ml_pipeline, self.model_path)
        print(f"Модель сохранена в {self.model_path}")
        
    def load_model(self):
        try:
            if self.use_bert:
                self.clf = joblib.load(self.model_path)
                # Убедимся, что BERT работает даже после загрузки классификатора
                self._init_bert()
            else:
                self.ml_pipeline = joblib.load(self.model_path)
            self.is_trained = True
            print(f"Модель загружена из {self.model_path}")
        except Exception as e:
            print(f"Не удалось загрузить модель: {e}")
            self.is_trained = False

    def check_keywords(self, text: str) -> list[str]:
        """Возвращает список сработавших паттернов"""
        matches = []
        for pattern in self.regex_patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)
        return matches

    def check_safe_patterns(self, text: str) -> bool:
        """Проверка на сервисные сообщения (коды, пароли), которые НЕ надо блокировать"""
        safe_triggers = [
            r"код\s*[:\-]?\s*\d+",
            r"code\s*[:\-]?\s*\d+",
            r"пароль\s*[:\-]?\s*\d+",
            r"password\s*[:\-]?\s*\d+",
            r"никому не сообщайте",
            r"don'?t share",
            r"ваш код",
            r"your code"
        ]
        import re
        for pattern in safe_triggers:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def load_dataset(self, dataset_path: str):
        scam_files = glob.glob(os.path.join(dataset_path, "scam_*.txt"))
        safe_files = glob.glob(os.path.join(dataset_path, "safe_*.txt"))
        
        # Пробуем сначала конкретные файлы
        scam_path = os.path.join(dataset_path, "scam_samples.txt")
        safe_path = os.path.join(dataset_path, "safe_samples.txt")
        
        # Смешанные датасеты (Пользователь предоставил "phishing_dataset_*.txt")
        mixed_files = glob.glob(os.path.join(dataset_path, "phishing_dataset_*.txt"))
        
        data = []
        labels = [] # 1 - scam, 0 - safe
        
        # Загружаем файлы старого разделения
        if os.path.exists(scam_path):
            with open(scam_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(line.strip())
                        labels.append(1)
        if os.path.exists(safe_path):
            with open(safe_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(line.strip())
                        labels.append(0)
                        
        # Загружаем новые смешанные файлы
        for m_file in mixed_files:
            print(f"Загрузка смешанного датасета: {os.path.basename(m_file)}")
            try:
                with open(m_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        
                        # Формат: 'Scam: "Text"' или 'Safe: "Text"'
                        if line.startswith("Scam:"):
                            content = line[5:].strip().strip('"')
                            data.append(content)
                            labels.append(1)
                        elif line.startswith("Safe:"):
                            content = line[5:].strip().strip('"')
                            data.append(content)
                            labels.append(0)
            except Exception as e:
                print(f"Ошибка загрузки {m_file}: {e}")
                        
        return data, labels

    def train(self, dataset_path: str):
        print("Загрузка датасета...")
        X, y = self.load_dataset(dataset_path)
        if not X:
            print("Датасет пуст!")
            return
            
        print(f"Размер датасета: {len(X)} примеров")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Обучение ML модели (BERT={self.use_bert})...")
        
        if self.use_bert:
            # 1. Эмбеддинги (Семантика)
            print("Генерация эмбеддингов...")
            X_train_emb = self._get_bert_embeddings(X_train)
            X_test_emb = self._get_bert_embeddings(X_test)
            
            # 2. Статистические признаки (Стилистика)
            print("Генерация статистических признаков...")
            X_train_stats = np.array([self.feature_extractor.extract(t) for t in X_train])
            X_test_stats = np.array([self.feature_extractor.extract(t) for t in X_test])
            
            # 3. Объединение: [Batch, 312] + [Batch, 5] -> [Batch, 317]
            X_train_combined = np.hstack((X_train_emb, X_train_stats))
            X_test_combined = np.hstack((X_test_emb, X_test_stats))
            
            self.clf.fit(X_train_combined, y_train)
            predicted = self.clf.predict(X_test_combined)
        else:
            # TF-IDF Pipeline expects raw text usually, but we have our own clean_text
            # Let's clean it here explicitly if we want to match previous behavior
            # Or assume TfidfVectorizer handles it (it does lowercasing by default)
            # But our clean_text does extra stuff. Let's map it.
            X_train_clean = [clean_text(t) for t in X_train]
            X_test_clean = [clean_text(t) for t in X_test]
            
            self.ml_pipeline.fit(X_train_clean, y_train)
            predicted = self.ml_pipeline.predict(X_test_clean)

        self.is_trained = True
        self.save_model()
        
        print(f"Accuracy: {metrics.accuracy_score(y_test, predicted)}")
        print(metrics.classification_report(y_test, predicted, target_names=['Safe', 'Scam'], zero_division=0))

    def predict(self, text: str, strict_mode: bool = False, context: list[str] = None) -> dict:
        # 0. Whitelist check
        if self.whitelist.is_trusted(text):
             return {
                "text": text,
                "is_scam": False,
                "triggers": [],
                "ml_score": 0.0,
                "ml_verdict": "Safe",
                "reason": "В белом списке (Доверенный)",
                "link_analysis": {"score": 0.0, "has_links": False}
            }

        text_clean = clean_text(text)
        
        # 1. Regex проверка (Триггеры скама) - ТОЛЬКО на текущем тексте
        triggers = self.check_keywords(text_clean)
        
        # 1.1 Проверка безопасных паттернов (OTP, Системные сообщения) - ТОЛЬКО ЕСЛИ НЕ СТРОГИЙ РЕЖИМ
        is_safe_pattern = False
        if not strict_mode:
            is_safe_pattern = self.check_safe_patterns(text)

        # 2. ML проверка - Использует Контекст!
        ml_score = 0.0
        ml_verdict = "Unknown"
        
        # Подготовка текста для ML: Контекст + Текущий
        ml_input_text = text_clean
        if context:
            # Соединяем токеном [SEP], который BERT понимает как разделитель
            # Очистка контекстных сообщений тоже хорошая практика
            clean_context = [clean_text(c) for c in context]
            ml_input_text = " [SEP] ".join(clean_context + [text_clean])
            # print(f"DEBUG: ML Input with Context: {ml_input_text}") 
        
        if self.is_trained:
            try:
                if self.use_bert:
                     self._init_bert()
                     # 1. Эмбеддинги на Контексте + Тексте
                     text_emb = self._get_bert_embeddings([ml_input_text]) 
                     
                     # 2. Статистика (Ручные признаки) - Только на ТЕКУЩЕМ тексте
                     text_stats = self.feature_extractor.extract(text)
                     text_stats = text_stats.reshape(1, -1)
                     
                     # 3. Объединение
                     combined_features = np.hstack((text_emb, text_stats))
                     
                     probs = self.clf.predict_proba(combined_features)[0]
                else:
                    # TF-IDF
                    # Примечание: TF-IDF может не обрабатывать [SEP] правильно, но просто будет считать его токеном или проигнорирует.
                    # Он все равно добавляет 'мешок слов' из истории, что полезно.
                    probs = self.ml_pipeline.predict_proba([ml_input_text])[0]
                
                ml_score = probs[1]
            except Exception as e:
                print(f"Ошибка ML: {e}")
                ml_score = 0.0
                
            ml_verdict = "Scam" if ml_score > 0.5 else "Safe"
            
        # 3. Проверка Link Hunter - ТОЛЬКО на текущем тексте
        link_analysis = self.link_hunter.analyze(text)
        link_score = link_analysis['score']
        
        # Логика финального решения
        final_score = max(ml_score, link_score)
        
        # Логика по умолчанию
        is_scam = bool(bool(triggers) or (ml_score > 0.6) or (link_score > 0.7))
        reason = "Безопасно"
        
        # Проверка NER
        entities = self.ner.extract(text)

        if is_safe_pattern:
            if link_score < 0.8: 
                is_scam = False
                reason = "Безопасно (Системное сообщение)"
        
        if is_scam:
            if triggers:
                reason = "Сработал триггер (Ключевые слова)"
            elif link_score > 0.7:
                 reason = f"Фишинговая ссылка: {', '.join([r['reasons'][0] for r in link_analysis['suspicious_links']])}"
            elif ml_score > 0.6:
                reason = "Высокая уверенность ML"

        result = {
            "text": text,
            "is_scam": is_scam,
            "triggers": triggers,
            "ml_score": float(round(final_score, 4)),
            "ml_verdict": ml_verdict,
            "reason": reason,
            "link_analysis": link_analysis,
            "entities": entities
        }
        return result

    async def predict_async(self, text: str, strict_mode: bool = False, context: list[str] = None) -> dict:
        """Asynchronous wrapper for predict"""
        return await asyncio.to_thread(self.predict, text, strict_mode, context)
