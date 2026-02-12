import os
import glob
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
import numpy as np

from src.core.patterns import get_compiled_patterns
from src.core.link_hunter import LinkHunter
from src.core.whitelist import Whitelist
from src.core.ner import EntityExtractor
from src.core.ner import EntityExtractor
from src.core.features import FeatureExtractor
from src.core.sentiment import SentimentAnalyzer
from src.utils.text_processing import clean_text, normalize_homoglyphs, generate_homoglyphs
from src import config
import logging

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers не найдены. Используем TF-IDF fallback.")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime не найден.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_curve
from src.core.explainer import ExplainabilityEngine

class GuardianClassifier:
    def __init__(self, model_path=None):
        self.regex_patterns = get_compiled_patterns()
        self.link_hunter = LinkHunter()
        self.whitelist = Whitelist()
        self.ner = EntityExtractor()
        self.feature_extractor = FeatureExtractor()
        self.sentiment = SentimentAnalyzer()
        self.model_path = model_path if model_path else config.MODEL_PATH
        self.use_bert = TRANSFORMERS_AVAILABLE
        self.use_onnx = ONNX_AVAILABLE and os.path.exists(config.ONNX_MODEL_PATH)
        self.threshold = config.SKLEARN_THRESHOLD_DEFAULT
        
        # Инициализация модуля объяснений
        self.explainer = ExplainabilityEngine(self)
        
        if self.use_bert:
            # Гибридная модель: BERT + Статистика
            self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
            self.tokenizer = None
            self.bert_model = None
            self.ort_session = None
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
            logging.info("Загрузка RuBERT...")
            model_name = "cointegrated/rubert-tiny2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if self.use_onnx:
                logging.info(f"Загрузка ONNX модели ({config.ONNX_MODEL_PATH})...")
                try:
                    self.ort_session = ort.InferenceSession(config.ONNX_MODEL_PATH)
                    logging.info("ONNX Runtime готов к работе! 🚀")
                except Exception as e:
                    logging.warning(f"Ошибка загрузки ONNX: {e}. Откат к PyTorch.")
                    self.use_onnx = False
            
            if not self.use_onnx:
                self.bert_model = AutoModel.from_pretrained(model_name)
                # Заморозка BERT для использования как экстрактора признаков
                for param in self.bert_model.parameters():
                    param.requires_grad = False
                logging.info("RuBERT (PyTorch) загружен.")

    def _get_bert_embeddings(self, texts):
        if not self.use_bert:
            return None
        self._init_bert()
        
        embeddings = []
        # Пакетная обработка была бы лучше, но для небольшого датасета подойдет и цикл
        for text in texts:
            t = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
            
            if self.use_onnx:
                # ONNX Inference (Логический вывод через ONNX)
                ort_inputs = {
                    'input_ids': t['input_ids'].numpy(),
                    'attention_mask': t['attention_mask'].numpy()
                }
                ort_outs = self.ort_session.run(None, ort_inputs)
                # Выход 0 - это last_hidden_state
                emb = ort_outs[0][:, 0, :] # Эквивалент токена CLS
                embeddings.append(emb[0])
            else:
                # PyTorch Inference (Логический вывод через PyTorch)
                with torch.no_grad():
                    model_output = self.bert_model(**t)
                # Используем эмбеддинг токена CLS (индекс 0)
                emb = model_output.last_hidden_state[:, 0, :]
                embeddings.append(emb[0].numpy())
                
        return embeddings

    def save_model(self):
        if self.use_bert:
            # Сохранение классификатора и порога
            # Joblib дампит объект, поэтому self.threshold (член класса) сохранится, если мы дампим self.clf? 
            # Нет, self.clf - это просто объект sklearn. 
            # Нам нужно сохранить логику для сохранения порога.
            # Идеально было бы сохранить весь экземпляр GuardianClassifier, НО он может содержать непиклируемые паттерны.
            # Re.Pattern пиклируется в новых версиях python. 
            # Но давайте придерживаться сохранения базовой модели sklearn, чтобы избежать глубокого рефакторинга.
            # Мы можем сохранить порог в отдельном файле или "взломать" объект, если бы мы дампили self.
            # Пока что будем считать, что полагаемся на переобучение или жестко заданный дефолт, ИЛИ сохраняем словарь метаданных.
            # Сохраняем словарь с моделью и порогом.
            state = {
                'model': self.clf,
                'threshold': self.threshold
            }
            joblib.dump(state, self.model_path)
            # Примечание: Это меняет формат файла! load_model должна адаптироваться.
        else:
            joblib.dump(self.ml_pipeline, self.model_path)
        logging.info(f"Модель сохранена в {self.model_path}")
        
    def load_model(self):
        try:
            loaded_obj = joblib.load(self.model_path)
            
            if self.use_bert:
                if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
                    self.clf = loaded_obj['model']
                    self.threshold = loaded_obj.get('threshold', 0.6)
                else:
                    # Обратная совместимость для старого формата (только модель)
                    self.clf = loaded_obj
                    self.threshold = 0.6
                
                # Убедимся, что BERT работает после загрузки классификатора
                self._init_bert()
            else:
                self.ml_pipeline = loaded_obj
                
            self.is_trained = True
            logging.info(f"Модель загружена из {self.model_path} (Threshold: {self.threshold:.4f})")
        except Exception as e:
            logging.error(f"Не удалось загрузить модель: {e}")
            self.is_trained = False
            self.threshold = config.SKLEARN_THRESHOLD_DEFAULT # Значение по умолчанию (Fallback)

    def check_keywords(self, text: str) -> list[str]:
        """Возвращает список сработавших паттернов"""
        triggers = []
        # Нормализация перед проверкой ключевых слов
        text_norm = normalize_homoglyphs(text)
        
        for pattern in self.regex_patterns:
            if pattern.search(text_norm): # Ищем в нормализованном тексте
                triggers.append(pattern.pattern)
        return triggers

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
        
        # Сначала пробуем конкретные файлы
        scam_path = os.path.join(dataset_path, "scam_samples.txt")
        safe_path = os.path.join(dataset_path, "safe_samples.txt")
        
        # Смешанные датасеты (предоставленные пользователем "phishing_dataset_*.txt")
        mixed_files = glob.glob(os.path.join(dataset_path, "phishing_dataset_*.txt"))
        
        data = []
        labels = [] # 1 - scam (мошенничество), 0 - safe (безопасно)
        
        # Загрузка файлов конкретного класса (scam_*.txt)
        for f_path in scam_files:
            base_name = os.path.basename(f_path)
            logging.info(f"Загрузка Scam датасета: {base_name}")
            
            # OVERSAMPLING (Передискретизация) для исправления дисбаланса классов / усиления обучения
            multiplier = 1
            if "enrichment" in base_name or "samples" in base_name:
                multiplier = 50 
                logging.info(f"  -> Применен буст x{multiplier}")

            try:
                with open(f_path, 'r', encoding='utf-8') as f:
                    file_lines = f.readlines()
                    for _ in range(multiplier):
                        for line in file_lines:
                            if line.strip():
                                data.append(line.strip())
                                labels.append(1)
            except Exception as e:
                logging.error(f"Ошибка загрузки {f_path}: {e}")

        # Загрузка файлов конкретного класса (safe_*.txt)
        for f_path in safe_files:
            base_name = os.path.basename(f_path)
            logging.info(f"Загрузка Safe датасета: {base_name}")
            
            # OVERSAMPLING (Передискретизация) безопасных примеров тоже
            multiplier = 1
            if "enrichment" in base_name or "samples" in base_name:
                multiplier = 50
                logging.info(f"  -> Применен буст x{multiplier}")

            try:
                with open(f_path, 'r', encoding='utf-8') as f:
                    file_lines = f.readlines()
                    for _ in range(multiplier):
                        for line in file_lines:
                            if line.strip():
                                data.append(line.strip())
                                labels.append(0)
            except Exception as e:
                logging.error(f"Ошибка загрузки {f_path}: {e}")
                        
        # Загрузка новых смешанных файлов
        for m_file in mixed_files:
            logging.info(f"Загрузка смешанного датасета: {os.path.basename(m_file)}")
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
                logging.error(f"Ошибка загрузки {m_file}: {e}")
                        
        return data, labels

    def train(self, dataset_path: str):
        print("Загрузка датасета...")
        X, y = self.load_dataset(dataset_path)
        if not X:
            print("Датасет пуст!")
            return
            
        print(f"Размер датасета: {len(X)} примеров")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Аугментация данных (Homoglyphs)
        print("Аугментация данных (Homoglyphs)...")
        aug_X = []
        aug_y = []
        for txt, lbl in zip(X_train, y_train):
            if lbl == 1: # Только для SCAM
                # Генерируем с вероятностью 50%
                import random
                if random.random() < 0.5:
                    spoofed = generate_homoglyphs(txt)
                    if spoofed != txt:
                        aug_X.append(spoofed)
                        aug_y.append(1)
        
        if aug_X:
            print(f"Добавлено {len(aug_X)} примеров с гомоглифами.")
            X_train.extend(aug_X)
            y_train.extend(aug_y)
        
        print(f"Обучение ML модели (BERT={self.use_bert})...")
        
        if self.use_bert:
            # Генерация эмбеддингов
            print("Генерация эмбеддингов...")
            X_train_emb = self._get_bert_embeddings(X_train)
            X_test_emb = self._get_bert_embeddings(X_test)
            
            # Генерация статистических признаков
            print("Генерация статистических признаков...")
            X_train_stats = np.array([self.feature_extractor.extract(t) for t in X_train])
            X_test_stats = np.array([self.feature_extractor.extract(t) for t in X_test])
            
            # Анализ эмоций
            print("Анализ эмоций (может занять время)...")
            def get_sent(texts):
                res = []
                for i, t in enumerate(texts):
                    if i % 1000 == 0 and i > 0: print(f"  Processed {i}/{len(texts)}")
                    s = self.sentiment.analyze(t)
                    res.append([s['negative'], s['neutral'], s['positive']])
                return np.array(res)

            X_train_sent = get_sent(X_train)
            X_test_sent = get_sent(X_test)

            # Объединение признаков
            print("Объединение признаков...")
            X_train_combined = np.hstack((X_train_emb, X_train_stats, X_train_sent))
            X_test_combined = np.hstack((X_test_emb, X_test_stats, X_test_sent))
            
            self.clf.fit(X_train_combined, y_train)
            
            # Оценка модели
            y_pred_proba = self.clf.predict_proba(X_test_combined)[:, 1]
            
            # Авто-калибровка порога
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            if len(thresholds) > 0:
                J = tpr - fpr
                ix = np.argmax(J)
                best_thresh = thresholds[ix]
                print(f"Оптимальный порог (Best Threshold): {best_thresh:.4f}")
                self.threshold = float(best_thresh) # Сохраняем в класс
            
            # Применение порога
            y_pred = (y_pred_proba >= self.threshold).astype(int)
            
            print("\n--- ОТЧЕТ О КЛАССИФИКАЦИИ ---")
            print(classification_report(y_test, y_pred, target_names=['Safe', 'Scam']))
            print(f"Precision: {precision_score(y_test, y_pred):.4f}")
            print(f"Recall:    {recall_score(y_test, y_pred):.4f} (Критически важно!)")
            print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
            print("-------------------------------\n")
            
        else:
            # TF-IDF Pipeline
            X_train_clean = [clean_text(t) for t in X_train]
            X_test_clean = [clean_text(t) for t in X_test]
            
            self.ml_pipeline.fit(X_train_clean, y_train)
            predicted = self.ml_pipeline.predict(X_test_clean)
            print(f"Точность модели (TF-IDF): {metrics.accuracy_score(y_test, predicted):.4f}")

        self.is_trained = True
        self.save_model()

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

        # Normalize homoglyphs
        text_norm = normalize_homoglyphs(text)
        text_clean = clean_text(text_norm)
        
        # 1. Regex check (Scam Triggers) - ONLY on current text
        triggers = self.check_keywords(text) 
        
        # 1.1 Safe Pattern check -- ONLY IF NOT STRICT
        is_safe_pattern = False
        if not strict_mode:
            is_safe_pattern = self.check_safe_patterns(text)

        # 2. ML check - Uses Context!
        ml_score = 0.0
        ml_verdict = "Unknown"
        
        # Prepare text for ML: Context + Current
        ml_input_text = text_clean
        if context:
            # Join with [SEP] token
            clean_context = [clean_text(normalize_homoglyphs(c)) for c in context]
            ml_input_text = " [SEP] ".join(clean_context + [text_clean])
        
        if self.is_trained:
            try:
                if self.use_bert:
                     self._init_bert()
                     # 1. Embeddings on Context + Text
                     text_emb = self._get_bert_embeddings([ml_input_text]) 
                     
                     # 2. Statistics (Manual features) - Only on CURRENT text
                     text_stats = self.feature_extractor.extract(text)
                     text_stats = text_stats.reshape(1, -1)
                     
                     # 3. Sentiment
                     sent = self.sentiment.analyze(text)
                     text_sent = np.array([[sent['negative'], sent['neutral'], sent['positive']]])

                     # 4. Combine
                     combined_features = np.hstack((text_emb, text_stats, text_sent))
                     
                     probs = self.clf.predict_proba(combined_features)[0]
                else:
                    # TF-IDF
                    probs = self.ml_pipeline.predict_proba([ml_input_text])[0]
                
                ml_score = probs[1]
            except Exception as e:
                logging.error(f"Ошибка ML: {e}")
                ml_score = 0.0
                
            # Dynamic threshold usage
            ml_verdict = "Scam" if ml_score > self.threshold else "Safe"
            
        # 3. Link Hunter check
        link_analysis = self.link_hunter.analyze(text)
        link_score = link_analysis['score']
        
        # Final decision logic
        final_score = max(ml_score, link_score)
        
        # Default logic uses calibrated ML threshold
        is_scam = bool(bool(triggers) or (ml_score > self.threshold) or (link_score > 0.7))
        reason = "Безопасно"
        
        # NER Check
        entities = self.ner.extract(text_norm)
        
        # --- NER Heuristics ---
        # 1. Fake Authority / Impersonation (MVD, FSB + Urgency/Money)
        risky_orgs = ['МВД', 'ФСБ', 'Полиция', 'Центробанк', 'Госуслуги', 'Следственный комитет']
        found_risky_org = [org for org in entities.get('ORG', []) if any(r.lower() in org.lower() for r in risky_orgs)]
        
        if found_risky_org:
            # If Authority is mentioned, we need Urgency or Sensitive info to call it scam
            if entities.get('URGENCY') or entities.get('SENSITIVE') or "перевод" in text_norm.lower():
                is_scam = True
                reason = f"Подозрительная активность от имени организации: {found_risky_org[0]}"
                # Force high score for UI
                final_score = max(final_score, 0.95)

        if is_safe_pattern:
            if link_score < 0.8: 
                is_scam = False
                reason = "Безопасно (Системное сообщение)"
        
        if is_scam:
            if found_risky_org and "Подозрительная активность" in reason:
                 pass # Reason already set
            elif triggers:
                reason = "Сработал триггер (Ключевые слова)"
            elif link_score > 0.7:
                 reason = f"Фишинговая ссылка: {', '.join([r['reasons'][0] for r in link_analysis['suspicious_links']])}"
            elif ml_score > self.threshold:
                reason = "Высокая уверенность ML"

        # Explainability (Only if not strict to avoid infinite loops)
        explanation = []
        if not strict_mode: 
            explanation = self.explainer.explain(text, final_score, triggers, entities)

        result = {
            "text": text,
            "is_scam": is_scam,
            "triggers": triggers,
            "ml_score": float(round(final_score, 4)),
            "ml_verdict": ml_verdict,
            "reason": reason,
            "link_analysis": link_analysis,
            "entities": entities,
            "explanation": explanation
        }
        return result

    async def predict_async(self, text: str, strict_mode: bool = False, context: list[str] = None) -> dict:
        """Asynchronous wrapper for predict"""
        return await asyncio.to_thread(self.predict, text, strict_mode, context)
