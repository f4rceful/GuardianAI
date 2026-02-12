from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import uvicorn
from contextlib import asynccontextmanager

# Добавляем корень проекта в путь
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.classifier import GuardianClassifier
from src import config
import logging

classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Загрузка модели при старте
    config.setup_logging()
    global classifier
    logging.info("⏳ Инициализация классификатора GuardianAI...")
    classifier = GuardianClassifier()
    # Загрузка ONNX если есть
    classifier._init_bert()
    logging.info("✅ Модель загружена и готова к работе.")
    yield
    # Очистка ресурсов

app = FastAPI(title="GuardianAI API", version="2.0", lifespan=lifespan)

class PredictionRequest(BaseModel):
    text: str
    strict_mode: bool = False
    context: list[str] = []

class PredictionResponse(BaseModel):
    is_scam: bool
    score: float
    reason: list[str]
    verdict: str
    entities: dict
    explanation: list[dict]

@app.get("/")
def read_root():
    return {"status": "active", "service": "GuardianAI", "version": "2.1"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not classifier:
        raise HTTPException(status_code=503, detail="Модель не инициализирована")
    
    try:
        # Прогноз
        result = classifier.predict(request.text, strict_mode=request.strict_mode, context=request.context)
        
        # Распаковка результата
        is_scam = result["is_scam"]
        score = result["ml_score"]
        verdict = result["ml_verdict"]
        
        # Нормализация списка причин
        reason_list = result["triggers"] if result["triggers"] else ([result["reason"]] if isinstance(result["reason"], str) else result["reason"])
        if isinstance(reason_list, str): reason_list = [reason_list]

        return PredictionResponse(
            is_scam=is_scam,
            score=score,
            reason=reason_list,
            verdict=verdict,
            entities=result.get("entities", {}),
            explanation=result.get("explanation", [])
        )
    except Exception as e:
        logging.error(f"Ошибка во время прогнозирования: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class FeedbackRequest(BaseModel):
    text: str
    is_scam_report: bool # True = СПАМ, False = НЕ СПАМ
    original_score: float = 0.0

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    try:
        feedback_file = config.FEEDBACK_FILE
        
        # Создание директории
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
        
        # Проверка существования файла
        file_exists = os.path.isfile(feedback_file)
        
        with open(feedback_file, 'a', encoding='utf-8') as f:
            if not file_exists:
                f.write("text,user_verdict,original_score\n")
            
            clean_text = request.text.replace('"', '""').replace('\n', ' ')
            verdict = "SCAM" if request.is_scam_report else "SAFE"
            f.write(f'"{clean_text}",{verdict},{request.original_score}\n')
            
        return {"status": "success", "message": "Отзыв сохранен"}
    except Exception as e:
        logging.error(f"Ошибка сохранения отзыва: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
