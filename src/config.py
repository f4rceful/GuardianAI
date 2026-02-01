import os
import logging
import sys

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # GuardianAI/
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# File Paths
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "model_hybrid.joblib")
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "rubert_tiny2.onnx")

# Logging Configuration
LOG_FILE = os.path.join(LOGS_DIR, "guardian.log")

def setup_logging():
    """Configures logging for the application"""
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logging.info(f"Logging initialized. Logs directory: {LOGS_DIR}")

# App Settings
HOST = "0.0.0.0"
PORT = 8550
SKLEARN_THRESHOLD_DEFAULT = 0.6
