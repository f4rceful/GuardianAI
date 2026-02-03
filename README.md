# GuardianAI Core 🛡️

**Next-Gen Phishing & Scam Detection Engine**

GuardianAI is an advanced AI backend designed to protect users from social engineering, phishing, and scam attacks. It powers the **GuardianAI Android App** and Desktop clients, providing real-time text analysis with explainable verdicts.

![Status](https://img.shields.io/badge/Status-Production-green) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![API](https://img.shields.io/badge/API-FastAPI-009688) ![Model](https://img.shields.io/badge/Model-RuBERT%20%2B%20ONNX-orange)

---

## 🧠 Key Features

### 1. Hybrid Intelligence Architecture

Unlike simple keyword filters, GuardianAI combines three layers of defense:

* **Deep Learning (RuBERT Tiny2)**: Transformer-based model optimized for Russian language nuance and intent understanding (exported to ONNX for speed).
* **Machine Learning (Random Forest)**: Analyzes statistical features of the text.
* **Heuristic Engine**: Regex-based detection for known scam patterns, crypto-wallets, and malicious links.

### 2. Explainable AI (XAI) 💡

The system doesn't just say "SCAM" — it explains **WHY**:

* **Impact Analysis**: Identifies exactly which words triggered the AI (Dynamic Occlusion Test).
* **Entity Recognition (NER)**: Extracts Organizations ("Sberbank"), Persons ("Mom"), and Money amounts.
* **Trigger Highlighting**: Visualizes dangerous patterns directly in the text.

### 3. Context Awareness 📱

The engine understands the source of the message. A request to "Update Telegram" is safe if contexts is `["Telegram App"]`, but dangerous if context is `["WhatsApp"]`.

### 4. Smart Link Hunter 🕵️‍♂️

* Detects homoglyphs (fake domains looking like real ones).
* Checks protocol security (HTTP vs HTTPS).
* Validates against a verified whitelist of official domains.

---

## 🔌 API Reference

The core runs as a **FastAPI** service.

### `POST /predict`

Analyze a message for scam probability.

**Request:**

```json
{
  "text": "Win a prize at http://fake-casino.com!",
  "strict_mode": false,
  "context": ["SMS"]
}
```

**Response:**

```json
{
  "is_scam": true,
  "score": 0.99,
  "verdict": "DANGEROUS",
  "reason": ["⛔ Spam Context Detected", "⚠️ Suspicious Link"],
  "explanation": [
    {"word": "casino", "type": "TRIGGER", "impact": 1.0}
  ]
}
```

---

## 🚀 Installation & Setup

### Prerequisites

* Python 3.10+
* NVIDIA GPU (Optional, recommended for training)

### 1. Clone & Install

```bash
git clone https://github.com/f4rceful/GuardianAI.git
cd GuardianAI

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the API Server

Start the backend service on port 8000:

```bash
python src/api/server.py
```

*The server will automatically load the ONNX model and initialize the pipeline.*

### 3. Run the Desktop UI (Optional)

For testing purposes, you can run the Flet-based dashboard:

```bash
python src/ui/main.py
```

---

## 📂 Project Structure

* `src/api`: FastAPI server implementation.
* `src/core`: Core logic (Classifier, NER, Explainability, Pattern Matching).
* `models/`: Trained models (ONNX, Joblib).
* `dataset/`: Training data (Safe/Scam samples).
* `tests/`: Unit and Stress tests.

---

*Developed for "Big Challenges" (Kvantorium).*
