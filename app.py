
import logging
import sys
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g
from werkzeug.exceptions import HTTPException

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


# Request logging middleware
@app.before_request
def before_request():
    g.request_id = str(uuid.uuid4())
    g.start_time = time.time()

    logger.info(
        "Request started | ID: %s | Method: %s | Path: %s",
        g.request_id,
        request.method,
        request.path
    )


@app.after_request
def after_request(response):
    duration = time.time() - getattr(g, "start_time", time.time())
    request_id = getattr(g, "request_id", "unknown")

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "Request completed | ID: %s | Status: %s | Duration: %.4f seconds",
        request_id,
        response.status_code,
        duration
    )

    return response


# Load trained model
def load_model():
    if not MODEL_PATH.exists():
        logger.error("Model file not found: %s", MODEL_PATH)
        raise FileNotFoundError(
            "student_result_model.pkl was not found. Run the training pipeline first."
        )

    logger.info("Loading prediction model")

    model = joblib.load(MODEL_PATH)

    logger.info("Model loaded successfully")

    return model


# Home endpoint
@app.get("/")
def home():
    logger.info("Home endpoint accessed")

    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    }), 200


# Health check endpoint
@app.get("/health")
def health_check():
    logger.info("Health check requested")

    return jsonify({
        "status": "healthy",
        "service": "student-result-prediction"
    }), 200


# Prediction endpoint
@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        logger.warning("Prediction request received without JSON body")

        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "Prediction request missing fields: %s",
            missing_fields
        )

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        sample = pd.DataFrame([
            {feature: data[feature] for feature in FEATURES}
        ])

        model = load_model()

        prediction_code = int(model.predict(sample)[0])

        prediction = "PASS" if prediction_code == 1 else "FAIL"

        logger.info(
            "Prediction successful | Result: %s",
            prediction
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        }), 200

    except (ValueError, TypeError) as error:
        logger.warning("Invalid prediction input: %s", error)

        return jsonify({
            "error": str(error)
        }), 400

    except Exception:
        logger.exception("Unexpected error during prediction")

        return jsonify({
            "error": "Internal server error"
        }), 500


# Handle HTTP errors
@app.errorhandler(HTTPException)
def handle_http_error(error):
    logger.warning(
        "HTTP error | Status: %s | Description: %s",
        error.code,
        error.description
    )

    return jsonify({
        "error": error.description
    }), error.code


# Handle unexpected application errors
@app.errorhandler(Exception)
def handle_unexpected_error(error):
    logger.exception("Unexpected application error")

    return jsonify({
        "error": "Internal server error"
    }), 500


if __name__ == "__main__":
    logger.info("Starting Student Result Prediction API")

    app.run(
        host="0.0.0.0",
        port=5000
    )
