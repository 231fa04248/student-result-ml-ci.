import logging
import sys
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g
from werkzeug.exceptions import HTTPException

# ---------------- LOGGING CONFIGURATION ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")

# ---------------- FLASK APPLICATION ----------------

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

# Log feature names only. Do not log student identities or confidential data.
SAFE_INPUT_FIELDS = set()


# ---------------- REQUEST / RESPONSE LOGGING ----------------

@app.before_request
def before_request():
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    g.start_time = time.perf_counter()

    logger.info(
        "event=request_started request_id=%s method=%s path=%s",
        g.request_id,
        request.method,
        request.path
    )


@app.after_request
def after_request(response):
    duration_ms = (
        time.perf_counter() - getattr(g, "start_time", time.perf_counter())
    ) * 1000

    request_id = getattr(g, "request_id", "unknown")

    response.headers["X-Request-ID"] = request_id

    log_level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        log_level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    return response


# ---------------- MODEL LOADING ----------------

def load_model():
    try:
        if not MODEL_PATH.exists():
            logger.error(
                "event=model_load_failed model_file=%s",
                MODEL_PATH
            )
            raise FileNotFoundError(
                "student_result_model.pkl was not found. "
                "Run the training pipeline first."
            )

        logger.info(
            "event=model_loading model_file=%s",
            MODEL_PATH
        )

        model = joblib.load(MODEL_PATH)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH
        )

        return model

    except Exception:
        logger.exception(
            "event=model_load_failed model_file=%s",
            MODEL_PATH
        )
        raise


# ---------------- HOME ENDPOINT ----------------

@app.get("/")
def home():
    logger.info("event=home_endpoint_accessed")

    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    }), 200


# ---------------- HEALTH CHECK ----------------

@app.get("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "student-result-prediction"
    }), 200


# ---------------- PREDICTION ENDPOINT ----------------

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        logger.warning(
            "event=invalid_request reason=missing_or_invalid_json"
        )

        return jsonify({
            "error": "A valid JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=validation_error field_locations=%s error_count=%s",
            missing_fields,
            len(missing_fields)
        )

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        payload = {
            feature: data[feature]
            for feature in FEATURES
        }

        safe_input = {
            key: value
            for key, value in payload.items()
            if key in SAFE_INPUT_FIELDS
        }

        logger.info(
            "event=prediction_request feature_names=%s safe_input=%s",
            sorted(payload.keys()),
            safe_input
        )

        sample = pd.DataFrame([payload])

        model = load_model()

        prediction_code = int(model.predict(sample)[0])

        prediction = "PASS" if prediction_code == 1 else "FAIL"

        logger.info(
            "event=prediction_success prediction=%s",
            prediction
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        }), 200

    except (ValueError, TypeError) as error:
        logger.warning(
            "event=prediction_input_error error_type=%s",
            type(error).__name__
        )

        return jsonify({
            "error": str(error)
        }), 400

    except Exception:
        logger.exception("event=prediction_failed")

        return jsonify({
            "error": "Prediction failed"
        }), 500


# ---------------- HTTP ERROR HANDLING ----------------

@app.errorhandler(HTTPException)
def handle_http_error(error):
    logger.warning(
        "event=http_error status=%s description=%s",
        error.code,
        error.description
    )

    return jsonify({
        "error": error.description
    }), error.code


# ---------------- APPLICATION ERROR HANDLING ----------------

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    logger.exception("event=application_error")

    return jsonify({
        "error": "Internal server error"
    }), 500


# ---------------- APPLICATION STARTUP ----------------

if __name__ == "__main__":
    logger.info("event=application_starting")

    app.run(
        host="0.0.0.0",
        port=5000
    )
