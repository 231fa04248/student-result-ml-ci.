
import logging
import sys
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g, Response
from werkzeug.exceptions import HTTPException
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# ---------------- LOGGING CONFIGURATION ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")

# ---------------- PROMETHEUS METRICS ----------------

HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the Student Result Prediction API",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total Student Result Prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the Student Result Prediction model is loaded"
)

# ---------------- FLASK APPLICATION ----------------

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

SAFE_INPUT_FIELDS = set()

# Global model variable
model = None

# ---------------- REQUEST / RESPONSE LOGGING ----------------

@app.before_request
def before_request():
    g.request_id = request.headers.get(
        "X-Request-ID", str(uuid.uuid4())
    )

    g.start_time = time.perf_counter()

    logger.info(
        "event=request_started request_id=%s method=%s path=%s",
        g.request_id,
        request.method,
        request.path
    )


@app.after_request
def after_request(response):
    duration = (
        time.perf_counter()
        - getattr(g, "start_time", time.perf_counter())
    )

    duration_ms = duration * 1000
    request_id = getattr(g, "request_id", "unknown")

    response.headers["X-Request-ID"] = request_id

    # Exclude Prometheus scraping requests
    if request.path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code)
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path
        ).observe(duration)

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
    global model

    try:
        if not MODEL_PATH.exists():
            MODEL_LOADED.set(0)

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

        MODEL_LOADED.set(1)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH
        )

        return model

    except Exception:
        model = None
        MODEL_LOADED.set(0)

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
    global model

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        PREDICTIONS.labels(result="error").inc()

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
        PREDICTIONS.labels(result="error").inc()

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

        # Load model if it has not been loaded yet
        if model is None:
            load_model()

        sample = pd.DataFrame([payload])

        prediction_code = int(model.predict(sample)[0])

        PREDICTIONS.labels(result="success").inc()

        prediction = (
            "PASS" if prediction_code == 1 else "FAIL"
        )

        logger.info(
            "event=prediction_success prediction=%s",
            prediction
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        }), 200

    except (ValueError, TypeError) as error:
        PREDICTIONS.labels(result="error").inc()

        logger.warning(
            "event=prediction_input_error error_type=%s",
            type(error).__name__
        )

        return jsonify({
            "error": str(error)
        }), 400

    except Exception:
        PREDICTIONS.labels(result="error").inc()

        logger.exception("event=prediction_failed")

        return jsonify({
            "error": "Prediction failed"
        }), 500


# ---------------- PROMETHEUS METRICS ENDPOINT ----------------

@app.get("/metrics")
def metrics():
    return Response(
        response=generate_latest(),
        status=200,
        mimetype=CONTENT_TYPE_LATEST
    )


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

    try:
        load_model()
    except Exception:
        logger.exception("event=application_startup_failed")
        raise

    app.run(
        host="0.0.0.0",
        port=5000
    )
