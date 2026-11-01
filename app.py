
from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Step 4: Configure Application Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


# Step 5: Log Model Loading
def load_model():
    if not MODEL_PATH.exists():
        logger.error(
            "event=model_file_missing model_file=%s",
            MODEL_PATH
        )
        raise FileNotFoundError(
            "student_result_model.pkl was not found. "
            "Run the training pipeline first."
        )

    try:
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


# Step 6: Request and Response Logging
@app.before_request
def before_request():
    request.request_id = request.headers.get(
        "X-Request-ID", str(uuid.uuid4())
    )
    request.start_time = time.perf_counter()


@app.after_request
def after_request(response):
    duration_ms = (
        time.perf_counter() - request.start_time
    ) * 1000

    level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request.request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    response.headers["X-Request-ID"] = request.request_id

    return response


# Step 7: Handle Application Errors
@app.errorhandler(Exception)
def handle_exception(error):
    logger.exception(
        "event=application_error request_id=%s error_type=%s",
        getattr(request, "request_id", "unknown"),
        type(error).__name__
    )

    return jsonify({
        "error": "Internal server error"
    }), 500


# Home Endpoint
@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


# Step 9: Health Endpoint
@app.get("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "student-result-prediction"
    }), 200


# Step 8: Safe Prediction Logging
@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        logger.warning(
            "event=validation_error reason=invalid_json_or_empty_body"
        )

        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=validation_error missing_fields=%s",
            missing_fields
        )

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        logger.info(
            "event=prediction_request feature_names=%s",
            FEATURES
        )

        model = load_model()

        prediction_code = int(model.predict(sample)[0])

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

    except (ValueError, TypeError) as e:
        logger.warning(
            "event=prediction_validation_failed error_type=%s",
            type(e).__name__
        )

        return jsonify({
            "error": str(e)
        }), 400

    except Exception:
        logger.exception(
            "event=prediction_failed request_id=%s",
            request.request_id
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


# Run Flask Application
if __name__ == "__main__":
    logger.info("event=application_starting")

    app.run(
        host="0.0.0.0",
        port=5000
    )
