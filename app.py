"""
Bisaya Translator Backend
=========================
Flask API that translates between English and Cebuano (Bisaya) using
CTranslate2 machine-translation models and SentencePiece tokenizers.

Endpoints
---------
POST /translate   – translate text (see docstring on the route)
GET  /health      – liveness check

Quick start
-----------
    # 1. activate your virtual environment
    .\\venv\\Scripts\\Activate.ps1          # Windows PowerShell
    source venv/bin/activate               # macOS / Linux

    # 2. install dependencies
    pip install -r requirements.txt

    # 3. run the server
    python app.py                          # development (debug mode)
    waitress-serve --port=5000 app:app     # production (Windows)

Model Conversion (if version mismatch occurs)
---------------------------------------------
If you see "Unsupported model binary version" error, reconvert your model:

    pip install ctranslate2==4.5.0 transformers sentencepiece

    # Convert English → Cebuano model:
    ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-ceb \\
        --output_dir english-cebuano-model --quantization int8

    # Convert Cebuano → English model:
    ct2-transformers-converter --model Helsinki-NLP/opus-mt-ceb-en \\
        --output_dir cebuano-english-model --quantization int8

IMPORTANT: The CTranslate2 version used to convert the model MUST match
the version installed in production. Pin the exact version in requirements.txt.
"""

import os
import logging
import time

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import sentencepiece as spm
import ctranslate2

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
MAX_INPUT_LENGTH = 5_000          # max characters per request
DEBUG_MODE = os.getenv("FLASK_DEBUG", "1") == "1"   # default ON for dev

# Resolve model directories relative to *this* file so the paths work no
# matter which directory the server is launched from.
# IMPORTANT: Directory names must NOT contain spaces for Railway deployment.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EN_CEB_MODEL_DIR = os.path.join(_BASE_DIR, "english-cebuano-model")
CEB_EN_MODEL_DIR = os.path.join(_BASE_DIR, "cebuano-english-model")

# Expected CTranslate2 version - must match the version used to convert models
EXPECTED_CT2_VERSION = "4.5.0"

# ──────────────────────────────────────────────────────────────────────────────
# App & logging
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow cross-origin requests from your Flutter web / mobile app

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Version check - log CTranslate2 version on startup
# ──────────────────────────────────────────────────────────────────────────────
logger.info("CTranslate2 version: %s (expected: %s)", ctranslate2.__version__, EXPECTED_CT2_VERSION)
if ctranslate2.__version__ != EXPECTED_CT2_VERSION:
    logger.warning(
        "CTranslate2 version mismatch! Installed: %s, Expected: %s. "
        "This may cause 'binary version' errors if the model was converted with a different version.",
        ctranslate2.__version__, EXPECTED_CT2_VERSION
    )

# ──────────────────────────────────────────────────────────────────────────────
# Load models & tokenizers (once, at import time)
# ──────────────────────────────────────────────────────────────────────────────
def _load_sp(path: str) -> spm.SentencePieceProcessor:
    """Load a SentencePiece model file and return the processor."""
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp


def _load_translator_with_version_check(model_dir: str, model_name: str) -> ctranslate2.Translator:
    """
    Load a CTranslate2 translator with detailed error handling for version mismatches.
    
    If model loading fails due to version mismatch, provides clear instructions
    on how to reconvert the model using ct2-transformers-converter.
    """
    if not os.path.isdir(model_dir):
        raise RuntimeError(
            f"Model directory not found: {model_dir}\n"
            f"Make sure the directory exists and contains model.bin, config.json, and .spm.model files."
        )
    
    model_bin_path = os.path.join(model_dir, "model.bin")
    if not os.path.isfile(model_bin_path):
        raise RuntimeError(
            f"model.bin not found in {model_dir}\n"
            f"The model may not have been converted yet. Convert using:\n"
            f"  ct2-transformers-converter --model <huggingface_model> --output_dir {model_dir}"
        )
    
    try:
        return ctranslate2.Translator(model_dir)
    except RuntimeError as e:
        error_msg = str(e)
        if "binary version" in error_msg.lower() or "unsupported model" in error_msg.lower():
            raise RuntimeError(
                f"CTranslate2 model version mismatch for '{model_name}'!\n\n"
                f"Error: {error_msg}\n\n"
                f"This means the model was converted with a different CTranslate2 version.\n"
                f"Current CTranslate2 version: {ctranslate2.__version__}\n\n"
                f"To fix this, reconvert your model with the SAME version:\n"
                f"  pip install ctranslate2=={ctranslate2.__version__} transformers sentencepiece\n"
                f"  ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-ceb --output_dir english-cebuano-model\n"
                f"  ct2-transformers-converter --model Helsinki-NLP/opus-mt-ceb-en --output_dir cebuano-english-model\n\n"
                f"Or install the CTranslate2 version that matches your model."
            ) from e
        raise


try:
    logger.info("Loading English → Cebuano model from: %s", EN_CEB_MODEL_DIR)
    en_ceb_translator = _load_translator_with_version_check(EN_CEB_MODEL_DIR, "English-Cebuano")
    en_sp   = _load_sp(os.path.join(EN_CEB_MODEL_DIR, "en.spm.model"))   # source tokenizer
    ceb_sp  = _load_sp(os.path.join(EN_CEB_MODEL_DIR, "ceb.spm.model"))  # target tokenizer

    logger.info("Loading Cebuano → English model from: %s", CEB_EN_MODEL_DIR)
    ceb_en_translator = _load_translator_with_version_check(CEB_EN_MODEL_DIR, "Cebuano-English")
    ceb_sp2 = _load_sp(os.path.join(CEB_EN_MODEL_DIR, "ceb.spm.model"))  # source tokenizer
    en_sp2  = _load_sp(os.path.join(CEB_EN_MODEL_DIR, "en.spm.model"))   # target tokenizer

    logger.info("All models loaded successfully ✓")
except RuntimeError as e:
    logger.error(str(e))
    raise
except Exception:
    logger.exception(
        "Failed to load models. Make sure the model directories exist and "
        "each contains: model.bin, config.json, en.spm.model, ceb.spm.model"
    )
    raise

# A lookup dict keeps the route logic clean.
_MODELS = {
    "en-ceb": {
        "translator": en_ceb_translator,
        "source_sp": en_sp,
        "target_sp": ceb_sp,
    },
    "ceb-en": {
        "translator": ceb_en_translator,
        "source_sp": ceb_sp2,
        "target_sp": en_sp2,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Request hooks (optional: request timing)
# ──────────────────────────────────────────────────────────────────────────────
@app.before_request
def _start_timer():
    g.start_time = time.perf_counter()


@app.after_request
def _log_request(response):
    duration_ms = (time.perf_counter() - g.start_time) * 1000
    logger.info(
        "%s %s → %s  (%.1f ms)",
        request.method, request.path, response.status_code, duration_ms,
    )
    return response

# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/translate", methods=["POST"])
def translate():
    """
    Translate text between English and Cebuano.

    **Request** (JSON, Content-Type: application/json):
        {
            "text": "Hello, how are you?",
            "direction": "en-ceb"          // optional, default "en-ceb"
        }

    **Response** (JSON):
        { "translated": "Kumusta, kumusta ka?" }

    **direction** must be one of:
        • "en-ceb"  – English → Cebuano  (default)
        • "ceb-en"  – Cebuano → English

    **Errors** return { "error": "..." } with an appropriate HTTP status code.
    """

    # 1. Parse JSON body ---------------------------------------------------
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "error": "Request body must be valid JSON with header "
                     "'Content-Type: application/json'."
        }), 400

    # 2. Validate "text" field ---------------------------------------------
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "'text' is required and must be a non-empty string."
        }), 400

    text = text.strip()
    if len(text) > MAX_INPUT_LENGTH:
        return jsonify({
            "error": f"Text too long ({len(text)} chars). "
                     f"Maximum is {MAX_INPUT_LENGTH}."
        }), 400

    # 3. Validate "direction" field ----------------------------------------
    direction = data.get("direction", "en-ceb")
    model_cfg = _MODELS.get(direction)
    if model_cfg is None:
        return jsonify({
            "error": f"Invalid direction '{direction}'. "
                     f"Use one of: {', '.join(_MODELS)}."
        }), 400

    # 4. Tokenize → Translate → Detokenize --------------------------------
    try:
        source_sp  = model_cfg["source_sp"]
        target_sp  = model_cfg["target_sp"]
        translator = model_cfg["translator"]

        tokens         = source_sp.encode(text, out_type=str)
        result         = translator.translate_batch([tokens])
        output_tokens  = result[0].hypotheses[0]
        translated_text = target_sp.decode(output_tokens)
    except Exception as exc:
        logger.exception("Translation error for direction=%s", direction)
        return jsonify({"error": f"Translation failed: {exc}"}), 500

    return jsonify({"translated": translated_text})


@app.route("/health", methods=["GET"])
def health():
    """Liveness probe — returns 200 if the server is up."""
    return jsonify({"status": "ok", "models_loaded": list(_MODELS.keys())})


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    #  • For DEVELOPMENT only.  In production use a real WSGI server:
    #      waitress-serve --port=5000 app:app        (Windows)
    #      gunicorn -w 4 -b 0.0.0.0:5000 app:app    (Linux / macOS)
    app.run(host="0.0.0.0", port=5000, debug=DEBUG_MODE)