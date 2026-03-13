"""
Truth Platform — Flask Web App
Endpoints:
  POST /api/analyze        — Phase 1: point/counterpoint
  POST /api/score          — Phase 2: claim scoring (requires cache_key from phase 1)
  POST /api/upload-pdf     — Extract text from PDF
  GET  /health             — Health check
"""

import hashlib
import io
import json
import secrets
import time
import traceback
from threading import Lock

import pypdf
from flask import Flask, Response, jsonify, render_template, request

from engine import run_phase1, run_phase2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max upload


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Phase 1: extract article, find opposing view, generate point/counterpoint."""
    data = request.get_json()
    if not data or "input" not in data:
        return jsonify({"error": "Missing 'input' field (URL or article text)"}), 400

    user_input = str(data["input"]).strip()
    is_url = user_input.startswith("http://") or user_input.startswith("https://")
    if not is_url and len(user_input) < 20:
        return jsonify({"error": "Input too short. Provide a URL or at least 20 characters of article text."}), 400
    if len(user_input) > 50000:
        return jsonify({"error": "Input too long. Max 50,000 characters."}), 400

    try:
        result = run_phase1(user_input)
        clean = {k: v for k, v in result.items() if not k.startswith("_")}
        clean["cache_key"] = _store_phase1(result)
        return jsonify(clean)
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        traceback.print_exc()
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return jsonify({"error": "Rate limited. Try again in a moment."}), 429
        return jsonify({"error": f"Analysis error: {error_msg}"}), 500


@app.route("/api/score", methods=["POST"])
def score():
    """Phase 2: score claims. Requires cache_key from Phase 1."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    if "cache_key" not in data:
        return jsonify({"error": "Missing 'cache_key'. Run /api/analyze first."}), 400

    phase1 = _load_phase1(data["cache_key"])
    if not phase1:
        return jsonify({"error": "Cache key not found or expired — re-run analyze first"}), 404

    try:
        result = run_phase2(phase1)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return jsonify({"error": "Rate limited by search or API. Try again in a moment."}), 429
        return jsonify({"error": f"Scoring error: {error_msg}"}), 500


@app.route("/api/upload-pdf", methods=["POST"])
def upload_pdf():
    """Extract text from an uploaded PDF and return it."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a multipart form with field 'file'."}), 400

    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF (.pdf)"}), 400

    try:
        reader = pypdf.PdfReader(io.BytesIO(f.read()))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        full_text = "\n\n".join(pages)
        if not full_text.strip():
            return jsonify({"error": "Could not extract text from PDF. It may be image-based or encrypted."}), 422

        full_text = full_text[:50000]
        return jsonify({
            "text": full_text,
            "page_count": len(reader.pages),
            "char_count": len(full_text)
        })
    except pypdf.errors.PdfReadError as e:
        return jsonify({"error": f"Invalid or corrupted PDF: {str(e)}"}), 422
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"PDF extraction failed: {str(e)}"}), 500


# ─── Thread-safe in-memory cache ──────────────────────────────────────────────

_PHASE1_CACHE = {}
_CACHE_LOCK = Lock()
_CACHE_MAX = 50


def _store_phase1(result: dict) -> str:
    """Store phase1 result and return a unique cache key."""
    key = secrets.token_hex(6)
    with _CACHE_LOCK:
        _PHASE1_CACHE[key] = {"data": result, "ts": time.time()}
        # Evict oldest if over limit
        if len(_PHASE1_CACHE) > _CACHE_MAX:
            oldest_key = min(_PHASE1_CACHE, key=lambda k: _PHASE1_CACHE[k]["ts"])
            del _PHASE1_CACHE[oldest_key]
    return key


def _load_phase1(key: str) -> dict | None:
    """Load phase1 result by cache key. Returns None if not found or expired (30 min)."""
    with _CACHE_LOCK:
        entry = _PHASE1_CACHE.get(key)
        if not entry:
            return None
        # Expire after 30 minutes
        if time.time() - entry["ts"] > 1800:
            del _PHASE1_CACHE[key]
            return None
        # Touch for LRU
        entry["ts"] = time.time()
        return entry["data"]


if __name__ == "__main__":
    app.run(debug=True, port=5555, host="0.0.0.0")
