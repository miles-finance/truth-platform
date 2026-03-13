"""
Truth Platform — Flask Web App
Endpoints:
  POST /api/analyze        — Phase 1: point/counterpoint
  POST /api/score          — Phase 2: claim scoring (takes phase1 result)
  POST /api/full           — Both phases in sequence (slow)
"""

from flask import Flask, request, jsonify, render_template, Response
import json
import traceback
import io
import pypdf
from engine import run_phase1, run_phase2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max upload


@app.route("/")
def index():
    return render_template("index.html")


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
    """Phase 2: score claims. Accepts either a cache_key or the full phase1 result."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    # Reconstitute phase1 result
    if "cache_key" in data:
        phase1 = _load_phase1(data["cache_key"])
        if not phase1:
            return jsonify({"error": "Cache key not found — re-run analyze first"}), 404
    elif "phase1" in data:
        phase1 = data["phase1"]
    else:
        return jsonify({"error": "Provide 'cache_key' or 'phase1' result"}), 400

    try:
        result = run_phase2(phase1)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return jsonify({"error": "Rate limited by search or API. Try again in a moment."}), 429
        return jsonify({"error": f"Scoring error: {error_msg}"}), 500


@app.route("/api/full", methods=["POST"])
def full_analysis():
    """Both phases — slower but complete. Streams progress."""
    data = request.get_json()
    if not data or "input" not in data:
        return jsonify({"error": "Missing 'input' field"}), 400

    def generate():
        try:
            yield f"data: {json.dumps({'status': 'phase1_start', 'message': 'Analyzing article and finding opposing view...'})}\n\n"
            phase1 = run_phase1(data["input"])
            clean1 = {k: v for k, v in phase1.items() if not k.startswith("_")}
            clean1["cache_key"] = _store_phase1(phase1)
            yield f"data: {json.dumps({'status': 'phase1_done', 'result': clean1})}\n\n"

            yield f"data: {json.dumps({'status': 'phase2_start', 'message': 'Scoring claims with evidence...'})}\n\n"
            phase2 = run_phase2(phase1)
            yield f"data: {json.dumps({'status': 'phase2_done', 'result': phase2})}\n\n"

            yield f"data: {json.dumps({'status': 'complete'})}\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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

        # Cap at 50k chars (same as text input limit)
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


# ─── Simple in-memory cache for phase1 results ───────────────────────────────
import hashlib, time

_PHASE1_CACHE = {}

def _store_phase1(result: dict) -> str:
    key = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
    _PHASE1_CACHE[key] = result
    # Keep cache small
    if len(_PHASE1_CACHE) > 50:
        oldest = list(_PHASE1_CACHE.keys())[0]
        del _PHASE1_CACHE[oldest]
    return key

def _load_phase1(key: str) -> dict | None:
    return _PHASE1_CACHE.get(key)


if __name__ == "__main__":
    app.run(debug=True, port=5555, host="0.0.0.0")
