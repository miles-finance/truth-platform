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
from engine import run_phase1, run_phase2

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Phase 1: extract article, find opposing view, generate point/counterpoint."""
    data = request.get_json()
    if not data or "input" not in data:
        return jsonify({"error": "Missing 'input' field (URL or article text)"}), 400

    try:
        result = run_phase1(data["input"])
        # Don't expose internal text blobs in the API response
        clean = {k: v for k, v in result.items() if not k.startswith("_")}
        # But stash the raw texts under a separate key for the /api/score call
        clean["cache_key"] = _store_phase1(result)
        return jsonify(clean)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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
        return jsonify({"error": str(e)}), 500


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
    import os
    port = int(os.environ.get("PORT", 5555))
    app.run(debug=False, port=port, host="0.0.0.0")
