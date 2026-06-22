"""
Captcha OCR — Flask API server (production-ready)

POST /CamCap
Body:  { "base64ImageUrl": "data:image/gif;base64,..." }
Response: { "solution": "290" }
"""

import base64
import io
import logging
import os
import time

import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_H = 64
IMG_W = 128

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

_DIR        = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_DIR, "saved_models")
_ONNX_INT8  = os.path.join(_MODELS_DIR, "cnn_classifier_int8.onnx")
_ONNX_FP32  = os.path.join(_MODELS_DIR, "cnn_classifier.onnx")
_PT_MODEL   = os.path.join(_MODELS_DIR, "cnn_classifier.pth")

_ort_session = None
_pt_model    = None


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_ort():
    global _ort_session
    if _ort_session is not None:
        return _ort_session
    try:
        import onnxruntime as ort
        path = _ONNX_INT8 if os.path.exists(_ONNX_INT8) else _ONNX_FP32
        if not os.path.exists(path):
            return None
        import multiprocessing
        n = multiprocessing.cpu_count()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads     = n
        opts.inter_op_num_threads     = n
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
        _ort_session = ort.InferenceSession(
            path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        log.info("ONNX Runtime loaded: %s  threads=%d", os.path.basename(path), n)
        return _ort_session
    except Exception as e:
        log.warning("ONNX Runtime not available: %s", e)
        return None


def _load_pt():
    global _pt_model
    if _pt_model is not None:
        return _pt_model
    if not os.path.exists(_PT_MODEL):
        return None
    try:
        import torch
        import torch.nn as nn

        class _ResBlock(nn.Module):
            def __init__(self, channels, dropout=0.1):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
                self.relu = nn.ReLU(inplace=True)
            def forward(self, x): return self.relu(x + self.block(x))

        class _Head(nn.Module):
            def __init__(self, in_features, num_classes=10):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(in_features, 256), nn.ReLU(inplace=True),
                    nn.Dropout(0.3), nn.Linear(256, num_classes),
                )
            def forward(self, x): return self.fc(x)

        class _CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(1,  32, 3, padding=1, bias=False), nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2,2),
                    nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(True), _ResBlock(64),       nn.MaxPool2d(2,2),
                    nn.Conv2d(64,128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True), _ResBlock(128),      nn.MaxPool2d(2,2),
                    nn.Conv2d(128,256,3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True), _ResBlock(256, 0.2), nn.MaxPool2d(2,2),
                    nn.Conv2d(256,256,3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2,2),
                )
                self.gap   = nn.AdaptiveAvgPool2d(1)
                self.attn  = nn.Sequential(
                    nn.Conv2d(256, 64, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 3, 1),   nn.Sigmoid(),
                )
                self.head1 = _Head(256)
                self.head2 = _Head(256)
                self.head3 = _Head(256)
            def forward(self, x):
                g = self.gap(self.backbone(x)).squeeze(-1).squeeze(-1)
                return self.head1(g), self.head2(g), self.head3(g)

        torch.set_num_threads(os.cpu_count() or 4)
        m = _CNN()
        ckpt = torch.load(_PT_MODEL, map_location="cpu", weights_only=True)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        _pt_model = m
        log.info("PyTorch model loaded (ONNX not found — run export_onnx.py for 5x speedup)")
        return m
    except Exception as e:
        log.error("PyTorch load failed: %s", e)
        return None


def _warmup():
    """
    Run several dummy inferences right after loading.
    PyTorch/ONNX allocate kernel caches on the first real call — doing it
    now means the first browser request hits a warm engine instead of paying
    that cost (~1-2 s) at request time.
    """
    dummy = np.zeros((1, 1, IMG_H, IMG_W), dtype=np.float32)
    sess = _load_ort()
    if sess:
        for _ in range(5):
            sess.run(None, {"image": dummy})
        log.info("ONNX warmup done (5 passes)")
        return
    import torch
    model = _load_pt()
    if model:
        t = torch.zeros(1, 1, IMG_H, IMG_W)
        with torch.no_grad():
            for _ in range(5):
                model(t)
        log.info("PyTorch warmup done (5 passes)")


def preload():
    _load_ort() or _load_pt()
    _warmup()


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _to_input(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    r, g, b = img.split()
    r = np.asarray(r, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    bch = np.asarray(b, dtype=np.float32)
    # Enhance colored text vs white/dotted background:
    # colored pixels have low blue relative to red+green,
    # white pixels have equal high values in all channels.
    # This gives colored text high saliency regardless of hue.
    gray = np.clip(0.299*r + 0.587*g + 0.114*bch, 0, 255).astype(np.uint8)
    gray = Image.fromarray(gray).resize((IMG_W, IMG_H), Image.BILINEAR)
    arr  = np.asarray(gray, dtype=np.uint8)
    # Apply CLAHE — same as training preprocessing
    arr  = _clahe.apply(arr)
    buf  = np.empty((1, 1, IMG_H, IMG_W), dtype=np.float32)
    np.subtract(arr, 127.5, out=buf[0, 0], casting="unsafe")
    buf[0, 0] /= 127.5
    return buf


# ── Inference ─────────────────────────────────────────────────────────────────

def _infer(image_bytes: bytes) -> str:
    buf = _to_input(image_bytes)
    sess = _load_ort()
    if sess:
        d1, d2, d3 = sess.run(None, {"image": buf})
        return str(d1[0].argmax()) + str(d2[0].argmax()) + str(d3[0].argmax())
    import torch
    model = _load_pt()
    if model is None:
        raise RuntimeError(f"No model found in {_MODELS_DIR}.")
    with torch.no_grad():
        o1, o2, o3 = model(torch.from_numpy(buf))
    return str(o1.argmax(1).item()) + str(o2.argmax(1).item()) + str(o3.argmax(1).item())


def _infer_batch(images_bytes: list) -> list:
    """
    Run all images in a SINGLE forward pass — one HTTP round-trip, one matrix multiply.
    Returns list of solution strings in the same order as input.
    """
    n = len(images_bytes)
    batch = np.empty((n, 1, IMG_H, IMG_W), dtype=np.float32)
    for i, image_bytes in enumerate(images_bytes):
        buf = _to_input(image_bytes)
        batch[i] = buf[0]

    sess = _load_ort()
    if sess:
        d1, d2, d3 = sess.run(None, {"image": batch})
        return [
            str(d1[i].argmax()) + str(d2[i].argmax()) + str(d3[i].argmax())
            for i in range(n)
        ]

    import torch
    model = _load_pt()
    if model is None:
        raise RuntimeError(f"No model found in {_MODELS_DIR}.")
    with torch.no_grad():
        o1, o2, o3 = model(torch.from_numpy(batch))
    return [
        str(o1[i].argmax().item()) + str(o2[i].argmax().item()) + str(o3[i].argmax().item())
        for i in range(n)
    ]


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/CamCap", methods=["POST", "OPTIONS"])
def cam_cap():
    if request.method == "OPTIONS":
        return "", 204
    t0 = time.perf_counter()

    # 1. Parse JSON
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    b64 = data.get("base64ImageUrl", "").strip()
    if not b64:
        return jsonify({"error": "Missing base64ImageUrl field"}), 400

    # 2. Decode base64
    try:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        b64 += "=" * (-len(b64) % 4)
        image_bytes = base64.b64decode(b64)
    except Exception as e:
        log.warning("base64 decode failed: %s", e)
        return jsonify({"error": f"Invalid base64: {e}"}), 400

    # 3. Infer
    try:
        solution = _infer(image_bytes)
    except Exception as e:
        log.exception("Inference error")
        return jsonify({"error": str(e)}), 500

    ms = (time.perf_counter() - t0) * 1000
    log.info("solution=%s  total=%.1f ms", solution, ms)
    return jsonify({"solution": solution})


@app.route("/CamCapBatch", methods=["POST", "OPTIONS"])
def cam_cap_batch():
    if request.method == "OPTIONS":
        return "", 204
    t0 = time.perf_counter()

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    images_b64 = data.get("images", [])
    if not images_b64 or not isinstance(images_b64, list):
        return jsonify({"error": "Missing or empty 'images' array"}), 400

    # Decode all base64 strings
    images_bytes = []
    for b64 in images_b64:
        try:
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            b64 += "=" * (-len(b64) % 4)
            images_bytes.append(base64.b64decode(b64))
        except Exception as e:
            return jsonify({"error": f"Invalid base64: {e}"}), 400

    # Single batched forward pass
    try:
        solutions = _infer_batch(images_bytes)
    except Exception as e:
        log.exception("Batch inference error")
        return jsonify({"error": str(e)}), 500

    ms = (time.perf_counter() - t0) * 1000
    log.info("batch=%d  solutions=%s  total=%.1f ms", len(solutions), solutions, ms)
    return jsonify({"solutions": solutions})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ── Entry point ───────────────────────────────────────────────────────────────

# Pre-load at import time so Gunicorn workers are warm before first request
preload()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("Server ready → http://0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
