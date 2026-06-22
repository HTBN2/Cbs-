"""
Standalone captcha solver — no dependencies on the training project.

Drop this file + saved_models/ folder anywhere and it works.

Requirements:
    pip install onnxruntime pillow numpy opencv-python

Usage:
    from solver import solve
    print(solve("image.gif"))   # "290"
"""

import os
import cv2
import numpy as np
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
IMG_H = 64
IMG_W = 128

_DIR          = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR   = os.path.join(_DIR, "saved_models")
_ONNX_INT8    = os.path.join(_MODELS_DIR, "cnn_classifier_int8.onnx")
_ONNX_FP32    = os.path.join(_MODELS_DIR, "cnn_classifier.onnx")
_PT_MODEL     = os.path.join(_MODELS_DIR, "cnn_classifier.pth")

# ── Singletons ────────────────────────────────────────────────────────────────
_session   = None   # ONNX Runtime session
_pt_model  = None   # PyTorch fallback
_buf       = np.empty((1, 1, IMG_H, IMG_W), dtype=np.float32)  # reused every call
_clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


# ── Image preprocessing ───────────────────────────────────────────────────────

def _preprocess(img_path: str) -> np.ndarray:
    """Load image → float32 (1,1,H,W) normalised to [-1, 1]. Matches training pipeline exactly."""
    img = Image.open(img_path).convert("RGB")
    r, g, b = img.split()
    r   = np.asarray(r, dtype=np.float32)
    g   = np.asarray(g, dtype=np.float32)
    bch = np.asarray(b, dtype=np.float32)
    # Weighted grayscale then CLAHE — same as training's preprocess_image()
    gray = np.clip(0.299*r + 0.587*g + 0.114*bch, 0, 255).astype(np.uint8)
    gray = Image.fromarray(gray).resize((IMG_W, IMG_H), Image.BILINEAR)
    arr  = np.asarray(gray, dtype=np.uint8)
    arr  = _clahe.apply(arr)
    np.subtract(arr, 127.5, out=_buf[0, 0], casting="unsafe")
    _buf[0, 0] /= 127.5
    return _buf


# ── ONNX Runtime backend (primary — fastest) ──────────────────────────────────

def _load_ort():
    global _session
    if _session is not None:
        return _session
    try:
        import onnxruntime as ort
    except ImportError:
        return None

    path = _ONNX_INT8 if os.path.exists(_ONNX_INT8) else _ONNX_FP32
    if not os.path.exists(path):
        return None

    import multiprocessing
    n = multiprocessing.cpu_count()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads        = n
    opts.inter_op_num_threads        = n
    opts.graph_optimization_level    = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode              = ort.ExecutionMode.ORT_SEQUENTIAL

    _session = ort.InferenceSession(path, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
    return _session


def _run_ort(img_path: str):
    sess = _load_ort()
    if sess is None:
        return None
    buf = _preprocess(img_path)
    d1, d2, d3 = sess.run(None, {"image": buf})
    return str(d1[0].argmax()) + str(d2[0].argmax()) + str(d3[0].argmax())


# ── PyTorch fallback ──────────────────────────────────────────────────────────

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
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
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
                    nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(True), _ResBlock(64),        nn.MaxPool2d(2,2),
                    nn.Conv2d(64,128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True), _ResBlock(128),       nn.MaxPool2d(2,2),
                    nn.Conv2d(128,256,3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True), _ResBlock(256, 0.2),  nn.MaxPool2d(2,2),
                    nn.Conv2d(256,256,3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2,2),
                )
                self.gap  = nn.AdaptiveAvgPool2d(1)
                self.attn = nn.Sequential(
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
        return m
    except Exception as e:
        print(f"[solver] PyTorch load failed: {e}")
        return None


def _run_pt(img_path: str):
    import torch
    model = _load_pt()
    if model is None:
        return None
    buf = _preprocess(img_path)
    t = torch.from_numpy(buf.copy())
    with torch.no_grad():
        d1, d2, d3 = model(t)
    return str(d1.argmax(1).item()) + str(d2.argmax(1).item()) + str(d3.argmax(1).item())


# ── Public API ────────────────────────────────────────────────────────────────

def preload():
    """Call once at server startup — eliminates cold-start latency."""
    if _load_ort() is None:
        _load_pt()


def solve(image_path: str) -> str:
    """
    Read a 3-digit captcha image and return the digits as a string.

    Args:
        image_path: path to .gif / .png / .jpg

    Returns:
        str: exactly 3 digits, e.g. "290"

    Raises:
        FileNotFoundError: if the image doesn't exist
        RuntimeError:      if no model is found in saved_models/
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    result = _run_ort(image_path) or _run_pt(image_path)

    if result is None:
        raise RuntimeError(
            "No model found in saved_models/\n"
            "Copy cnn_classifier.onnx (or cnn_classifier.pth) into saved_models/"
        )

    return result
