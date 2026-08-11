import onnxruntime as ort

ort.preload_dlls()

import os
import shutil

import numpy as np
from insightface.app import FaceAnalysis as fa
from PIL import Image


def build_app():
    # Build the app first thing, takes some time and resources, then just reuse it
    # GPU first, then falls back on CPU if needed
    app = fa(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def scan_photo(app: fa, path):
    # Scans one photo and returns the faces
    image = np.array(Image.open(path).convert("RGB"))
    faces = app.get(image)
    return image, faces


def crop_face(image, face):
    # Crops just the face from the image
    h, w = image.shape[:2]
    x1, y1, x2, y2 = np.round(face.bbox).astype(int)
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w)
    y2 = min(y2, h)
    return image[y1:y2, x1:x2]


def cosine_sim(a, b):
    return np.dot(a, b)


def refresh_folder(folder):
    # Wipe and create (if needed) folder with results
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)


def create_folder(folder):
    os.makedirs(folder)
