import onnxruntime as ort

ort.preload_dlls()

from insightface.app import FaceAnalysis as fa
import numpy as np
from PIL import Image
import os
import shutil
import sys

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

#Actual start of code

app = build_app()

start_folder = "searched_faces"
dest_folder = "found_faces"
refresh_folder(dest_folder)

first_crops = []
first_embeddings = []

# Loop for faces to be searched in the first phase
face_idx = 0
for filename in os.listdir(start_folder):
    path = os.path.join(start_folder, filename)
    image, faces = scan_photo(app, path)
    for face in faces:
        crop = crop_face(image, face)
        if crop.size == 0:
            continue
        crop_path = f"{dest_folder}/face{face_idx}.jpg"
        face_idx = face_idx + 1
        Image.fromarray(crop).save(crop_path)
        # Save normed embeddings data for good comparison
        first_embeddings.append(face.normed_embedding)

if len(first_embeddings) == 0:
    print("Oops! Looks like we couldn't find anybody in this photo!")
    sys.exit(1) #So that it actually stops the script

print(f"You can now find the faces in {dest_folder}")
print("Please enter who do you want to search(type the number of the face)")
print('Example: "1 2 3" means search for faces indexed 1, 2 and 3')
number = input()

targets = []
tokens = number.split()

# Prepare the folder which is gonna have the returned photos in which the target face appears
photos_folder = "photos_2"
output_folder = "returned_photos"
refresh_folder(output_folder)

for token in tokens:
    idx  = int(token)
    embedding = first_embeddings[idx]
    # The output folder for each face (token)
    folder = f"{output_folder}/returned_face{idx}"
    create_folder(folder)
    # Targets now hold a tuple, 3 things on each list element
    # Face index in found faces to search for, its embedding, and the name of the folder the returned photos need to go
    targets.append((idx, embedding, folder))

# May need adjusting, copied my brother and i in the same folder, good for now
THRESHOLD = 0.34

for filename in os.listdir(photos_folder):
        path = os.path.join(photos_folder, filename)
        image, faces = scan_photo(app, path)
        for face in faces:
            for idx, embedding, folder in targets:
                sim = cosine_sim(embedding, face.normed_embedding)
                if sim > THRESHOLD:
                    shutil.copy(path, folder)
                    # Breaks at the first appereance of that person in the photo, no need to search for more
                    break

# Printing the output paths
for idx, embedding, folder in targets:
    count = len(os.listdir(folder))
    if count == 0:
        print(f"No photo was found containing face {idx}")
        shutil.rmtree(folder)
    else:
        print(f"Person {idx}: {count} photos in {output_folder}/{folder}")













