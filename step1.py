import onnxruntime as ort

ort.preload_dlls()

from insightface.app import FaceAnalysis as fa
import numpy as np
from PIL import Image
import os
import shutil
import sys

# Build the app first thing, takes some time and resources, then just reuse it
# GPU first, then falls back on CPU if needed
def build_app():
    app = fa(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

app = build_app()

start_folder = "my_face"
choosing_dest = "guide_faces"
shutil.rmtree(choosing_dest, ignore_errors=True)
os.makedirs(choosing_dest, exist_ok=True)

not_ready = False
while(not_ready):
    print(f"Please put in {start_folder} photos with the people you want to search")
    print("Is it done?(y/n)")
    answer = input()
    if answer != "y" or answer !="n":
        print("Invalid instructions. Please try again")
        continue
    if answer == "y":
        not_ready = True
    else:
        not_ready = False

# The name of the staring photo
filename = os.listdir(start_folder)[0]  # It's just one file in the folder
start_path = os.path.join(start_folder, filename)

def scan_photo(app, path):
    image = np.array(Image.open(path).convert("RGB"))
    faces = app.get(image)
    return image, faces

start_image, start_faces = scan_photo(app, start_path)

all_crops = []
all_embeddings = []

if len(start_faces) == 0:
    print("Oops! Looks like we couldn't find anybody in this photo!")
    sys.exit(1) #So that it actually stops the script

def crop_face(image, face):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = np.round(face.bbox).astype(int)
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w)
    y2 = min(y2, h)
    return image[y1:y2, x1:x2]

for i, face in enumerate(start_faces):
    crop = crop_face(start_image, face)
    all_embeddings.append(face.normed_embedding)
    crop_path = f"{choosing_dest}/face{i}.jpg"
    all_crops.append(crop_path)
    Image.fromarray(crop).save(crop_path)
print(f"You can now find the faces in {choosing_dest}")
print("Please choose yourself(type the number of the face)")
number = input()
# so now target photo has the data from the face that was chosen
target = all_embeddings[int(number)]

# Prepare the folder which is gonna have the returned photos in which the target face appears
photos_folder = "photos_2"
target_folder = "returned_photos"
shutil.rmtree(target_folder, ignore_errors=True)
os.makedirs(target_folder, exist_ok=True)

THRESHOLD = 0.32

debug_folder = "debug_sim"
shutil.rmtree(debug_folder, ignore_errors=True)
os.makedirs(debug_folder, exist_ok=True)

cnt = 0

def cosine_sim(a, b):
    return np.dot(a, b)

with open("sim_scores.txt", "w") as f:
    for filename in os.listdir(photos_folder):
        comp_path = os.path.join(photos_folder, filename)
        comp_image, comp_faces = scan_photo(app, comp_path)
        for i, face in enumerate(comp_faces):
            sim = cosine_sim(target, face.normed_embedding)
            # This is for debugging, seeing the scores for everyone
            # not letting the coords go off bounds of the photo, because we ll get an error/warning
            crop_debug = crop_face(comp_image, face)
            if crop_debug.size == 0:
                continue
            crop_dbg_path = f"{debug_folder}/face{cnt}.jpg"
            f.write(f"Similarity score for face{cnt} is {sim}\n")
            cnt = cnt + 1
            Image.fromarray(crop_debug).save(crop_dbg_path)
            # Debug stops here

            if sim > THRESHOLD:
                shutil.copy(comp_path, target_folder)
                break
