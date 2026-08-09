import onnxruntime as ort

ort.preload_dlls()

from insightface.app import FaceAnalysis as fa
import numpy as np
from PIL import Image
import os
import shutil


# build the app so that all the models are loaded at the start
# GPU first, then falls back on cpu
app = fa(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


start_folder = "my_face"
choosing_dest = "guide_faces"
shutil.rmtree(choosing_dest, ignore_errors=True)
os.makedirs(choosing_dest, exist_ok=True)

# filename is name of the photo
filename = os.listdir(start_folder)[0]  # its just one file in the folder
start_path = os.path.join(start_folder, filename)

pil = Image.open(start_path).convert("RGB")
start_image = np.array(pil)

# app.get expects numpy array
# faces is a list of objects of every face in the photo, with a lot of stuff to it
start_faces = app.get(start_image)

all_crops = []
all_embeddings = []

h, w = start_image.shape[:2]

if len(start_faces) == 0:
    print("Oops! Looks like we couldn't find anybody in this photo!")

for i, face in enumerate(start_faces):
    x1, y1, x2, y2 = np.round(face.bbox).astype(int)
    # not letting the coords go off bounds of the photo, because we ll get an error/warning
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w)
    y2 = min(y2, h)

    crop = start_image[y1:y2, x1:x2]

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
with open("sim_scores.txt", "w") as f:
    for filename in os.listdir(photos_folder):
        comp_path = os.path.join(photos_folder, filename)
        comp_image = np.array(Image.open(comp_path).convert("RGB"))
        h, w = comp_image.shape[:2]
        comp_faces = app.get(comp_image)
        for i, face in enumerate(comp_faces):
            sim = np.dot(target, face.normed_embedding)
            # This is for debugging, seeing the scores for everyone

            x1, y1, x2, y2 = np.round(face.bbox).astype(int)
            # not letting the coords go off bounds of the photo, because we ll get an error/warning

            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, w)
            y2 = min(y2, h)

            crop_debug = comp_image[y1:y2, x1:x2]
            crop_dbg_path = f"{debug_folder}/face{cnt}.jpg"
            if crop_debug.size == 0:
                continue
            f.write(f"Similarity score for face{cnt} is {sim}\n")
            cnt = cnt + 1
            Image.fromarray(crop_debug).save(crop_dbg_path)
            # Debug stops here

            if sim > THRESHOLD:
                shutil.copy(comp_path, target_folder)
                break
