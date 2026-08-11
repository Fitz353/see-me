import os
import shutil

from insightface.app import FaceAnalysis as fa
from PIL import Image

from engine import (
    build_app,
    cosine_sim,
    create_folder,
    crop_face,
    refresh_folder,
    scan_photo,
)


def scan_reference_faces(app: fa, start_folder, dest_folder):
    first_embeddings = []
    # Loop for faces to be searched in the first phase
    face_idx = 1
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
    return first_embeddings


def build_targets(mode, tokens, first_embeddings, output_folder):
    targets = []
    for token in tokens:
        idx = int(token)
        embedding = first_embeddings[idx - 1]  # Faces are 1-indexed
        # The output folder for each face (token)
        folder = f"{output_folder}/returned_face{idx}"
        if mode == "separate":
            create_folder(folder)
            targets.append((idx, embedding, folder))
        # Targets now hold a tuple, 3 things on each list element
        # Face index in found faces to search for, its embedding, and the name of the folder the returned photos need to go
        elif mode == "toghether":
            targets.append((idx, embedding, folder))
    return targets


def search_multi_solo(app: fa, photos_folder, targets, threshold):
    for filename in os.listdir(photos_folder):
        path = os.path.join(photos_folder, filename)
        image, faces = scan_photo(app, path)
        for face in faces:
            for idx, embedding, folder in targets:
                sim = cosine_sim(embedding, face.normed_embedding)
                if sim > threshold:
                    shutil.copy(path, folder)
                    # Breaks at the first appereance of that person in the photo, no need to search for more
                    break

def search_toghether(app: fa, photos_folder, targets, threshold, output_folder):
    for filename in os.listdir(photos_folder):
        path = os.path.join(photos_folder, filename)
        image, faces = scan_photo(app, path)

        found_all = True
        for idx, embedding, folder in targets:
            found_this_one = False
            for face in faces:
                sim = cosine_sim(embedding, face.normed_embedding)
                if sim > threshold:
                    found_this_one = True
                    break
            if not found_this_one:
                found_all= False
                break
        if found_all:
            #just one folder for the output
            shutil.copy(path, output_folder)

def check_input(tokens, first_embeddings):
    if len(tokens) == 0:
        print("Error: no faces selected")
        return False

    seen = set()  # Basically hash map, looks up faster
    for token in tokens:
        if not token.isdigit():
            print(f"Error: '{token}' is not a valid number")
            return False
        # If it passed this, this means it's a digit
        idx = int(token)
        if idx < 1:
            print(f"Error: face {idx} doesn't exist")
            return False
        if idx > len(first_embeddings):
            print(f"Error: face{idx} doesn't exist")
            return False
        if idx in seen:
            print(f"Error: face {idx} already typed")
            return False
        seen.add(idx)
    return True

def check_mode(mode):
    input = False
    while not input:
        if not(mode == "separate" or mode == "toghether"):
            print("Please choose an available mode")
            input = False
            break
        else:
            input = True
    return input
