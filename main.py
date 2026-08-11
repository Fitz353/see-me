import os
import shutil
import sys

from engine import build_app, refresh_folder
from search import build_targets, check_input, scan_reference_faces, search_multi_solo, search_toghether, check_mode

# May need adjusting, copied my brother and i in the same folder, good for now
THRESHOLD = 0.34

# Actual start of code

app = build_app()

start_folder = "searched_faces"
dest_folder = "found_faces"
refresh_folder(dest_folder)

first_embeddings = scan_reference_faces(app, start_folder, dest_folder)

if len(first_embeddings) == 0:
    print("Oops! Looks like we couldn't find anybody in this photo!")
    sys.exit(1)  # So that it actually stops the script

print(f"You can now find the faces in {dest_folder}")
print("Please choose the mode you want: separate or toghether")

while True:
    mode = input()
    if check_mode(mode):
        break    

print("Please enter who do you want to search(type the number of the face)")
print('Example: "1 2 3" means search for faces indexed 1, 2 and 3')


# Prompt loop and error-checking
while True:
    number = input()
    tokens = number.split()
    if check_input(tokens, first_embeddings):
        break

# Prepare the folder which is gonna have the returned photos in which the target face appears
photos_folder = "photos_2"
output_folder = "returned_photos"
refresh_folder(output_folder)

targets = build_targets(mode, tokens, first_embeddings, output_folder)

if mode == "separate":
    search_multi_solo(app, photos_folder, targets, THRESHOLD)

    # Printing the output paths for separate
    for idx, embedding, folder in targets:
        count = len(os.listdir(folder))
        if count == 0:
            print(f"No photo was found containing face {idx}")
            shutil.rmtree(folder)
        else:
            print(f"Person {idx}: {count} photos in {folder}")
elif mode == "toghether": 
    search_toghether(app, photos_folder, targets, THRESHOLD, output_folder)
