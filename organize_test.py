
import os
import shutil

folder = "dogs_vs_cats_subset/test"

os.makedirs(os.path.join(folder, "cats"), exist_ok=True)
os.makedirs(os.path.join(folder, "dogs"), exist_ok=True)

for file in os.listdir(folder):
    source = os.path.join(folder, file)

    if not os.path.isfile(source):
        continue

    if not file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue

    name = file.lower()

    if name.startswith("cat."):
        shutil.move(source, os.path.join(folder, "cats", file))
    elif name.startswith("dog."):
        shutil.move(source, os.path.join(folder, "dogs", file))

print("Test folder organized!")
print("Cat images:", len(os.listdir(os.path.join(folder, "cats"))))
print("Dog images:", len(os.listdir(os.path.join(folder, "dogs"))))