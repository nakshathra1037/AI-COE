import os
import shutil

base_path = "dogs_vs_cats_subset"

for split in ["train", "test", "val"]:
    source_path = os.path.join(base_path, split)

    if not os.path.exists(source_path):
        continue

    for category in ["cat", "dog"]:
        folder_name = "cats" if category == "cat" else "dogs"
        target_path = os.path.join(source_path, folder_name)
        os.makedirs(target_path, exist_ok=True)

        for file in os.listdir(source_path):
            if file.lower().startswith(category + ".") and file.lower().endswith(".jpg"):
                shutil.move(
                    os.path.join(source_path, file),
                    os.path.join(target_path, file)
                )

print("Dataset organized successfully!")
