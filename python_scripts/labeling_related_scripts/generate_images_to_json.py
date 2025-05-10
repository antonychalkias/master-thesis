import os
import json
import urllib.parse  # Needed to handle spaces in paths

# Base folder where you serve the images from
served_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
base_url = "http://localhost:8000"

folders = [
    os.path.join(served_root, "dataset_foods"),
    os.path.join(served_root, "ordered_dataset")
]

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
image_tasks = []
total_files = 0

for folder in folders:
    print(f"Scanning folder: {folder}")
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(valid_extensions):
                abs_path = os.path.join(root, file)

                # Convert to relative URL path and URL-encode it
                relative_path = os.path.relpath(abs_path, served_root)
                relative_path = relative_path.replace("\\", "/")
                url_path = f"{base_url}/{urllib.parse.quote(relative_path)}"

                image_tasks.append({"image": url_path})
                total_files += 1
                print(f"Added: {url_path}")

# Output
output_path = os.path.join(os.path.dirname(__file__), "for_labeling.json")
with open(output_path, "w") as f:
    json.dump(image_tasks, f, indent=2)

print(f"\n🎉 Created '{output_path}' with {total_files} image entries.")
