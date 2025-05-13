import json
import os

# Path to the JSON file
json_file_path = os.path.join(os.path.dirname(__file__), "for_labeling.json")

# Load the existing JSON data
with open(json_file_path, 'r') as f:
    image_tasks = json.load(f)

# Count before filtering
count_before = len(image_tasks)

# Check what URLs we have
dataset_foods_urls = 0
ordered_dataset_urls = 0
other_urls = 0
for task in image_tasks:
    url = task["image"]
    if "dataset_foods" in url:
        dataset_foods_urls += 1
    elif "ordered_dataset" in url:
        ordered_dataset_urls += 1
    else:
        other_urls += 1
        print(f"Other URL: {url}")

print(f"URLs breakdown:")
print(f"- dataset_foods: {dataset_foods_urls}")
print(f"- ordered_dataset: {ordered_dataset_urls}")
print(f"- other: {other_urls}")
print(f"Total: {len(image_tasks)}")

# Filter out images from ordered_dataset
filtered_tasks = [task for task in image_tasks if "ordered_dataset" not in task["image"]]

# Count after filtering
count_after = len(filtered_tasks)
removed_count = count_before - count_after

# Save the filtered data back to the JSON file
with open(json_file_path, 'w') as f:
    json.dump(filtered_tasks, f, indent=2)

print(f"\n✅ Done! Removed {removed_count} images from 'ordered_dataset'.")
print(f"Original count: {count_before}, New count: {count_after}")
