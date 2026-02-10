import os
import requests
import zipfile
import json
from tqdm import tqdm

# Configuration
DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "images")
URL_IMAGES = "http://images.cocodataset.org/zips/val2017.zip"
URL_ANNOTATIONS = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)


def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"{save_path} already exists. Skipping download.")
        return

    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(save_path, "wb") as f, tqdm(
        desc=save_path,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)


def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def process_coco_annotations(anno_path, output_path):
    print("Processing annotations to match Encoder format...")

    with open(anno_path, "r") as f:
        coco = json.load(f)

    # Create a map of image_id -> file_name
    img_map = {img["id"]: img["file_name"] for img in coco["images"]}

    # Format: [{"file_name": "...", "caption": "..."}]
    formatted_data = []

    for anno in coco["annotations"]:
        img_id = anno["image_id"]
        caption = anno["caption"]

        if img_id in img_map:
            formatted_data.append(
                {
                    "file_name": os.path.join(
                        "val2017", img_map[img_id]
                    ),  # Ensure subdirectory matches extract
                    "caption": caption,
                }
            )

    with open(output_path, "w") as f:
        json.dump(formatted_data, f, indent=2)

    print(f"Saved {len(formatted_data)} pairs to {output_path}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Download Images
    img_zip = os.path.join(DATA_DIR, "val2017.zip")
    download_file(URL_IMAGES, img_zip)

    # 2. Download Annotations
    anno_zip = os.path.join(DATA_DIR, "annotations.zip")
    download_file(URL_ANNOTATIONS, anno_zip)

    # 3. Extract
    extract_zip(img_zip, IMG_DIR)  # Extracts to data/images/val2017
    extract_zip(anno_zip, DATA_DIR)  # Extracts to data/annotations

    # 4. Process JSON
    raw_anno = os.path.join(DATA_DIR, "annotations", "captions_val2017.json")
    final_json = os.path.join(DATA_DIR, "captions.json")

    process_coco_annotations(raw_anno, final_json)

    print("\n--- Setup Complete ---")
    print(f"Images located in: {os.path.join(IMG_DIR, 'val2017')}")
    print(f"JSON located at:   {final_json}")
    print("\nYou can now run training with:")
    print(f"python train_encoder.py --json_path {final_json} --img_dir {IMG_DIR}")


if __name__ == "__main__":
    main()
