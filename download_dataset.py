import json
import os

AUGMENTED_JSONL = "/home/rg625/datasets/pixmo_stratified.jsonl"
DOWNLOADED_JSONL = "/home/rg625/datasets/pixmo_local.jsonl"
OUTPUT_JSONL = "/home/rg625/datasets/pixmo_ready.jsonl"


def main():
    print("Loading downloaded image index...")
    image_map = {}
    with open(DOWNLOADED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # Use the original text as the unique fingerprint
            image_map[data["caption"]] = data["local_path"]

    print("Pairing with augmented text...")
    paired_data = []
    missing_count = 0

    with open(AUGMENTED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            original_text = row.get("original", "")

            # If the image successfully downloaded for this text, lock them together
            if original_text in image_map:
                row["local_filename"] = os.path.basename(image_map[original_text])
                paired_data.append(row)
            else:
                missing_count += 1

    print(f"Successfully paired {len(paired_data)} images!")
    if missing_count > 0:
        print(
            f"Dropped {missing_count} rows (images failed to download or were corrupt)."
        )

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for row in paired_data:
            f.write(json.dumps(row) + "\n")

    print("Done! Data is perfectly synced and ready to pack.")


if __name__ == "__main__":
    main()
