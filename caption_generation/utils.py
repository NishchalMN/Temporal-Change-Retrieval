import json
from pathlib import Path
from collections import Counter


def check_caption_quality(captions_file):
    with open(captions_file) as f:
        data = json.load(f)

    all_captions = []
    for item in data:
        all_captions.extend(item.get("captions", []))

    counter = Counter(all_captions)
    duplicates = {k: v for k, v in counter.items() if v > 1}

    print(f"Total captions: {len(all_captions)}")
    print(f"Unique captions: {len(counter)}")
    print(f"Duplicate captions: {len(duplicates)}")

    if all_captions:
        avg_len = sum(len(c.split()) for c in all_captions) / len(all_captions)
        min_len = min(len(c.split()) for c in all_captions)
        max_len = max(len(c.split()) for c in all_captions)
        print(f"Caption length: {avg_len:.1f} words (min={min_len}, max={max_len})")

    print("\nMost common captions:")
    for caption, count in counter.most_common(5):
        print(f"  {count}x: {caption[:60]}...")


def validate_instruction_file(instruction_file):
    with open(instruction_file) as f:
        data = json.load(f)

    errors = []

    for idx, item in enumerate(data):

        if "id" not in item:
            errors.append(f"Item {idx}: Missing 'id'")
        if "image" not in item or len(item.get("image", [])) != 2:
            errors.append(f"Item {idx}: Invalid 'image' field")
        if "conversations" not in item:
            errors.append(f"Item {idx}: Missing 'conversations'")

        for conv in item.get("conversations", []):
            if "from" not in conv or "value" not in conv:
                errors.append(f"Item {idx}: Invalid conversation")
                break

    if errors:
        print(f"  Found {len(errors)} errors:")
        for error in errors[:5]:
            print(f"    - {error}")
        return False
    else:
        print(f"  OK: {len(data)} items validated")
        return True


def validate_output_directory(output_dir, image_dir=None):
    output_dir = Path(output_dir)

    caption_files = list(output_dir.glob("*_captions.json"))
    print(f"\nFound {len(caption_files)} caption files:")
    for f in caption_files:
        with open(f) as fp:
            data = json.load(fp)
        print(f"  {f.name}: {len(data)} items")

    instruction_files = list(output_dir.glob("*_instructions.json"))
    print(f"\nFound {len(instruction_files)} instruction files:")
    for f in instruction_files:
        with open(f) as fp:
            data = json.load(fp)
        print(f"  {f.name}: {len(data)} items")

    merged_file = output_dir / "multi_dataset_instruction_training.json"
    if merged_file.exists():
        with open(merged_file) as f:
            data = json.load(f)
        print(f"\nMerged training file: {len(data)} samples")

        # Change distribution
        changed = sum(1 for x in data if x.get("changeflag", 1) == 1)
        print(f"  Changed: {changed}")
        print(f"  Unchanged: {len(data) - changed}")

    if image_dir:
        image_dir = Path(image_dir)
        if image_dir.exists():
            print(f"\nChecking images in {image_dir}...")
            total = 0
            for dataset_dir in image_dir.iterdir():
                if dataset_dir.is_dir():
                    count = len(list(dataset_dir.rglob("*.png")))
                    print(f"  {dataset_dir.name}: {count} images")
                    total += count
            print(f"  Total: {total} images")
        else:
            print(f"\nImage directory not found: {image_dir}")

    print("\nValidation complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validation utilities")
    parser.add_argument(
        "--output-dir", default="/home/yog/scratch/CHD/generated_captions"
    )
    parser.add_argument("--image-dir", default="/home/yog/scratch/CHD/prepared_images")
    parser.add_argument("--check-captions", help="Check specific caption file")

    args = parser.parse_args()

    if args.check_captions:
        check_caption_quality(args.check_captions)
    else:
        validate_output_directory(args.output_dir, args.image_dir)
