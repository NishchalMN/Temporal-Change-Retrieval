import sys
import os
import json
from pathlib import Path
import torch.multiprocessing as mp

sys.path.insert(0, "/home/yog/scratch/CHD/submission")
sys.path.insert(
    0, "/scratch/zt1/project/mqzhu-prj/user/yog/CHD/data/external/changechat/ChangeChat"
)


CONFIG = {
    "model_path": "/home/yog/scratch/CHD/experiments/changechat-lora-7b/checkpoint-1374",
    "model_base": "/home/yog/.cache/huggingface/hub/models--liuhaotian--llava-v1.6-vicuna-7b/snapshots/deae57a8c0ccb0da4c2661cc1891cc9d06503d11",
    "vision_tower": "/scratch/zt1/project/mqzhu-prj/user/yog/CHD/data/external/changechat/ChangeChat/hf-models/clip-vit-large-patch14-336",
    "output_dir": "/home/yog/scratch/CHD/generated_captions",
    "num_captions": 3,
    "num_gpus": 4,
}


DATASETS = {
    "whu_cd": {
        "path": "/home/yog/scratch/CHD/data/external/whu_cd/",
        "splits": ["all"],
    },
    "xbd": {"path": "/home/yog/scratch/CHD/data/external/xbd/", "splits": ["all"]},
}


def load_processed(output_file):
    if not output_file.exists():
        return set()

    try:
        with open(output_file, "r") as f:
            data = json.load(f)
        processed = {item["original_paths"]["image_a"] for item in data}
        print(f"Found {len(processed)} already-processed pairs", flush=True)
        return processed
    except:
        return set()


def process_gpu_shard(gpu_id, datasets_info, config, shard_info):
    from inference import ChangeChatInference
    from dataset_loader import WHUCDAdapter, XBDAdapter

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"\n[GPU {gpu_id}] Starting...", flush=True)

    engine = ChangeChatInference(
        model_path=config["model_path"],
        model_base=config["model_base"],
        vision_tower=config["vision_tower"],
        num_gpus=1,
    )

    total_gpus, gpu_rank = shard_info

    adapter_map = {"whu_cd": WHUCDAdapter, "xbd": XBDAdapter}

    # Iterate over datasets
    for dataset_name, dataset_config in datasets_info.items():
        adapter_class = adapter_map[dataset_name]
        adapter = adapter_class(dataset_config["path"])

        for split in dataset_config["splits"]:
            print(f"\n[GPU {gpu_id}] Processing {dataset_name}/{split}", flush=True)

            output_path = (
                Path(config["output_dir"])
                / f"{dataset_name}_{split}_captions_gpu{gpu_id}.json"
            )

            processed_ids = load_processed(output_path)

            if output_path.exists():
                with open(output_path, "r") as f:
                    results = json.load(f)
                print(f"[GPU {gpu_id}] Resuming from {len(results)} pairs", flush=True)
            else:
                results = []

            pairs = adapter.get_image_pairs(split)

            if not pairs:
                continue

            # Split the captioning across GPUs
            shard_pairs = [p for i, p in enumerate(pairs) if i % total_gpus == gpu_rank]
            remaining = [p for p in shard_pairs if p.image_a_path not in processed_ids]

            if not remaining:
                print(f"[GPU {gpu_id}] All pairs processed!", flush=True)
                continue

            print(f"[GPU {gpu_id}] Processing {len(remaining)} pairs", flush=True)

            for idx, pair in enumerate(remaining):
                if (idx + 1) % 50 == 0 or idx == 0:
                    print(
                        f"[GPU {gpu_id}] {idx+1}/{len(remaining)} completed", flush=True
                    )

                try:
                    img_a, img_b = adapter.load_image_pair(pair)
                    captions = engine.generate_multiple_captions(
                        img_a, img_b, num_captions=config["num_captions"]
                    )

                    results.append(
                        {
                            "image_id": pair.image_id,
                            "captions": captions,
                            "dataset": pair.dataset,
                            "split": pair.split,
                            "has_change": (
                                bool(pair.has_change)
                                if pair.has_change is not None
                                else None
                            ),
                            "original_paths": {
                                "image_a": pair.image_a_path,
                                "image_b": pair.image_b_path,
                                "label": pair.label_path,
                            },
                        }
                    )
                    if (idx + 1) % 50 == 0:
                        with open(output_path, "w") as f:
                            json.dump(results, f, indent=2)

                except Exception as e:
                    print(f"[GPU {gpu_id}] Error: {e}", flush=True)
                    continue

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            print(f"[GPU {gpu_id}] Saved {len(results)} samples", flush=True)

    print(f"\n[GPU {gpu_id}] Done!", flush=True)


def merge_gpu_outputs(datasets_info, output_dir):
    print("\nMerging GPU outputs...")

    for dataset_name, dataset_config in datasets_info.items():
        for split in dataset_config["splits"]:
            gpu_files = list(
                Path(output_dir).glob(f"{dataset_name}_{split}_captions_gpu*.json")
            )

            if not gpu_files:
                continue

            merged = []
            for gpu_file in sorted(gpu_files):
                with open(gpu_file, "r") as f:
                    data = json.load(f)
                    merged.extend(data)

            output_file = Path(output_dir) / f"{dataset_name}_{split}_captions.json"
            with open(output_file, "w") as f:
                json.dump(merged, f, indent=2)

            print(f"Merged {dataset_name}/{split}: {len(merged)} samples")


def main():
    mp.set_start_method("spawn", force=True)
    processes = []

    for gpu_id in range(CONFIG["num_gpus"]):
        p = mp.Process(
            target=process_gpu_shard,
            args=(gpu_id, DATASETS, CONFIG, (CONFIG["num_gpus"], gpu_id)),
        )
        p.start()
        processes.append(p)
        print(f"Started worker on GPU {gpu_id}")

    for p in processes:
        p.join()

    merge_gpu_outputs(DATASETS, CONFIG["output_dir"])


if __name__ == "__main__":
    main()
