"""
Batch preprocess all math datasets from guanning-ai/reasoning-benchmarks
into verl-compatible parquet format under ./data/<name>/
"""

import os
import datasets

INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."

# (hf_id, local_name, splits_to_save)
# splits_to_save: list of (hf_split, save_as) tuples
DATASETS = [
    # === Evaluation-only (test only) ===
    ("guanning/amc23",              "amc23",            [("test", "test")]),
    ("guanning/aime24",             "aime24",           [("test", "test")]),
    ("guanning/aime25",             "aime25",           [("test", "test")]),
    ("guanning/olympiadbench",      "olympiadbench",    [("test", "test")]),
    ("guanning-ai/beyondaime",      "beyondaime",       [("test", "test")]),
    ("guanning-ai/minervamath",     "minervamath",      [("test", "test")]),
    ("guanning-ai/gsm8k-platinum",  "gsm8k-platinum",   [("test", "test")]),
    # === Train + Test ===
    ("guanning/math",               "math",             [("train", "train"), ("test", "test")]),
    ("guanning/gsm8k",              "gsm8k",            [("train", "train"), ("test", "test")]),
    ("guanning-ai/gsm8k-metamath",  "gsm8k-metamath",   [("train", "train"), ("test", "test")]),
    # === Train only ===
    ("guanning-ai/bigmath",         "bigmath",          [("train", "train")]),
    ("guanning-ai/gsm8k-mugglemath","gsm8k-mugglemath", [("train", "train")]),
    ("guanning-ai/gsm8k-mumath",    "gsm8k-mumath",     [("train", "train")]),
]


def make_map_fn(data_source, split_name):
    def process_fn(example, idx):
        question_raw = example["problem"]
        answer_raw = example["answer"]
        question = question_raw + " " + INSTRUCTION

        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(answer_raw)},
            "extra_info": {
                "split": split_name,
                "index": idx,
                "answer": str(answer_raw),
                "question": question_raw,
            },
        }
        return data
    return process_fn


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    base_dir = os.path.abspath(base_dir)

    for hf_id, local_name, splits in DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing: {hf_id} -> data/{local_name}/")
        out_dir = os.path.join(base_dir, local_name)
        os.makedirs(out_dir, exist_ok=True)

        try:
            ds = datasets.load_dataset(hf_id)
        except Exception as e:
            print(f"  ERROR loading {hf_id}: {e}")
            continue

        for hf_split, save_as in splits:
            if hf_split not in ds:
                print(f"  WARNING: split '{hf_split}' not found, skipping")
                continue

            split_ds = ds[hf_split]
            processed = split_ds.map(
                function=make_map_fn(hf_id, save_as),
                with_indices=True,
                remove_columns=split_ds.column_names,
            )

            out_path = os.path.join(out_dir, f"{save_as}.parquet")
            processed.to_parquet(out_path)
            print(f"  {save_as}: {len(processed)} rows -> {out_path}")

    print(f"\nDone! All parquet files saved under {base_dir}/")


if __name__ == "__main__":
    main()
