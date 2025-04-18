import os
from collections import Counter

# --- CONFIG ---
CONFIG = {
    "dataset_dir": "dataset",
    "subsets": ["train", "val", "test"],
    "class_names": ["happy", "sad", "dead"]
}

def count_classes():
    for subset in CONFIG["subsets"]:
        counter = Counter()
        label_dir = os.path.join(CONFIG["dataset_dir"], "labels", subset)
        if not os.path.exists(label_dir):
            print(f"⚠️  Skipping '{subset}': no such directory.")
            continue

        for file in os.listdir(label_dir):
            if not file.endswith(".txt"):
                continue
            path = os.path.join(label_dir, file)
            with open(path, "r") as f:
                for line in f:
                    cls_id = int(line.strip().split()[0])
                    counter[cls_id] += 1

        print(f"\n📊 Class distribution in '{subset}' subset:")
        for cls_id, name in enumerate(CONFIG["class_names"]):
            print(f"  {name:<6} → {counter.get(cls_id, 0)} samples")

if __name__ == "__main__":
    count_classes()