import json
import os
from preprocess_caption import Vocabulary
import torch

# Load COCO-style JSON file
annotations_path = "data/coco/annotations/subset_captions_train2017.json"
with open(annotations_path, 'r') as f:
    data = json.load(f)

captions = [ann["caption"] for ann in data["annotations"]]

# Build vocab
vocab = Vocabulary(freq_threshold=1)
vocab.build_vocabulary(captions)

# Save vocab to file
torch.save(vocab, "vocab.pkl")
print("✅ Saved vocab.pkl with", len(vocab), "tokens.")
