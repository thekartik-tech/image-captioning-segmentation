import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class CocoDataset(Dataset):
    def __init__(self, image_dir, annotation_path, vocab, transform=None):
        with open(annotation_path, 'r') as f:
            data = json.load(f)

        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform
        self.image_info = {img['id']: img['file_name'] for img in data['images']}
        self.annotations = data['annotations']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]
        caption = ann['caption']
        image_id = ann['image_id']
        image_file = self.image_info[image_id]
        img_path = os.path.join(self.image_dir, image_file)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        tokens = self.vocab.numericalize(caption)
        tokens = [self.vocab.stoi["<SOS>"]] + tokens + [self.vocab.stoi["<EOS>"]]
        caption_tensor = torch.tensor(tokens)

        return image, caption_tensor

    def collate_fn(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images, 0)

        # Pad captions
        lengths = [len(cap) for cap in captions]
        max_len = max(lengths)
        padded = torch.zeros(len(captions), max_len).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            padded[i, :end] = cap[:end]

        return images, padded
