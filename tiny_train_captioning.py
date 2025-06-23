import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from cnn_encoder import CNNEncoder
from lstm_decoder import LSTMDecoder
from data_loader import CocoDataset

# ✅ Paths
image_dir = "data/coco/train2017_subset"
annotation_file = "data/coco/annotations/subset_captions_train2017.json"
vocab_path = "vocab.pkl"
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

# ✅ Load Vocabulary
vocab = torch.load(vocab_path, weights_only=False)
print(f"✅ Loaded vocab with {len(vocab)} tokens")

# ✅ Dataset & Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = CocoDataset(image_dir, annotation_file, vocab, transform)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

# ✅ Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = CNNEncoder(embed_size=128).to(device)
decoder = LSTMDecoder(embed_size=128, hidden_size=256, vocab_size=len(vocab), num_layers=1).to(device)

# ✅ Loss & Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
params = list(decoder.parameters()) + list(encoder.fc.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)

# ✅ Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, captions) in enumerate(data_loader):
        images, captions = images.to(device), captions.to(device)

        features = encoder(images)
        outputs = decoder(features, captions[:, :-1])
        targets = captions[:, 1:]

        # Match output and target lengths
        if outputs.size(1) > targets.size(1):
            outputs = outputs[:, :targets.size(1), :]
        elif outputs.size(1) < targets.size(1):
            targets = targets[:, :outputs.size(1)]

        loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"📚 Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ✅ Save Model
torch.save(encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
torch.save(decoder.state_dict(), os.path.join(save_dir, "decoder.pth"))
print("✅ Models saved to:", save_dir)
