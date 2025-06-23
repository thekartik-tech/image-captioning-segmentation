import torch

vocab = torch.load("vocab.pkl", weights_only=False)
print("📦 Vocabulary size:", len(vocab))
print("🔤 Sample tokens:", list(vocab.itos.items())[:10])
