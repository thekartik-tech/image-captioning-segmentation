import os
import torch
from torchvision import transforms
from PIL import Image

from cnn_encoder import CNNEncoder
from lstm_decoder import LSTMDecoder

# ✅ For vocab deserialization
import torch.serialization
from preprocess_caption import Vocabulary
torch.serialization.add_safe_globals([Vocabulary])

class TinyCaptionGenerator:
    def __init__(self, encoder_path, decoder_path, vocab_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ Load vocab
        self.vocab = torch.load(vocab_path, weights_only=False)

        # ✅ Load models
        self.encoder = CNNEncoder(embed_size=128).to(self.device)
        self.decoder = LSTMDecoder(embed_size=128, hidden_size=256, vocab_size=len(self.vocab), num_layers=1).to(self.device)

        self.encoder.eval()
        self.decoder.eval()

        if os.path.exists(encoder_path) and os.path.getsize(encoder_path) > 1000:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            print("✅ encoder.pth loaded.")
        else:
            print("⚠️ encoder.pth is missing or invalid.")

        if os.path.exists(decoder_path) and os.path.getsize(decoder_path) > 1000:
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print("✅ decoder.pth loaded.")
        else:
            print("⚠️ decoder.pth is missing or invalid.")

        # ✅ Image preprocessor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def predict(self, image_path, max_len=20):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        feature = self.encoder(image)
        sampled_ids = [self.vocab.stoi.get("<SOS>", 0)]

        last_id = None
        repeat_count = 0
        max_repeats = 5

        for _ in range(max_len):
            inputs = torch.tensor([sampled_ids], device=self.device)
            outputs = self.decoder(feature, inputs)
            _, predicted = outputs[:, -1, :].max(1)
            predicted_id = predicted.item()

            if predicted_id == last_id:
                repeat_count += 1
            else:
                repeat_count = 0
            last_id = predicted_id

            sampled_ids.append(predicted_id)

            if predicted_id == self.vocab.stoi.get("<EOS>", 1) or repeat_count >= max_repeats:
                break

        caption_tokens = [self.vocab.itos.get(i, "<UNK>") for i in sampled_ids[1:-1]]
        print("🔍 Tokens:", caption_tokens)

        if not caption_tokens or all(t == "<SOS>" for t in caption_tokens):
            return "⚠️ Could not generate caption."
        return " ".join(caption_tokens)
