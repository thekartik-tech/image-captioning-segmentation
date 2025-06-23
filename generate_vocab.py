import pickle
from vocab_class import DummyVocab  # ✅ Import the class before pickling

vocab = DummyVocab()

with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("✅ vocab.pkl created successfully!")
