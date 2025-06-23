class DummyVocab:
    def __init__(self):
        self.stoi = {"<SOS>": 0, "<EOS>": 1}
        self.itos = {0: "<SOS>", 1: "<EOS>"}
    
    def numericalize(self, text):
        return [self.stoi.get(word, 0) for word in text.split()]
