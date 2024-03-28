import os
import json
import sentencepiece as spm

def train_sentencepiece_model(input_file, model_prefix, vocab_size=866):
    """Train a SentencePiece model."""
    spm.SentencePieceTrainer.Train(
        f'--input={input_file} --model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} --character_coverage=1.0 --model_type=bpe'
    )

def load_sp_model(model_path):
    """Load the trained SentencePiece model."""
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(model_path)
    return sp_model

def sp_tokenize(sp_model, text):
    """Tokenizes input text using the loaded SentencePiece model."""
    return sp_model.EncodeAsPieces(text)

def load_external_vocab(file_path):
    """Loads an external vocabulary file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        vocab = json.load(file)
    return vocab

def preprocess_data(data_dir='data'):
    # Define file paths
    all_sentences_file = os.path.join(data_dir, 'all_sentences.txt')
    # ge_en_vocab_file = os.path.join(data_dir, 'ge_en_vocab.json')
    sp_model_prefix = os.path.join(data_dir, 'ge_en')
    sp_model_file = sp_model_prefix + '.model'

    # Train SentencePiece model if it doesn't exist
    if not os.path.exists(sp_model_file):
        print("Training SentencePiece model...")
        train_sentencepiece_model(all_sentences_file, sp_model_prefix)
    
    # Load the trained SentencePiece model
    sp_model = load_sp_model(sp_model_file)
    
    # Load the external Georgian-English vocabulary
    # ge_en_vocab = load_external_vocab(ge_en_vocab_file)

    # Example usage: Tokenize the first few sentences from all_sentences.txt
    with open(all_sentences_file, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i >= 5:  # Limit to first 5 sentences for demonstration
                break
            tokens = sp_tokenize(sp_model, line.strip())
            print(f"Original: {line.strip()}\nTokenized: {tokens}\n")

if __name__ == '__main__':
    preprocess_data()
