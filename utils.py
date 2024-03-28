import sentencepiece as spm
import torch

def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as file:
        for line in file:
            token, index = line.split('\t')
            vocab[token] = int(index.strip())
    return vocab

def tokenize(text, sp_model):
    return sp_model.EncodeAsIds(text)