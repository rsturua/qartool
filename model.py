import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        emb_con = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(emb_con, hidden)
        prediction = self.fc_out(torch.cat((output.squeeze(0), embedded.squeeze(0), context.squeeze(0)), dim=1))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden = self.encoder(src)
        
        input = trg[0,:]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = trg[t] if teacher_force else top1
        return outputs


class TranslationDataset(Dataset):
    def __init__(self, ge_file, en_file, sp_model):
        # Load tokenized data from files
        with open(ge_file, 'r', encoding='utf-8') as f:
            self.ge_sentences = [line.strip() for line in f.readlines()]
        with open(en_file, 'r', encoding='utf-8') as f:
            self.en_sentences = [line.strip() for line in f.readlines()]

        self.sp_model = sp_model

    def __len__(self):
        return len(self.ge_sentences)

    def __getitem__(self, idx):
        ge_sentence = self.ge_sentences[idx]
        en_sentence = self.en_sentences[idx]

        # Tokenize sentences using SentencePiece model
        ge_tokens = self.sp_model.EncodeAsPieces(ge_sentence)
        en_tokens = self.sp_model.EncodeAsPieces(en_sentence)

        # Convert tokens to tensors
        ge_tensor = torch.tensor(ge_tokens, dtype=torch.long)  # Convert to tensor
        en_tensor = torch.tensor(en_tokens, dtype=torch.long)  # Convert to tensor

        return ge_tensor, en_tensor


def load_sp_model(model_path):
    """Load the trained SentencePiece model."""
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(model_path)
    return sp_model

# Add the following lines after the TranslationDataset class definition:

# Define SentencePiece model path and load the model
sp_model_path = 'data/ge_en.model'
sp_model = load_sp_model(sp_model_path)

# Instantiate the TranslationDataset
dataset = TranslationDataset('data/ge_tokenized.txt', 'data/en_tokenized.txt', sp_model)



# Define hyperparameters and other configurations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = 1000  # Define appropriately
output_dim = 1000  # Define appropriately
emb_dim = 256  # Define appropriately
hid_dim = 512  # Define appropriately
n_layers = 2  # Define appropriately
dropout = 0.5  # Define appropriately
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Instantiate your dataset and dataloader
# dataset = TranslationDataset('data/ge_tokenized.txt', 'data/en_tokenized.txt')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate encoder and decoder
encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout)
decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout)

# Instantiate your Seq2Seq model
model = Seq2Seq(encoder, decoder, device)

# Ensure that model parameters are on the correct device
model.to(device)

# Define your loss function
criterion = nn.CrossEntropyLoss()

# Check if model has parameters
if not list(model.parameters()):
    raise ValueError("Model has no parameters. Check the initialization of your Seq2Seq model.")

# Define your optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Iterate over your training dataset
    for batch in dataloader:
        # Unpack the batch into input and target sequences
        input_seq, target_seq = batch

        # Unpack input_seq
        input_seq = input_seq[0]  # Assuming the input sequence is the first element of the tuple

        # Move sequences to device
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        # Forward pass
        outputs = model(input_seq, target_seq)

        # Compute the loss
        loss = criterion(outputs, target_seq)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print or log the loss
        print(f'Epoch [{epoch+1}/{num_epochs}], Batch loss: {loss.item()}')
