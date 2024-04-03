import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
import random
from torch.nn.utils.rnn import pad_sequence

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


# Load the SentencePiece model
def load_sp_model(model_path):
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(model_path)
    return sp_model

sp_model_path = 'data/ge_en.model'
sp_model = load_sp_model(sp_model_path)

class TranslationDataset(Dataset):
    def __init__(self, ge_file, en_file, sp_model):
        self.sp_model = sp_model
        with open(ge_file, 'r', encoding='utf-8') as ge_f, open(en_file, 'r', encoding='utf-8') as en_f:
            self.pairs = [(ge_line.strip(), en_line.strip()) for ge_line, en_line in zip(ge_f, en_f) if ge_line.strip() and en_line.strip()]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ge_sentence, en_sentence = self.pairs[idx]
        ge_indices = self.sp_model.EncodeAsIds(ge_sentence)
        en_indices = self.sp_model.EncodeAsIds(en_sentence)
        max_index = input_dim - 1  # Assuming input_dim is your vocabulary size
        if max(ge_indices) > max_index or max(en_indices) > max_index:
            print(f"Warning: Token index out of range. Max index: {max_index}, Found indices: {max(ge_indices)}, {max(en_indices)}")
        return torch.tensor(ge_indices, dtype=torch.long), torch.tensor(en_indices, dtype=torch.long)

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    ge_tensors, en_tensors = zip(*batch)
    ge_tensors_padded = pad_sequence(ge_tensors, batch_first=True, padding_value=sp_model.pad_id())
    en_tensors_padded = pad_sequence(en_tensors, batch_first=True, padding_value=sp_model.pad_id())
    return ge_tensors_padded, en_tensors_padded


batch_size = 64  
num_epochs = 10
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


input_dim = len(sp_model)  
output_dim = input_dim  
emb_dim = 256  
hid_dim = 512  
n_layers = 2
dropout = 0.5

encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout)
decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=sp_model.pad_id())

# Dataset and DataLoader
dataset = TranslationDataset('data/ge_tokenized.txt', 'data/en_tokenized.txt', sp_model)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# Define your model architecture, optimizer, and loss function
model = Seq2Seq(encoder, decoder, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=sp_model.pad_id())  # Ignore padding

def train(model, dataloader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for ge_tensors, en_tensors in dataloader:
            if ge_tensors is None:  # Skip the batch if it's filtered out by custom_collate_fn
                continue
            ge_tensors, en_tensors = ge_tensors.to(device), en_tensors.to(device)
            optimizer.zero_grad()
            output = model(ge_tensors, en_tensors[:-1])
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            en_tensors = en_tensors[1:].reshape(-1)
            loss = criterion(output, en_tensors)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

def translate_sentence(model, sentence, sp_model, device):
    model.eval()
    tokens = sp_model.EncodeAsIds(sentence.strip())
    tokens_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tokens_tensor, tokens_tensor, 0)  # 0 turns off teacher forcing
    translated_tokens = [sp_model.IdToPiece(token_id) for token_id in outputs.argmax(2).cpu().numpy().flatten()[1:]]
    return ''.join(translated_tokens)

# Train the model
train(model, dataloader, optimizer, criterion, num_epochs)

# Translate a sentence
ge_sentence = "მაგალითი წინადადება"
translation = translate_sentence(model, ge_sentence, sp_model, device)
print(f"Original: {ge_sentence}\nTranslated: {translation}")