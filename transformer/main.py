import torch
from classifier import Classifier
from torchtext import data, datasets
import torch.optim as optim
import torch.nn as nn
import time
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader, random_split

# Define the tokenizer and the vocabulary
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# Helper function to yield tokens
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# Load the datasets
train_iter, test_iter = IMDB()

# Build the vocabulary
MAX_VOCAB_SIZE = 25000
unk_token = '<unk>'
vocab = build_vocab_from_iterator(yield_tokens(train_iter), max_tokens=MAX_VOCAB_SIZE, specials=[unk_token])
vocab.set_default_index(vocab[unk_token])

# Function to process text
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: 1.0 if x == 'pos' else 0.0

# Process the data
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    return torch.tensor(label_list, dtype=torch.float32), torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True), torch.tensor(lengths)

# Set the device
BATCH_SIZE = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create DataLoader
train_loader = DataLoader(list(IMDB(split='train')), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(list(IMDB(split='test')), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# Optionally, split the training data into training and validation sets
train_size = int(0.8 * len(train_loader.dataset))
valid_size = len(train_loader.dataset) - train_size
train_dataset, valid_dataset = random_split(train_loader.dataset, [train_size, valid_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)


print("---Data processing complete.---")
print('Initializing our model now...')

# Initializing the model

EMBED_DIM = 200
NUM_HEADS = 8
FORWARD_EXPANSION = 3
MAX_LENGTH = 512
VOCAB_SIZE = len(vocab)

classifier = Classifier(
    VOCAB_SIZE, MAX_LENGTH, EMBED_DIM, NUM_HEADS, FORWARD_EXPANSION
)
print(classifier)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier.to(device)

# Training

optimizer = optim.SGD(classifier.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
criterion.to(device)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elasped_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elasped_secs

def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()

        labels, texts, lengths = batch
        if texts.shape[1] > MAX_LENGTH:
            texts = texts[:, :MAX_LENGTH]
        
        predictions = model(texts).squeeze(1)

        loss = criterion(predictions, labels)

        acc = binary_accuracy(predictions, labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            labels, texts, lengths = batch
            if texts.shape[1] > MAX_LENGTH:
              texts = texts[:, :MAX_LENGTH]
                    
            predictions = model(texts).squeeze(1)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

N_EPOCHS = 10
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(
        classifier, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(classifier, valid_loader, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(classifier.state_dict(), 'sent-classifier.pt')

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
     
