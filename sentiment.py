import torch
from torchtext import datasets
from torchtext import data
import spacy
import torch.nn.functional as F
import torch.nn as nn
import random
import torch.optim as optim
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
TEXT = data.Field(tokenize = 'spacy')
LABEL = data.LabelField(tensor_type= torch.FloatTensor)
train, test = datasets.IMDB.splits(TEXT, LABEL)
train, valid = train.split(random_state = random.seed(SEED))
TEXT.build_vocab(train, max_size = 25000, vectors = "glove.6B.100d")
LABEL.build_vocab(train)
BATCH_SIZE = 64
train_itr, valid_itr, test_itr = data.BucketIterator.splits((train,test,valid), batch_size = BATCH_SIZE, sort_key = lambda x:len(x.text), repeat = False)
class RNN(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional, dropout = dropout)
    self.fc = nn.Linear(hidden_dim *2, output_dim)
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    embedded = self.dropout(self.embedding(x))
    output, (hidden, cell) = self.rnn(embedded)
    hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
    return self.fc(hidden.squeeze(0))
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
pretrained_embedding = TEXT.vocab.vectors
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)
def binary_accuracy(preds, y):
  rounds_preds = torch.round(F.sigmoid(preds))
  correct = (rounds_preds == y).float()
  acc = correct.sum()/len(correct)
  return acc
def train(model, iterator, optimizer, criterion):
  epoch_loss = 0
  epoch_acc = 0
  model.train()
  for batch in iterator:
    optimizer.zero_grad()
    predictions = model(batch.text).squeeze(1)
    loss = criterion(predictions, batch.label)
    acc = binary_accuracy(predictions, batch.label)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    epoch_acc += acc.item()
  return epoch_loss/len(iterator), epoch_acc/len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
       for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
NLP = spacy.load('en')
def predict_sentiment(sentence):
  tokenized = [tok.text for tok in NLP.tokenizer(sentence)]
  indexed = [TEXT.vocab.stoi[t] for t in tokenized]
  tensor = torch.LongTensor(indexed).to(device)
  tensor = tensor.unsqueeze(1)
  predection = F.sigmoid(model(tensor))
  return predection.item()
 

