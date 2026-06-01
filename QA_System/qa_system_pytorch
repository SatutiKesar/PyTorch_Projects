# -*- coding: utf-8 -*-
"""QA_System_PyTorch.ipynb"""

from google.colab import drive
drive.mount('/content/drive')

# @title
import pandas as pd

df = pd.read_csv('/content/drive/Colab Notebooks/QA_Dataset.csv')
print(df.head())

# Tokenize
def tokenize(text):
  text = text.lower()
  text = text.replace('?', '')
  text = text.replace("'", "")
  return text.split()

tokenize('What is the capital of India?')

# Building Vocab
vocab = { '<UNK> ':0}

def build_vocab(row):
  print(row['question'], row['answer'])
  tokenized_question = tokenize(row['question'])
  tokenized_answer = tokenize(row['answer'])
  merged_tokens = tokenized_question + tokenized_answer

  for token in merged_tokens:

    if token not in vocab:
      vocab[token] = len(vocab)

df.apply(build_vocab, axis=1)

vocab

len(vocab)

# Converting words to Numerical Indicies
def text_to_indices(text, vocab):

  indexed_text = []

  for token in tokenize(text):

    if token in vocab:
      indexed_text.append(vocab[token])
    else:
      indexed_text.append(vocab['<UNK> '])

  return indexed_text

text_to_indices("Satuti is here", vocab)

import torch
from torch.utils.data import Dataset, DataLoader

# Dataset class
class QADataset(Dataset):

  def __init__(self, df, vocab) :
    self.df = df
    self.vocab = vocab

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, index) :
    numerical_question = text_to_indices(self.df.iloc[index]['question'], self.vocab)
    numerical_answer = text_to_indices(self.df.iloc[index]['answer'], self.vocab)

    return torch.tensor(numerical_question), torch.tensor(numerical_answer)

dataset = QADataset(df, vocab)

dataset[0]                                      # First question

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for question, answer in dataloader:
  print(question, answer[0])

import torch.nn as nn

# Using NN Module class
class SimpleRNN(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim=50)
    self.rnn = nn.RNN(50, 64, batch_first=True)
    self.fc = nn.Linear(64, vocab_size)

  def forward(self, question):
    embedded_question = self.embedding(question)
    hidden, final = self.rnn(embedded_question)
    output = self.fc(final.squeeze(0))

    return output

# Creating Parameters
learning_rate = 0.001
epochs = 20

model = SimpleRNN(len(vocab))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(epochs):

  total_loss = 0

  for question, answer in dataloader:

    optimizer.zero_grad()

    # Forward Pass
    output = model(question)

    # Loss
    loss = criterion(output, answer[0])

    # Gradients calculate
    loss.backward()

    # Update Parameters
    optimizer.step()

    # Updating Loss
    total_loss += loss.item()
  print(f"Epoch: {epoch+1}, Loss: {total_loss:4f}")

# Final Function Prediction
def predict(model, question, threshold=0.5):

  # Convert question to numbers
  numerical_question = text_to_indices(question, vocab)

  # Tensor
  question_tensor = torch.tensor(numerical_question).unsqueeze(0)

  # Send to model
  output = model(question_tensor)

  # Convert logits to Probabilities
  probs = torch.nn.functional.softmax(output, dim=1)

  # Find index of Maximum Probability
  value, index = torch.max(probs, dim=1)

  if value < threshold:
    print("I don't know")

  print(list(vocab.keys())[index])

predict(model, 'What is the capital of India')

