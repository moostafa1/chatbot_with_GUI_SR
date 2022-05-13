import json
import nltk_utils as ul  #tokenize, stem, bag_of_words
import numpy as np

import torch
#import torch.nn as nn
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
from torch import optim

with open('intents.json', 'r') as f:
    intents = json.load(f)

#zprint(intents)

all_words = []
tags = []
xy = []    # pattern + label
for intent in intents['intents']:
    #print(intent, end = '\n\n')
    #print(intent['tag'], end = '\n\n')
    tag = intent['tag']     #our target
    tags.append(tag)
    #print(intent['patterns'], end = '\n\n')
    for pattern in intent['patterns']:
        token = ul.tokenize(pattern)
        all_words.extend(token)
        xy.append((token, tag))   # pattern + corresponding tag
    #for response in intent['responses']:
        #print(response)

#print(tags)
#print(all_words)
#print(xy)

ignore_words = ['?', '!', '.', ',']
all_words_ignoring = [w for w in all_words if w not in ignore_words]
all_words_stem = ul.stem(all_words_ignoring)
#print(all_words_stem)
#print(len(all_words_stem), end='\n\n')


all_words = sorted(set(all_words_stem))
tags = sorted(set(tags))
#print(tags, end='\n\n')
#print(all_words)
#print(len(all_words), end='\n\n')



X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    #print(pattern_sentence, tag)
    bag = ul.bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    #print(tag, label)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
#print("X_train:\n", X_train)
#print("y_train:\n", y_train)



class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # access dataset with index
    def __getitem__(self, index):
        return self.x_data[index] ,self.y_data[index]

    def __len__(self):
        return self.n_samples


# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(all_words)   #  = len(X_train[0])
learning_rate = 0.01
num_epochs = 300
# print(input_size, len(all_words))
# print(output_size, len(tags))


dataset = ChatDataset()
#print("Dataset: ", dataset)
#print("__getitem__:", dataset.__getitem__(5))
#print("__len__:", dataset.__len__())

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)  #num_workers=2 -->for multi processing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#criterion = nn.NLLLoss()
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)
#print(model.parameters())


for epoch in range(num_epochs):
    for (words, labels) in train_loader:       # loops for word and its label
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # forward
        outputs = model(words)     # running the batch through the NN to see the predictions
        loss = criterion(outputs, labels)   # computing loss

        # backward and optimizer step
        optimizer.zero_grad()      # clears old gradient from the last step (otherwise you'd accumulatethe gradients from all loss.backword() calls)
        loss.backward()       # compute derivatives of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation
        optimizer.step()    # update weights of the NN

    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')



data = {
        "model_state": model.state_dict(),   # state_dict() --> python dictionary object maps each layer to its parameter tensor
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size":hidden_size,
        "all_words": all_words,
        "tags": tags
        }


FILE = "data.pth"
torch.save(data, FILE)

print(f'\ntraining complete. file saved to file {FILE}')
#print("model_state:\n", model.state_dict())
