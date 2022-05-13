import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)   # cpu

with open("intents.json", 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)


input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "Sam"


def get_response(msg):
    sentence = tokenize(msg)
    BoWs = bag_of_words(sentence, all_words)
    BoWs = BoWs.reshape(1, BoWs.shape[0])
    BoWs = torch.from_numpy(BoWs).to(device)     # converts it to torch tensor
    output = model(BoWs)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]     # gets the the tag that the inputted "sentence" belongs to

    #print(tag)
    #print(predicted)
    #print(torch.max(output, dim=1))
    #print(BoWs.shape[0])
    #print(BoWs)

    probs = torch.softmax(output, dim=1)   # squashes output to be between zero and one
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                #print(f"{bot_name}: {random.choice(intent['responses'])}", end='\n\n')
                return random.choice(intent['responses'])

    #else:
        #print(f"{bot_name}: I don't understand.....", end='\n\n')
    return "I don't understand....."



# print("Let's chat! type 'quit' to exit\n\n\n")
# while True:
#     msg = input("YOU: ")
#    if msg == 'quit':
#        break
#    get_response(msg)
