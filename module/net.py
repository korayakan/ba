import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as fn


SERIALIZED_MODEL_NAME = 'ba_model.pt'

INPUT_SIZE = 9
OUTPUT_SIZE = 5


class LSTM(nn.Module):

    def __init__(self, hidden_size=6, num_of_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        # The LSTM takes coordinates as input, and outputs hidden states
        self.lstm = nn.LSTM(INPUT_SIZE, hidden_size, num_layers=num_of_layers)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_size, OUTPUT_SIZE)

    def forward(self, input_seq):
        hidden_space, _ = self.lstm(input_seq)
        tag_space = self.hidden2tag(hidden_space.view(len(input_seq), -1))
        tag_scores = fn.log_softmax(tag_space, dim=1)
        return tag_scores


def load_model(model_path=SERIALIZED_MODEL_NAME):
    if os.path.isfile(model_path):
        model = torch.load(model_path)
        model.eval()
        return model
    else:
        print('Model was not trained yet!')
        sys.exit(0)


def evaluate(coordinate_inputs, model_path=SERIALIZED_MODEL_NAME):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Using {} for prediction'.format(device))
    with torch.no_grad():
        model = load_model(model_path)
        model.to(device)
        input_seq = torch.tensor(coordinate_inputs)
        input_seq = input_seq.unsqueeze(1)
        input_seq = input_seq.to(device)
        tag_scores = model(input_seq)
        probabilities, tags = tag_scores.topk(1)
        predictions = []
        for i in range(len(probabilities)):
            predictions.append((torch.exp(probabilities[i]).item(), tags[i].item()))
        return predictions
