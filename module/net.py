import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as fn


SERIALIZED_MODEL_NAME = 'ba_model.pt'

INPUT_SIZE = 9
HIDDEN_SIZE = 6
OUTPUT_SIZE = 5


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.hidden_size = HIDDEN_SIZE

        # The LSTM takes coordinates as input, and outputs hidden states
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, input_seq):
        hidden_space, _ = self.lstm(input_seq)
        tag_space = self.hidden2tag(hidden_space.view(len(input_seq), -1))
        tag_scores = fn.log_softmax(tag_space, dim=1)
        return tag_scores


def load_model():
    if os.path.isfile(SERIALIZED_MODEL_NAME):
        model = LSTM()
        model.load_state_dict(torch.load(SERIALIZED_MODEL_NAME))
        model.eval()
        return model
    else:
        print('Model was not trained yet!')
        sys.exit(0)


def evaluate(coordinate_inputs):
    with torch.no_grad():
        model = load_model()
        input_seq = torch.tensor(coordinate_inputs)
        tag_scores = model(input_seq.unsqueeze(1))
        probabilities, tags = tag_scores.topk(1)
        predictions = []
        for i in range(len(probabilities)):
            predictions.append((torch.exp(probabilities[i]).item(), tags[i].item()))
        return predictions
