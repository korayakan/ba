from . import data, net

import torch
import torch.nn as nn
import torch.optim as opt


SERIALIZED_MODEL_NAME = 'saved_model/ba_model.pt'

N_EPOCH = 1
PRINT_EVERY = 5000
PLOT_EVERY = 1000
LEARNING_RATE = 0.1
# LEARNING_RATE = 0.005


def train():
    model = net.LSTM()
    loss_function = nn.NLLLoss()
    optimizer = opt.SGD(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(N_EPOCH):
        for coordinate_inputs, coordinate_tags in data.prepare_training_data():
            # print(coordinate_inputs)
            # print(coordinate_tags)

            # Step 1. Clear Pytorch gradients
            model.zero_grad()

            # Step 2. Get inputs ready for the network
            input_seq = torch.tensor(coordinate_inputs)
            targets = torch.tensor(coordinate_tags)
            # print(input_seq)
            # print(targets)

            # Step 3. Run forward pass
            tag_scores = model(input_seq.unsqueeze(1))

            # Step 4. Compute the loss, gradients, and update the55 parameters
            loss = loss_function(tag_scores, targets)
            # print(loss)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), SERIALIZED_MODEL_NAME)
