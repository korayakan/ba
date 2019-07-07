from . import data, net

import torch
import torch.nn as nn
import torch.optim as opt


PRINT_EVERY = 5000
PLOT_EVERY = 1000


def train(epochs, print_every=1, learning_rate=0.1):
    model = net.LSTM()
    loss_function = nn.NLLLoss()
    optimizer = opt.SGD(model.parameters(), lr=learning_rate)
    data_set = data.prepare_training_data()
    split_data = {0: [], 1: [], 2: [], 3: []}
    for i in range(len(data_set)):
        if i % 4 == 0:
            split_data[0].append(data_set[i])
        if i % 4 == 1:
            split_data[1].append(data_set[i])
        if i % 4 == 2:
            split_data[2].append(data_set[i])
        if i % 4 == 3:
            split_data[3].append(data_set[i])
    # for i in range(4):
    #     print(len(split_data[i]))

    steps = 0
    running_loss = 0
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        steps += 1
        validation_idx = epoch % 4
        validation_data = split_data[validation_idx]
        training_data = []
        for i in range(4):
            if i != validation_idx:
                training_data.extend(split_data[i])

        for coordinate_inputs, coordinate_tags in training_data:
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

            # Step 4. Compute the loss, gradients, and update the parameters
            loss = loss_function(tag_scores, targets)
            # print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for coordinate_inputs_val, coordinate_tags_val in validation_data:
                    input_seq_val = torch.tensor(coordinate_inputs_val)
                    targets_val = torch.tensor(coordinate_tags_val)
                    tag_scores_val = model.forward(input_seq_val.unsqueeze(1))
                    loss_val = loss_function(tag_scores_val, targets_val)
                    test_loss += loss_val.item()

            train_losses.append(running_loss / (len(training_data) * print_every))
            test_losses.append(test_loss / len(validation_data))
            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {running_loss / (len(training_data) * print_every):.3f}.. "
                  f"Test loss: {test_loss / len(validation_data):.3f}.. ")
                  #f"Test accuracy: {accuracy / len(testloader):.3f}")
            running_loss = 0
            model.train()

    torch.save(model.state_dict(), net.SERIALIZED_MODEL_NAME)
    return train_losses, test_losses
