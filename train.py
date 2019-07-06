from data import *
from model import *
import torch.optim as optim


SERIALIZED_MODEL_NAME = 'ba_model.pt'

N_EPOCH = 100
PRINT_EVERY = 5000
PLOT_EVERY = 1000
LEARNING_RATE = 0.1
# LEARNING_RATE = 0.005


model = LSTM()
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
for epoch in range(N_EPOCH):
    for coordinate_inputs, coordinate_tags in prepare_training_data():
        # print(coordinate_inputs)
        # print(coordinate_tags)
        # Step 1. Clear Pytorch gradients
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        input_seq = torch.tensor(coordinate_inputs)
        targets = torch.tensor(coordinate_tags)

        # print(input_seq)
        # print(targets)

        # Step 3. Run our forward pass.
        tag_scores = model(input_seq.unsqueeze(1))

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        # print(loss)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), SERIALIZED_MODEL_NAME)
