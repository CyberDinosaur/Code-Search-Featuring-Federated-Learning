import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

global_model = Model()

dataset = torch.randn(100, 10)
labels = torch.randn(100, 1)

def local_update(model, dataset, labels, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(dataset)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return model.state_dict()

num_rounds = 10
num_clients = 5

for round in range(num_rounds):
    local_updates = []  

    for client in range(num_clients):
        local_model = Model()
        local_model.load_state_dict(global_model.state_dict())

        subset_dataset = dataset[client * 20: (client + 1) * 20]
        subset_labels = labels[client * 20: (client + 1) * 20]

        local_update_params = local_update(local_model, subset_dataset, subset_labels)
        local_updates.append(local_update_params)

    aggregated_params = {}
    for key in local_updates[0].keys():
        aggregated_params[key] = torch.stack([params[key] for params in local_updates]).mean(dim=0)

    global_model.load_state_dict(aggregated_params)

    print("Global model parameters after round {}: ".format(round + 1))
    print(global_model.state_dict())
    print("\n")