import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import models
import utils

import numpy as np

# Define Hyper-parameters
input_size = 784
hidden_size_1 = 300
hidden_size_2 = 100
num_classes = 10
num_iter = 50000
batch_size = 60
learning_rate = 1.2e-3

threshold = 1.0e-07
num_experiments = 10

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

truncation_error = list()
for _ in range(num_experiments):
    # Initialize the model
    model = models.LeNet(input_size, hidden_size_1, hidden_size_2, num_classes).to(device)

    inputs_fc1 = list()
    def collect_input_fc1(self, input, output):
        if device != torch.device('cpu'):
            inputs_fc1.append(input[0].cpu().detach().numpy())
        else:
            inputs_fc1.append(input[0].detach().numpy())


    inputs_fc2 = list()
    def collect_input_fc2(self, input, output):
        if device != torch.device('cpu'):
            inputs_fc2.append(input[0].cpu().detach().numpy())
        else:
            inputs_fc2.append(input[0].detach().numpy())


    inputs_fc3 = list()
    def collect_input_fc3(self, input, output):
        if device != torch.device('cpu'):
            inputs_fc3.append(input[0].cpu().detach().numpy())
        else:
            inputs_fc3.append(input[0].detach().numpy())

    model.fc1.register_forward_hook(collect_input_fc1)
    model.fc2.register_forward_hook(collect_input_fc2)
    model.fc3.register_forward_hook(collect_input_fc3)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print('Training...')
    utils.train_model(model, train_loader, num_iter, device, criterion, optimizer)

    # Test the model
    accuracy = utils.test_model(model, test_loader, device)

    # Compute Si and mean
    print('Computing Si...')
    test_model = models.LeNet(input_size, hidden_size_1, hidden_size_2, num_classes).to(device)
    test_model.fc1.weight.data = model.fc1.weight.data
    test_model.fc1.bias.data = model.fc1.bias.data
    test_model.fc2.weight.data = model.fc2.weight.data
    test_model.fc2.bias.data = model.fc2.bias.data
    test_model.fc3.weight.data = model.fc3.weight.data
    test_model.fc3.bias.data = model.fc3.bias.data
    Si_fc1, mean_fc1 = utils.compute_sensitivity_indices(inputs_fc1, test_model)

    test_model = models.TestNet_fc2(hidden_size_1, hidden_size_2, num_classes).to(device)
    test_model.fc2.weight.data = model.fc2.weight.data
    test_model.fc2.bias.data = model.fc2.bias.data
    test_model.fc3.weight.data = model.fc3.weight.data
    test_model.fc3.bias.data = model.fc3.bias.data
    Si_fc2, mean_fc2 = utils.compute_sensitivity_indices(inputs_fc2, test_model)

    test_model = models.TestNet_fc3(hidden_size_2, num_classes).to(device)
    test_model.fc3.weight.data = model.fc3.weight.data
    test_model.fc3.bias.data = model.fc3.bias.data
    Si_fc3, mean_fc3 = utils.compute_sensitivity_indices(inputs_fc3, test_model)

    Si = {'fc1': Si_fc1, 'fc2': Si_fc2, 'fc3': Si_fc3}
    mean = {'fc1': mean_fc1, 'fc2': mean_fc2, 'fc3': mean_fc3}

    non_zero_Si_indices = {'fc1': np.where((Si['fc1'] < threshold) == False)[0],
                           'fc2': np.where((Si['fc2'] < threshold) == False)[0],
                           'fc3': np.where((Si['fc3'] < threshold) == False)[0]}

    zero_Si_indices = {'fc1': np.where(Si['fc1'] < threshold)[0],
                       'fc2': np.where(Si['fc2'] < threshold)[0],
                       'fc3': np.where(Si['fc3'] < threshold)[0]}

    # Compute bias adjustments
    layers = ['fc1', 'fc2', 'fc3']
    sizes = [hidden_size_1, hidden_size_2, num_classes]
    bias_adjustment = dict()
    with torch.no_grad():
        layer = layers[0]
        bias_adjustment[layer] = np.zeros(sizes[0])
        for i in zero_Si_indices[layer]:
            for parameter in model.named_parameters():
                if parameter[0] == layer + '.weight':
                    bias_adjustment[layer] += parameter[1].data[:, i].numpy() * mean[layer][i]

        for j in range(1, len(layers)):
            prev_layer = layers[j - 1]
            layer = layers[j]
            bias_adjustment[layer] = np.zeros(sizes[j])
            for i in zero_Si_indices[layer]:
                for parameter in model.named_parameters():
                    if parameter[0] == layer + '.weight':
                        bias_adjustment[layer] += parameter[1].data[:, i].numpy() * mean[layer][i]

    print('Pruning following number of neurons...')
    print('fc1: {}'.format(len(zero_Si_indices['fc1'])))
    print('fc2: {}'.format(len(zero_Si_indices['fc2'])))
    print('fc3: {}'.format(len(zero_Si_indices['fc3'])))
    num_zero_Si_indices = len(zero_Si_indices['fc1']) + len(zero_Si_indices['fc2']) + len(zero_Si_indices['fc3'])

    # Initialize a reduced model
    input_size = input_size - len(zero_Si_indices['fc1'])
    hidden_size_1 = hidden_size_1 - len(zero_Si_indices['fc2'])
    hidden_size_2 = hidden_size_2 - len(zero_Si_indices['fc3'])
    new_model = models.LeNet(input_size, hidden_size_1, hidden_size_2, num_classes)

    x = model.fc1.weight.data
    x = torch.index_select(x, 1, torch.from_numpy(non_zero_Si_indices['fc1']))
    x = torch.index_select(x, 0, torch.from_numpy(non_zero_Si_indices['fc2']))
    new_model.fc1.weight.data = x

    x = model.fc1.bias.data
    x = torch.index_select(x, 0, torch.from_numpy(non_zero_Si_indices['fc2']))
    new_model.fc1.bias.data = x

    x = model.fc2.weight.data
    x = torch.index_select(x, 1, torch.from_numpy(non_zero_Si_indices['fc2']))
    x = torch.index_select(x, 0, torch.from_numpy(non_zero_Si_indices['fc3']))
    new_model.fc2.weight.data = x

    x = model.fc2.bias.data
    x = torch.index_select(x, 0, torch.from_numpy(non_zero_Si_indices['fc3']))
    new_model.fc2.bias.data = x

    x = model.fc3.weight.data
    x = torch.index_select(x, 1, torch.from_numpy(non_zero_Si_indices['fc3']))
    new_model.fc3.weight.data = x

    new_model.fc3.bias.data = model.fc3.bias.data

    masks = [non_zero_Si_indices['fc1']]

    loss = utils.compute_loss(model, test_loader, device, criterion)
    new_loss = utils.compute_loss(new_model, test_loader, device, criterion, masks)

    truncation_error.append((loss - new_loss)**2)

print(truncation_error)
print('Average truncation error when threshold is {}: {}'.format(threshold, np.average(truncation_error)))