import torch
import numpy as np

def compute_sensitivity_indices(inputs, test_model):
    Si = np.zeros((inputs[0].shape[1]))
    rng = np.random.default_rng()
    index = rng.choice(len(inputs)-1, size=400, replace=False)
    mean = 0
    for i in index:
        A = inputs[i]
        mean += np.mean(A, axis=0)
        j = np.random.randint(0, len(inputs))
        while j == i:
            j = np.random.randint(0, len(inputs))
        B = inputs[j]

        for d in range(inputs[0].shape[1]):
            A_ = np.copy(A)
            A_[:, d] = B[:, d]
            outputs_ = test_model(torch.from_numpy(A_))
            outputs = test_model(torch.from_numpy((A)))
            Si[d] += (torch.mean(outputs_) - torch.mean(outputs)).item() ** 2

    Si = Si / 400
    mean = mean / 400

    return Si, mean

def prune_weights(model, low_Si_indices, mean):
    bias_adjustment = 0
    with torch.no_grad():
        for i in low_Si_indices:
            bias_adjustment += model.fc2.weight[:, i].detach().numpy() * mean[i]
            model.fc2.weight[:, i].zero_()
            model.fc1.weight[i, :].zero_()
            model.fc1.bias[i].zero_()
        model.fc2.bias.add_(torch.from_numpy(bias_adjustment))

    return model

def count_zero_params(model):
    zeros = 0
    for param in model.parameters():
        if param.requires_grad:
            zeros += torch.sum((param.data==0).int()).item()
    return zeros

def test_model(model, test_loader, device, masks):
    # test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            for mask in masks:
                images = torch.index_select(images, 1, torch.from_numpy(mask)).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

    return 100 * correct / total

def train_model(model, train_loader, num_iter, device, criterion, optimizer, masks):
    # Train the model
    i = 0
    while i < num_iter:
        for images, labels in train_loader:
            # Move tensors to the configured device
            images = images.reshape(-1, 28 * 28).to(device)
            for mask in masks:
                images = torch.index_select(images, 1, torch.from_numpy(mask)).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backprpagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10000 == 0:
                print('Iteration [{}/{}], Loss: {:.4f}'
                      .format(i + 1, num_iter, loss.item()))

            i += 1

    return model