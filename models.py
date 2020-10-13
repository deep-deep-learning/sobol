import torch
import torch.nn as nn

# LeNet-300-100
class LeNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        super(LeNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        return out

# test net fc2
class TestNet_fc2(nn.Module):
    def __init__(self, hidden_size_1, hidden_size_2, num_classes):
        super(TestNet_fc2, self).__init__()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.tanh = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        out = self.fc2(x)
        out = self.tanh(out)
        out = self.fc3(out)
        return out

# test net fc3
class TestNet_fc3(nn.Module):
    def __init__(self, hidden_size_2, num_classes):
        super(TestNet_fc3, self).__init__()
        self.fc3 = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        out = self.fc3(x)
        return out

'''
# print info
print('60 Lowest Si Indices in FC2: ')
print(low_Si_indices)
print('Number of samples: '+str(int(len(inputs_fc2)*inputs_fc2[0].shape[0]/2)))

# plot Si
indices = range(hidden_size_1)
x_pos = np.arange(len(indices))

fig, ax = plt.subplots()
ax.bar(x_pos, Si_fc2, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_title('Si of 300 hidden units in FC2 '+str(len(inputs_fc2)*inputs_fc2[0].shape[0]/2))

plt.tight_layout()
plt.savefig('Si of 300 hidden units in FC2 '+str(len(inputs_fc2)*inputs_fc2[0].shape[0]/2)+'.png')
plt.close()
'''