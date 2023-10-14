import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.input_size = 28*28
        self.output_size = 10

        self.fc1 = nn.Linear(self.input_size, self.input_size//2)
        self.fc2 = nn.Linear(self.input_size//2, self.output_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.relu(self.fc1(x))
        return self.softmax(self.fc2(x))
    
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)