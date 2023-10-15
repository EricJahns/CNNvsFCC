import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.input_size = 28*28
        self.output_size = 10
        self.kernel_size = 4

        self.conv1 = nn.Conv2d(1, self.kernel_size, 3, padding=1)
        self.conv2 = nn.Conv2d(self.kernel_size, self.kernel_size*2, 3, padding=1)
        self.conv3 = nn.Conv2d(self.kernel_size*2, self.kernel_size*4, 3, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.kernel_size*4*3*3, self.output_size)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return self.softmax(self.fc1(self.flatten(x)))
    
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)