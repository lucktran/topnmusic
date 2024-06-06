import torch.nn as nn

# Define the neural network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        # self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        # self.relu4 = nn.ReLU()
        # self.fc5 = nn.Linear(hidden_size4, hidden_size5)
        # self.relu5 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        # out = self.relu4(out)
        # out = self.fc5(out)
        # out = self.relu5(out)
        # out = self.fc6(out)
        out = self.softmax(out)
        return out

