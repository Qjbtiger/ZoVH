import torch
import torch.nn as nn
import torchvision
from model.cnn import CNN
from model.synthetic_functions import SyntheticFunction

class Attack(SyntheticFunction):
    def __init__(self, x_init, lb=-0.2, ub=0.2, idx=1):
        super(Attack, self).__init__(x_init)

        assert len(x_init.shape) == 1, "x_init must be a 1D tensor"
        assert x_init.shape[0] == 28 * 28, "x_init must be of shape (28*28,)"

        # self.x = torch.nn.Parameter(x_init) # already defined in the parent class
        self.dim = 28 * 28
        self.device = x_init.device

        self.lb = lb
        self.ub = ub
        
        dataset = torchvision.datasets.MNIST('data', download=True, train=False, transform=torchvision.transforms.ToTensor())
        self.size = (1, 1, 28, 28)

        self.model = CNN().to(self.device)
        self.model.load_state_dict(torch.load('data/cnn/mnist_cnn.pt', map_location=self.device))
        self.model.eval()

        self.data, self.target = dataset[idx]
        self.data = self.data.to(self.device)

    def get_loss(self, image, label, targeted=True):
        input_image = (image - 0.1307) / 0.3081
        output = self.model(input_image)
        label_term = output[:, label]
        other = output + 0.0
        other[:, label] = -1e8
        other_term = torch.max(other, dim=1).values
        if targeted:
            loss = label_term - other_term
        else:
            loss = other_term - label_term
        
        loss = torch.squeeze(loss)
        
        return loss

    def get_pred(self, image):
        input_image = (image - 0.1307) / 0.3081
        output = self.model(input_image).detach().squeeze()
        return torch.argmax(output)
        
    def forward(self):
        x = self.x.reshape(*self.size)
        
        new_image = torch.clamp(x, self.lb, self.ub) + self.data
        new_image = torch.clamp(new_image, 0, 1)
        
        # pass through the gradient from new_image to x directly
        new_image = x + (new_image - x).detach()
        
        target_label = (self.target + 1) % 10
        
        loss = self.get_loss(new_image, target_label, targeted=True)

        return - loss
    