import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from config import get_config
torch.manual_seed(1)

config = get_config()

def model_summary(model, input_size=(1, 28, 28)):
    from torchsummary import summary
    summary(model, input_size=input_size)
    
def sgd_optimizer(model, lr=config['lr'], momentum=0.9, weight_decay=0.0):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def save_model(model, path):
    # Save the model
    torch.save(model.state_dict(), path)
    
def load_model(model, path):
    # Load the model
    model.load_state_dict(torch.load(path))
    return model