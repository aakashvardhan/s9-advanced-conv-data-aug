import torch
import torch.optim as optim
from config import get_config

torch.manual_seed(1)

config = get_config()


def model_summary(model, input_size=(1, 28, 28)):
    """
    Prints a summary of the given model architecture.

    Args:
        model (torch.nn.Module): The model to summarize.
        input_size (tuple, optional): The input size of the model. Defaults to (1, 28, 28).
    """
    from torchsummary import summary

    summary(model, input_size=input_size)


def sgd_optimizer(model, lr=config["lr"], momentum=0.9, weight_decay=0.0):
    """
    Returns a stochastic gradient descent (SGD) optimizer for the given model.

    Args:
        model (torch.nn.Module): The model for which the optimizer is created.
        lr (float, optional): The learning rate for the optimizer. Defaults to the value specified in the config.
        momentum (float, optional): The momentum factor for the optimizer. Defaults to 0.9.
        weight_decay (float, optional): The weight decay factor for the optimizer. Defaults to 0.0.

    Returns:
        torch.optim.SGD: The SGD optimizer for the given model.
    """
    return optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )


def save_model(model, path):
    """
    Save the model to the specified path.

    Args:
        model (torch.nn.Module): The model to be saved.
        path (str): The path where the model should be saved.

    Returns:
        None
    """
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """
    Loads the model from the specified path.

    Args:
        model (torch.nn.Module): The model to load the state_dict into.
        path (str): The path to the saved model state_dict.

    Returns:
        torch.nn.Module: The loaded model.
    """
    # Load the model
    model.load_state_dict(torch.load(path))
    return model
