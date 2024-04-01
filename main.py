from models.model import Net
from models.model_utils import model_summary, sgd_optimizer, save_model, load_model
from config import get_config
from utils import train,test
from visualize import show_misclassified_images, plt_misclassified_images
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from setup_cifar10_data import setup_cifar10
import utils
def main(config):
    train_data,test_data,train_loader, test_loader = setup_cifar10(config)
    model = Net(config).to(config['device'])
    model_summary(model, input_size=(3, 32, 32))
    optimizer = sgd_optimizer(model, lr=config['lr'])
    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=0.5)
    lr_plateau = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
    lr = []
    for epoch in range(1,config['epochs']+1):
        print("EPOCH:", epoch)
        train(model, config['device'], train_loader, optimizer, epoch)
        test_loss = test(model, config['device'], test_loader)
        if config['lr_scheduler'] == 'step_lr':
            scheduler.step()
            lr.append(optimizer.param_groups[0]['lr'])
            print("Learning rate:", optimizer.param_groups[0]['lr'])
        elif config['lr_scheduler'] == 'plateau':
            lr_plateau.step(utils.test_losses[-1])
            lr.append(optimizer.param_groups[0]['lr'])
            print("Learning rate:", optimizer.param_groups[0]['lr'])
        elif config['lr_scheduler'] == 'none':
            continue
    
    # format name of model file according to config['norm']
    model_file = 'model_' + config['norm'] + '.pth'
    save_model(model, model_file)
    
    return model, test_loader, lr

# if __name__ == '__main__':
#     config = get_config()
#     model, test_loader = main(config, lr_scheduler=False)
#     show_misclassified_images(model, test_loader, config['device'])
#     plt_misclassified_images()