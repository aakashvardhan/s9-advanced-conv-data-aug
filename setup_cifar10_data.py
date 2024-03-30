import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
import torch




class CIFAR10Dataset(datasets.CIFAR10):
    
    def __init__(self, root="~/data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
        if transform == "train":
            self.transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=(mean_value), p=0.75),
                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                ToTensorV2(),
            ])
        elif transform == "test":
            self.transform = A.Compose([
                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                ToTensorV2(),
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
    


def setup_cifar10(config):
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(1)

    if cuda:
        torch.cuda.manual_seed(1)
    train_data = CIFAR10Dataset(root='./data', train=True, download=True, transform="train")
    test_data = CIFAR10Dataset(root='./data', train=False, download=True, transform="test")
    
    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=config['batch_size'], num_workers=config['num_workers'], pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)

    return train_data,test_data,train_loader, test_loader