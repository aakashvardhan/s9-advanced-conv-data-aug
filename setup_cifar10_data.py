import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets





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
    


