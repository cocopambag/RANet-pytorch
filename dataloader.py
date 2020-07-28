import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image



def get_dataloaders(args):
    train_loader, val_loader, test_loader = None, None, None
    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(args.data_root, download=True, train=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]))
    elif args.data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(args.data_root, train=True,
                                      transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                      ]))
        val_set = datasets.CIFAR100(args.data_root, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    else:
        # ImageNet
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))
    if args.use_valid:
        train_set_index = torch.randperm(len(train_set))
        if os.path.exists(os.path.join(args.save, 'index.pth')):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
        else:
            print('!!!!!! Save train_set_index !!!!!!')
            torch.save(train_set_index, os.path.join(args.save, 'index.pth'))
        if args.data.startswith('cifar'):
            num_sample_valid = 5000
        else:
            num_sample_valid = 50000

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[:-num_sample_valid]),
                num_workers=args.workers, pin_memory=False)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[-num_sample_valid:]),
                num_workers=args.workers, pin_memory=False)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=False)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=False)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=False)
            test_loader = val_loader

    return train_loader, val_loader, test_loader

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, x, y, x_transform):
        self.x = x
        self.y = y
        self.x_transform = x_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = Image.fromarray((self.x[item] * 255).astype(np.uint8))
        x = self.x_transform(x)

        y = self.y[item]
        return x, y


def make_tensor(path):
    from matplotlib import pyplot as plt
    directory = next(os.walk(path))[1]
    x = []
    y = []
    idx = 0
    for dir in directory:
        image_path = next(os.walk(path + '/' + dir))[2]
        if not image_path:
            continue
        for i in image_path:
            image = plt.imread(path + '/' + dir + '/' + i)
            x += [image]
            y += [idx]
        idx += 1
    return x, y


def get_custom_dataloader(args):
    path = args.custom_path
    tr_path = path + '/train'
    tr_file = path + '/train.npy'

    val_path = path + '/validation'
    val_file = path + '/validation.npy'

    te_path = path + '/test'
    te_file = path + '/test.npy'

    tr_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    val_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    te_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    if os.path.isfile(tr_file):
        load = np.load(tr_file, allow_pickle=True)
        tr_x, tr_y = load[0], load[1]
    else:
        tr_x, tr_y = make_tensor(tr_path)
        np.save(tr_file, np.array([tr_x, tr_y]))

    if os.path.isfile(val_file):
        load = np.load(val_file, allow_pickle=True)
        val_x, val_y = load[0], load[1]
    else:
        val_x, val_y = make_tensor(val_path)
        np.save(val_file, np.array([val_x, val_y]))

    if os.path.isfile(te_file):
        load = np.load(te_file, allow_pickle=True)
        te_x, te_y = load[0], load[1]
    else:
        te_x, te_y = make_tensor(te_path)
        np.save(te_file, np.array([te_x, te_y]))

    tr_data = CustomDataset(tr_x, tr_y, tr_transform)
    val_data = CustomDataset(val_x, val_y, val_transform)
    te_data = CustomDataset(te_x, te_y, te_transform)

    tr_loader = DataLoader(tr_data, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=False)
    te_loader = DataLoader(te_data, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=False)

    return tr_loader, val_loader, te_loader


if __name__ == '__main__':
    get_custom_dataloader(1)