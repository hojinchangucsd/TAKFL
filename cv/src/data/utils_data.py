import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets 
from PIL import Image
import os
import random
from .datasetzoo import DatasetZoo

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
    
def get_transforms(dataset, noise_level=0, net_id=None, total=0):
    if dataset == 'mnist':
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

    elif dataset == 'usps':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(8, fill=0, padding_mode='constant'),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(8, fill=0, padding_mode='constant'),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif dataset == 'fmnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif dataset == 'cifar10':
        # transform_train = transforms.Compose([
        #     transforms.ToTensor(),
        #     AddGaussianNoise(0., noise_level, net_id, total), 
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # ])

        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     AddGaussianNoise(0., noise_level, net_id, total), 
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # ])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4942, 0.4851, 0.4504), (0.2020, 0.1991, 0.2011)),
        ])
    elif dataset == 'cifar100':
        # transform_train = transforms.Compose([
        #     transforms.ToTensor(),
        #     AddGaussianNoise(0., noise_level, net_id, total), 
        #     transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        # ])

        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     AddGaussianNoise(0., noise_level, net_id, total), 
        #     transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        # ])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2009, 0.1984, 0.2023]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5088, 0.4874, 0.4419], std=[0.2019, 0.2000, 0.2037]),
        ])

    elif dataset == 'svhn':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])

    elif dataset == 'stl10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    elif dataset == 'celeba':
        X_train, y_train, X_test, y_test = load_celeba_data(datadir)
    elif dataset == 'tinyimagenet':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    elif dataset == 'imagenet':
        transform_train = transforms.Compose([
        #transforms.Resize((32, 32)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            #transforms.Resize((32, 32)),
            #transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    elif dataset == 'femnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform_train = transform_test = None
    
    return transform_train, transform_test

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(np.sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:,row*size+i,col*size+j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class DatasetKD(Dataset):
    def __init__(self, dataset, logits):
        self.dataset = dataset
        self.logits = logits
    
    def set_logits(self, logits):
        self.logits = logits
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        logits = self.logits[item]
        return image, label, logits
    
class DatasetKD_ET(Dataset):
    def __init__(self, dataset, logits, labels):
        self.dataset = dataset
        self.logits = logits
        self.labels = labels
    
    def set_logits(self, logits):
        self.logits = logits
        
    def set_labels(self, labels):
        self.labels = labels
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        logits = self.logits[item]
        labels = self.labels[item]
        return image, label, logits, labels

class DatasetKD_AE(Dataset):
    def __init__(self, dataset, logits):
        self.dataset = dataset
        self.logits = logits
    
    def set_logits(self, logits):
        self.logits = logits
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        logits = [self.logits[i][item] for i in range(len(self.logits))]
        return image, label, logits
    
class DatasetKD_Self(Dataset):
    def __init__(self, dataset, logits, self_logits):
        self.dataset = dataset
        self.logits = logits
        self.self_logits = self_logits
    
    def set_logits(self, logits):
        self.logits = logits
    
    def set_self_logits(self, logits):
        self.self_logits = logits
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        logits = [self.logits[i][item] for i in range(len(self.logits))]
        self_logits = [self.self_logits[i][item] for i in range(len(self.self_logits))]
        return image, label, logits, self_logits
    
class DatasetKD_Self2(Dataset):
    def __init__(self, dataset, logits, self_logits):
        self.dataset = dataset
        self.logits = logits
        self.self_logits = self_logits
    
    def set_logits(self, logits):
        self.logits = logits
    
    def set_self_logits(self, logits):
        self.self_logits = logits
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        logits = self.logits[item]
        self_logits = [self.self_logits[i][item] for i in range(len(self.self_logits))]
        return image, label, logits, self_logits

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
    
def get_subset(dataset, idxs): 
    return DatasetSplit(dataset, idxs)

def get_dataset_global(dataset, datadir, batch_size=128, p_train=1.0, p_test=1.0, seed=2023):
    transform_train, transform_test = get_transforms(dataset, noise_level=0, net_id=None, total=0)
    
    if dataset == "imagenet":
        train_ds_global = datasets.ImageNet(root=datadir+'imagenet_resized/', split='train', transform=transform_train)
        test_ds_global = datasets.ImageNet(root=datadir+'imagenet_resized/', split='val', transform=transform_train)
    elif dataset == "food101":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5443, 0.4436, 0.3456), (0.2197, 0.2268, 0.2233)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5475, 0.4469, 0.3481), (0.2176, 0.2258, 0.2242)),
        ])
        train_ds_global = datasets.Food101(root=datadir+'resized32_data/', split='train', download=False, transform=transform_train)
        test_ds_global  = datasets.Food101(root=datadir+'resized32_data/', split='test', download=False, transform=transform_test)
    elif dataset == "gtsrb":
        train_ds_global = datasets.GTSRB(root=datadir+'resized32_data/', split='train', download=False, transform=transform_train)
        test_ds_global  = datasets.GTSRB(root=datadir+'resized32_data/', split='test', download=False, transform=transform_train)
    elif dataset == "cub":
        train_ds_global = datasets.ImageFolder(root=datadir+'resized32_data/'+'cub/CUB_200_2011/images', transform=transform_train)
        test_ds_global  = None
    elif dataset == "celeba":
        train_ds_global = datasets.CelebA(root=datadir+'resized32_data/', split='train', download=False, transform=transform_train)
        test_ds_global  = datasets.CelebA(root=datadir+'resized32_data/', split='test', download=False, transform=transform_train)
    elif dataset == "caltech256":
        train_ds_global = datasets.Caltech256(root=datadir+'resized32_data/', download=False, transform=transform_train)
        test_ds_global  = None
    elif dataset == "stanford_cars":
        train_ds_global = datasets.ImageFolder(root=datadir+'resized32_data/stanford_cars/cars_train/cars_train', transform=transform_train)
        test_ds_global  = datasets.ImageFolder(root=datadir+'resized32_data/stanford_cars/cars_test/cars_test', transform=transform_train)
    elif dataset == "tinyimagenet":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4800, 0.4481, 0.3983), (0.2110, 0.2086, 0.2081)),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4821, 0.4496, 0.3988), (0.2110, 0.2085, 0.2082)),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4716, 0.4417, 0.3929), (0.2103, 0.2084, 0.2078)),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
        #train_ds_global = datasets.ImageFolder(root=datadir+'resized32_data/tiny-imagenet-200/train', transform=transform_train)
        train_ds = datasets.ImageFolder(root=datadir+'tiny-imagenet-200/train', transform=transform_train)
        val_ds  = datasets.ImageFolder(root=datadir+'tiny-imagenet-200/val', transform=transform_val)
        test_ds  = datasets.ImageFolder(root=datadir+'tiny-imagenet-200/test', transform=transform_test)
        train_ds_global = train_ds
        test_ds_global = val_ds
        #test_ds_global = torch.utils.data.ConcatDataset([val_ds, test_ds])
    elif dataset == "cinic10":
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std)
        ])
        train_ds = datasets.ImageFolder(root=datadir+'cinic-10/train', transform=transform)
        val_ds = datasets.ImageFolder(root=datadir+'cinic-10/valid', transform=transform)
        test_ds = datasets.ImageFolder(root=datadir+'cinic-10/test', transform=transform)
        whole_ds = torch.utils.data.ConcatDataset([train_ds, val_ds, test_ds])

        train_size = int(229500)
        length = len(whole_ds)
        indices = list(range(length))
        # Shuffle indices with fixed seed
        seed=42
        np.random.seed(seed)
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        #print(f'Whole size: {length}, train: {len(train_indices)}, test: {len(test_indices)}')

        train_ds_global = torch.utils.data.Subset(whole_ds, train_indices)
        test_ds_global = torch.utils.data.Subset(whole_ds, test_indices)
    elif dataset == "stl10":
        transform_train = transforms.Compose([
            transforms.Resize(35),
            transforms.CenterCrop(32),
            transforms.ToTensor(), 
            transforms.Normalize((0.4409, 0.4276, 0.3861), (0.2187, 0.2141, 0.2114)),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        #public_ds = CustomSTL10(root=datadir+'resized32_data/stl10_unlabeled', transform=transform_train)
        train_ds_global = datasets.STL10(root=datadir, split='train', download=True, transform=transform_train)
        test_ds_global = datasets.STL10(root=datadir, split='test', download=True, transform=transform_train)
    else:
        train_ds_global = DatasetZoo(datadir, dataset=dataset, dataidxs=None, train=True, 
                                transform=transform_train, target_transform=None, download=True, p_data=p_train,
                                seed=seed)
    
        test_ds_global = DatasetZoo(datadir, dataset=dataset, dataidxs=None, train=False, 
                                transform=transform_train, target_transform=None, download=True, p_data=p_test,
                                seed=seed)
    
    train_dl_global = DataLoader(dataset=train_ds_global, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dl_global = DataLoader(dataset=test_ds_global, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_ds_global, test_ds_global, train_dl_global, test_dl_global

class CustomSTL10(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.labels = self._process_labels()
        self.image_files = sorted(os.listdir(root))

    def _process_labels(self):
        # Dummy labels for now
        return np.asarray([-1] * len(os.listdir(self.root)))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.image_files[idx])
        image = Image.open(img_name)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_distill_data(dataset, datadir, alg, train_ds_name, num_clusters):
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    if dataset == "imagenet100":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4807, 0.4574, 0.4083), (0.2056, 0.2035, 0.2041)),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_ds = datasets.ImageNet(root=datadir+'imagenet_resized/', split='train', transform=transform_train)
        test_ds = datasets.ImageNet(root=datadir+'imagenet_resized/', split='val', transform=transform_train)
        
        subset100 = np.array([], dtype='int')
        tar = np.array(train_ds.targets)
        #np.random.seed(2023)
        #labels = np.random.choice(np.arange(1000), size=100, replace=False)
        labels = [847, 874, 471, 476, 764, 138,  49, 226, 100, 426, 815, 836, 338,
                669, 743, 912, 320, 843, 796, 322, 261, 136, 841, 460, 699, 935,
                949, 877,  61, 332, 416,  35, 227, 493,  32, 478, 660,  13, 451,
                438, 323, 867, 168,  40, 156, 455, 691, 223, 354, 495, 799, 432,
                158, 866, 657, 768, 183, 852, 727, 249, 402, 507,  12, 880, 995,
                233, 176, 776, 830, 586, 865, 475, 610, 534, 953, 490, 160, 386,
                117, 942, 675,  24, 538, 494, 266, 295, 272,  11,   9, 154, 967,
                901, 123, 649, 737, 121,  20, 439, 641, 398]
        for i in labels:
            subset100 = np.hstack([subset100, np.where(tar==i)[0][0:500]])
        public_ds = torch.utils.data.Subset(train_ds, subset100)
    elif dataset == "food101":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5443, 0.4436, 0.3456), (0.2197, 0.2268, 0.2233)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5475, 0.4469, 0.3481), (0.2176, 0.2258, 0.2242)),
        ])
        train_ds = datasets.Food101(root=datadir+'resized32_data/', split='train', download=False, transform=transform_train)
        test_ds  = datasets.Food101(root=datadir+'resized32_data/', split='test', download=False, transform=transform_test)
        public_ds = train_ds
    elif dataset == "gtsrb":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3413, 0.3122, 0.3211), (0.1654, 0.1653, 0.1753)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3370, 0.3092, 0.3204), (0.1674, 0.1672, 0.1782)),
        ])
        train_ds = datasets.GTSRB(root=datadir+'resized32_data/', split='train', download=False, transform=transform_train)
        test_ds  = datasets.GTSRB(root=datadir+'resized32_data/', split='test', download=False, transform=transform_train)
        public_ds = torch.utils.data.ConcatDataset([train_ds, test_ds])
    elif dataset == "cub":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4859, 0.4997, 0.4322), (0.1581, 0.1579, 0.1657)),
        ])
        train_ds = datasets.ImageFolder(root=datadir+'resized32_data/'+'cub/CUB_200_2011/images', transform=transform_train)
        public_ds = train_ds
    elif dataset == "celeba":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5059, 0.4260, 0.3840), (0.2533, 0.2376, 0.2350)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4995, 0.4191, 0.3774), (0.2559, 0.2384, 0.2351)),
        ])
        train_ds = datasets.CelebA(root=datadir+'resized32_data/', split='train', download=False, transform=transform_train)
        test_ds  = datasets.CelebA(root=datadir+'resized32_data/', split='test', download=False, transform=transform_test)
        
        #public_size = int(len(train_ds)*0.6)
        public_size = int(60000)
        public_ds = torch.utils.data.Subset(train_ds, range(public_size))
        #public_ds = train_ds
        #public_ratio = 0.5  # 50% of train_ds will be used as public_ds
        #public_size = int(public_ratio * len(train_ds))
        #train_size = len(train_ds) - public_size
        #public_ds, train_ds = torch.utils.data.random_split(train_ds, [public_size, train_size])
    elif dataset == "caltech256":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5504, 0.5325, 0.5039), (0.2165, 0.2168, 0.2196)),
        ])
        train_ds = datasets.Caltech256(root=datadir+'resized32_data/', download=False, transform=transform_train)
        public_ds = train_ds
    elif dataset == "stanford_cars":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4704, 0.4600, 0.4548), (0.2399, 0.2417, 0.2456)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4687, 0.4586, 0.4538), (0.2393, 0.2406, 0.2445)),
        ])
        train_ds = datasets.ImageFolder(root=datadir+'resized32_data/stanford_cars/cars_train', transform=transform_train)
        test_ds  = datasets.ImageFolder(root=datadir+'resized32_data/stanford_cars/cars_test', transform=transform_test)
        public_ds = torch.utils.data.ConcatDataset([train_ds, test_ds])
    elif dataset == "tinyimagenet":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4800, 0.4481, 0.3983), (0.2110, 0.2086, 0.2081)),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4821, 0.4496, 0.3988), (0.2110, 0.2085, 0.2082)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4716, 0.4417, 0.3929), (0.2103, 0.2084, 0.2078)),
        ])
        train_ds = datasets.ImageFolder(root=datadir+'resized32_data/tiny-imagenet-200/train', transform=transform_train)
        val_ds  = datasets.ImageFolder(root=datadir+'resized32_data/tiny-imagenet-200/val', transform=transform_val)
        test_ds  = datasets.ImageFolder(root=datadir+'resized32_data/tiny-imagenet-200/test', transform=transform_test)
        public_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])
        
        public_size = int(60000)
        public_ds = torch.utils.data.Subset(public_ds, range(public_size))
    elif dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4942, 0.4851, 0.4504), (0.2020, 0.1991, 0.2011)),
        ])
        train_ds = datasets.CIFAR10(root=datadir, train=True, download=False, transform=transform_train)
        test_ds = datasets.CIFAR10(root=datadir, train=False, download=False, transform=transform_test)
        public_ds = torch.utils.data.ConcatDataset([train_ds, test_ds])
    elif dataset == "cifar100":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2009, 0.1984, 0.2023]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5088, 0.4874, 0.4419], std=[0.2019, 0.2000, 0.2037]),
        ])
        train_ds = datasets.CIFAR100(root=datadir, train=True, download=False, transform=transform_train)
        test_ds = datasets.CIFAR100(root=datadir, train=False, download=False, transform=transform_test)
        public_ds = torch.utils.data.ConcatDataset([train_ds, test_ds])
    elif dataset == "svhn":
        transform_train = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.1201, 0.1231, 0.1052))
        ])
        # transform_train = transforms.Compose([
        #     transforms.ToTensor(), 
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])
        transform_test = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])
        train_ds = datasets.SVHN(root=datadir, split= 'train', download=True, transform=transform_train)
        public_ds = train_ds
        #public_size = int(60000)
        #public_ds = torch.utils.data.Subset(public_ds, range(public_size))
    elif dataset == "stl10":
        transform_train = transforms.Compose([
            transforms.Resize(65),
            transforms.CenterCrop(64),
            transforms.ToTensor(), 
            transforms.Normalize((0.4409, 0.4276, 0.3861), (0.2187, 0.2141, 0.2114)),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        #public_ds = CustomSTL10(root=datadir+'resized32_data/stl10_unlabeled', transform=transform_train)
        public_ds = datasets.STL10(root=datadir, split='unlabeled', download=True, transform=transform_train)
        
        train_size = int(30000)
        length = len(public_ds)
        indices = list(range(length))
        seed=42
        np.random.seed(seed)
        np.random.shuffle(indices)

        ds_indices = indices[:train_size]
        public_ds = torch.utils.data.Subset(public_ds, ds_indices)
        
    public_ds_obj = get_distill_dataobj(alg, train_ds_name, num_clusters, public_ds)
    
    return public_ds_obj
    
def get_distill_dataobj(alg, train_ds_name, num_clusters, public_ds):
    if train_ds_name == "cifar10":
        num_classes = 10
    elif train_ds_name == "cifar100":
        num_classes = 100
    elif train_ds_name == "cinic10":
        num_classes = 10
    elif train_ds_name == "tinyimagenet":
        num_classes = 200
    elif train_ds_name == "food101":
        num_classes = 101
    elif train_ds_name == "stl10":
        num_classes = 10
    
    if alg in ["fedmh", "fedavg_mh", "fedprox_mh"]:
        soft_t = np.random.randn(len(public_ds), num_classes)
        public_ds_obj = DatasetKD(public_ds, soft_t)
    elif alg == "fedmhR":
        soft_t = np.random.randn(len(public_ds), num_classes)
        public_ds_obj = DatasetKD_Self2(public_ds, soft_t, [soft_t for _ in range(num_clusters)])
    elif alg == "fedet":
        soft_t = np.random.randn(len(public_ds), num_classes)
        hard_t = np.random.randn(len(public_ds), 1)
        public_ds_obj = DatasetKD_ET(public_ds, logits=soft_t, labels=hard_t)
    elif alg in ["fedhd", "fedmhT", "fedmhw_reg", "fedmhw"]:
        soft_t = np.random.randn(len(public_ds), num_classes)
        public_ds_obj = DatasetKD_Self(public_ds, [soft_t for _ in range(num_clusters)], [soft_t for _ in range(num_clusters)])
    
    return public_ds_obj
    
def dir_partition(num_users, niid_beta=0.5, nclasses=10, y_train=None, y_test=None, train_inds=None):
    idxs_train = np.arange(len(y_train))
    idxs_test = np.arange(len(y_test))

    n_train = y_train.shape[0]

    partitions_train = {i:np.array([],dtype='int') for i in range(num_users)}
    partitions_test = {i:np.array([],dtype='int') for i in range(num_users)}
    partitions_train_stat = {}
    partitions_test_stat = {}
    
    min_size = 0
    min_require_size = 3
    #np.random.seed(2022)
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(nclasses):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)

            proportions = np.random.dirichlet(np.repeat(niid_beta, num_users))
            proportions = np.array([p * (len(idx_j) < n_train/num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])
        
    #### Assigning samples to each client         
    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        partitions_train[j] = np.hstack([partitions_train[j], idx_batch[j]])

        dataidx = partitions_train[j]          
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        partitions_train_stat[j] = tmp

        for key in tmp.keys():
            dataidx = np.where(y_test==key)[0]
            partitions_test[j] = np.hstack([partitions_test[j], dataidx])

        dataidx = partitions_test[j]
        unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        partitions_test_stat[j] = tmp
        
    for j in range(num_users):
        partitions_train[j] = np.array(train_inds)[partitions_train[j]]
        
    return (partitions_train, partitions_test, partitions_train_stat, partitions_test_stat)


def iid_partition(num_users, nclasses=10, y_train=None, y_test=None, train_inds=None):
    idxs_train = np.arange(len(y_train))
    idxs_test = np.arange(len(y_test))

    n_train = y_train.shape[0]

    partitions_train = {i:np.array([],dtype='int') for i in range(num_users)}
    partitions_test = {i:np.array([],dtype='int') for i in range(num_users)}
    partitions_train_stat = {}
    partitions_test_stat = {}
    
    ind2label = {cls: np.array_split([i for i, label in enumerate(y_train) if label == cls], num_users) for cls in range(nclasses)}
    
    #print(f'IID Spliting: {ind2label}')
    for j in range(num_users):
        for cls in range(nclasses):
            partitions_train[j] = np.hstack([partitions_train[j], ind2label[cls][j]])
        
        dataidx = partitions_train[j]
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        partitions_train_stat[j] = tmp
        
        partitions_test[j] = np.hstack([partitions_test[j], idxs_test])

        dataidx = partitions_test[j]
        unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        partitions_test_stat[j] = tmp
        
    for j in range(num_users):
        partitions_train[j] = np.array(train_inds)[partitions_train[j]]
        
    return (partitions_train, partitions_test, partitions_train_stat, partitions_test_stat)

def get_partitions(num_users, train_ds_global, test_ds_global, args):
    
    if args.dataset == 'cifar10' and args.clustering_setting == '3_clusters' and not args.old_type:
        nclasses = 10
        
        X_train = train_ds_global.data
        Y_train = np.array(train_ds_global.target)

        X_test = test_ds_global.data
        Y_test = np.array(test_ds_global.target)
        
        indices = list(range(len(train_ds_global)))
        ind2label = {cls: [i for i, label in enumerate(Y_train) if label == cls] for cls in range(nclasses)}
        random.shuffle(indices)

        partitions_train = []
        partitions_test = []
        partitions_train_stat = []
        partitions_test_stat = []
        
        for k in range(len(num_users)):
            if k == 0:
                inds_subset = np.array([], dtype='int')
                for cls in range(nclasses):
                    r_inds = np.random.choice(np.array(ind2label[cls]), 500, replace=False)
                    inds_subset = np.hstack([inds_subset, r_inds])
                    ind2label[cls] = [i for i in ind2label[cls] if i not in r_inds]
                    
                random.shuffle(inds_subset)
                inds_subset = list(inds_subset)
            elif k == 1:
                inds_subset = np.array([], dtype='int')
                for cls in range(nclasses):
                    r_inds = np.random.choice(np.array(ind2label[cls]), 2000, replace=False)
                    inds_subset = np.hstack([inds_subset, r_inds])
                    ind2label[cls] = [i for i in ind2label[cls] if i not in r_inds]
                
                random.shuffle(inds_subset)
                inds_subset = list(inds_subset)
            else: 
                inds_subset = np.array([], dtype='int')
                for cls in range(nclasses):
                    #r_inds = np.random.choice(np.array(ind2label[cls]), 2000, replace=False)
                    r_inds = np.array(ind2label[cls])
                    inds_subset = np.hstack([inds_subset, r_inds])
                    ind2label[cls] = [i for i in ind2label[cls] if i not in r_inds]
                
                random.shuffle(inds_subset)
                inds_subset = list(inds_subset)
                
            Y_train_temp = Y_train[inds_subset]

            if args.partition == 'niid-labeldir':
                partitions_train_tmp, partitions_test_tmp, partitions_train_stat_tmp, \
                partitions_test_stat_tmp= dir_partition(num_users[k], niid_beta=args.niid_beta, nclasses=nclasses, 
                                                        y_train=Y_train_temp, y_test=Y_test, train_inds=inds_subset)
            elif args.partition == 'iid':
                partitions_train_tmp, partitions_test_tmp, partitions_train_stat_tmp, \
                partitions_test_stat_tmp= iid_partition(num_users[k], nclasses=nclasses, 
                                                        y_train=Y_train_temp, y_test=Y_test, train_inds=inds_subset)

            partitions_train.append(partitions_train_tmp)
            partitions_test.append(partitions_test_tmp)
            partitions_train_stat.append(partitions_train_stat_tmp)
            partitions_test_stat.append(partitions_test_stat_tmp)
        
    elif args.dataset == 'cifar10' and args.clustering_setting == '3_clusters' and args.old_type:
        print('!!!!!!!!!!!!!!!!!!!!! OLD TYPE PARTITIONING !!!!!!!!!!!!!!!!!!!!!!')
        nclasses = 10
        
        X_train = train_ds_global.data
        Y_train = np.array(train_ds_global.target)

        X_test = test_ds_global.data
        Y_test = np.array(test_ds_global.target)
        
        indices = list(range(len(train_ds_global)))
        random.shuffle(indices)

        partitions_train = []
        partitions_test = []
        partitions_train_stat = []
        partitions_test_stat = []
        
        for k in range(len(num_users)):
            if k == 0:
                inds_subset = indices[0:5000]
            elif k == 1:
                inds_subset = indices[5000:25000]
            else: 
                inds_subset = indices[25000:]
                
            Y_train_temp = Y_train[inds_subset]

            if args.partition == 'niid-labeldir':
                partitions_train_tmp, partitions_test_tmp, partitions_train_stat_tmp, \
                partitions_test_stat_tmp= dir_partition(num_users[k], niid_beta=args.niid_beta, nclasses=nclasses, 
                                                        y_train=Y_train_temp, y_test=Y_test, train_inds=inds_subset)
            elif args.partition == 'iid':
                partitions_train_tmp, partitions_test_tmp, partitions_train_stat_tmp, \
                partitions_test_stat_tmp= iid_partition(num_users[k], nclasses=nclasses, 
                                                        y_train=Y_train_temp, y_test=Y_test, train_inds=inds_subset)

            partitions_train.append(partitions_train_tmp)
            partitions_test.append(partitions_test_tmp)
            partitions_train_stat.append(partitions_train_stat_tmp)
            partitions_test_stat.append(partitions_test_stat_tmp)
            
    elif args.dataset == 'cifar100' and args.clustering_setting == '3_clusters' and not args.old_type:
        print('CIFAR-100 Partitioning for 3 clusters setting')
        nclasses = 100
        
        X_train = train_ds_global.data
        Y_train = np.array(train_ds_global.target)

        X_test = test_ds_global.data
        Y_test = np.array(test_ds_global.target)
        
        indices = list(range(len(train_ds_global)))
        ind2label = {cls: [i for i, label in enumerate(Y_train) if label == cls] for cls in range(nclasses)}
        random.shuffle(indices)

        partitions_train = []
        partitions_test = []
        partitions_train_stat = []
        partitions_test_stat = []
        
        for k in range(len(num_users)):
            if k == 0:
                inds_subset = np.array([], dtype='int')
                for cls in range(nclasses):
                    r_inds = np.random.choice(np.array(ind2label[cls]), 50, replace=False)
                    inds_subset = np.hstack([inds_subset, r_inds])
                    ind2label[cls] = [i for i in ind2label[cls] if i not in r_inds]
                    
                random.shuffle(inds_subset)
                inds_subset = list(inds_subset)
            elif k == 1:
                inds_subset = np.array([], dtype='int')
                for cls in range(nclasses):
                    r_inds = np.random.choice(np.array(ind2label[cls]), 200, replace=False)
                    inds_subset = np.hstack([inds_subset, r_inds])
                    ind2label[cls] = [i for i in ind2label[cls] if i not in r_inds]
                
                random.shuffle(inds_subset)
                inds_subset = list(inds_subset)
            else: 
                inds_subset = np.array([], dtype='int')
                for cls in range(nclasses):
                    #r_inds = np.random.choice(np.array(ind2label[cls]), 2000, replace=False)
                    r_inds = np.array(ind2label[cls])
                    inds_subset = np.hstack([inds_subset, r_inds])
                    ind2label[cls] = [i for i in ind2label[cls] if i not in r_inds]
                
                random.shuffle(inds_subset)
                inds_subset = list(inds_subset)
                
            Y_train_temp = Y_train[inds_subset]

            if args.partition == 'niid-labeldir':
                partitions_train_tmp, partitions_test_tmp, partitions_train_stat_tmp, \
                partitions_test_stat_tmp= dir_partition(num_users[k], niid_beta=args.niid_beta, nclasses=nclasses, 
                                                        y_train=Y_train_temp, y_test=Y_test, train_inds=inds_subset)
            elif args.partition == 'iid':
                partitions_train_tmp, partitions_test_tmp, partitions_train_stat_tmp, \
                partitions_test_stat_tmp= iid_partition(num_users[k], nclasses=nclasses, 
                                                        y_train=Y_train_temp, y_test=Y_test, train_inds=inds_subset)

            partitions_train.append(partitions_train_tmp)
            partitions_test.append(partitions_test_tmp)
            partitions_train_stat.append(partitions_train_stat_tmp)
            partitions_test_stat.append(partitions_test_stat_tmp) 
        
    return partitions_train, partitions_test, partitions_train_stat, partitions_test_stat


def get_partitions_customD(train_ds_global, test_ds_global, args):
    if args.dataset == 'cifar10':
        nclasses = 10
        samples_pc = [5000 for i in range(nclasses)]
    elif args.dataset == 'cifar100':
        nclasses = 100
        samples_pc = [500 for i in range(nclasses)]
    elif args.dataset == 'cinic10':
        nclasses = 10
        samples_pc = 500
    elif args.dataset == 'tinyimagenet':
        nclasses = 200
        samples_pc = 500
    elif args.dataset == 'food101':
        nclasses = 101
        samples_pc = 500
    elif args.dataset == 'stl10':
        nclasses = 10
        samples_pc = 500
        
    if args.dataset in ["cifar10", "cifar100"]:
        #X_train = train_ds_global.data
        #Y_train = np.array(train_ds_global.target)
        Y_train = np.array([el[1] for el in train_ds_global])

        #X_test = test_ds_global.data
        #Y_test = np.array(test_ds_global.target)
        Y_test = np.array([el[1] for el in test_ds_global])
    elif args.dataset in ["cinic10", "tinyimagenet", "food101", "stl10"]:
        Y_train = np.array([el[1] for el in train_ds_global])
        Y_test = np.array([el[1] for el in test_ds_global])
    
    indices = list(range(len(train_ds_global)))
    ind2label = {cls: [i for i, label in enumerate(Y_train) if label == cls] for cls in range(nclasses)}
    samples_pc = [len(ind2label[cls]) for cls in range(nclasses)]
    random.shuffle(indices)

    partitions_train = []
    partitions_test = []
    partitions_train_stat = []
    partitions_test_stat = []
    
    for k in range(len(args.num_users)):
        inds_subset = np.array([], dtype='int')
        for cls in range(nclasses):
            np.random.seed(2022)
            nn = min(int(args.data_ratios[k]*samples_pc[cls]), len(ind2label[cls]))
            r_inds = np.random.choice(np.array(ind2label[cls]), nn, replace=False)
            inds_subset = np.hstack([inds_subset, r_inds])
            ind2label[cls] = [i for i in ind2label[cls] if i not in r_inds]
            
        random.shuffle(inds_subset)
        inds_subset = list(inds_subset)
        
        Y_train_temp = Y_train[inds_subset]

        if args.partition == 'niid-labeldir':
            partitions_train_tmp, partitions_test_tmp, partitions_train_stat_tmp, \
            partitions_test_stat_tmp= dir_partition(args.num_users[k], niid_beta=args.niid_beta, nclasses=nclasses, 
                                                    y_train=Y_train_temp, y_test=Y_test, train_inds=inds_subset)
        elif args.partition == 'iid':
            partitions_train_tmp, partitions_test_tmp, partitions_train_stat_tmp, \
            partitions_test_stat_tmp= iid_partition(args.num_users[k], nclasses=nclasses, 
                                                    y_train=Y_train_temp, y_test=Y_test, train_inds=inds_subset)

        partitions_train.append(partitions_train_tmp)
        partitions_test.append(partitions_test_tmp)
        partitions_train_stat.append(partitions_train_stat_tmp)
        partitions_test_stat.append(partitions_test_stat_tmp)
    
    return partitions_train, partitions_test, partitions_train_stat, partitions_test_stat
        