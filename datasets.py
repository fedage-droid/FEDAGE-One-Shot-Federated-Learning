import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np

# Very lightweight CPU transform
base_transform = transforms.ToTensor()

def get_dataset(name, root="./data", train=True, download=True):
    name = name.lower()
    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=train, transform=base_transform, download=download)
    elif name == "cifar100":
        return datasets.CIFAR100(root=root, train=train, transform=base_transform, download=download)
    elif name == "stl10":
        split = 'train' if train else 'test'
        return datasets.STL10(root=root, split=split, transform=base_transform, download=download)
    elif name == "svhn":
        split = 'train' if train else 'test'
        return datasets.SVHN(root=root, split=split, transform=base_transform, download=download)
    elif name == "mnist" or name == "fashionmnist":
        dataset_class = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
        return dataset_class(root=root, train=train, transform=base_transform, download=download)
    else:
        raise ValueError(f"Dataset {name} is not supported.")

def dirichlet_partition(dataset, num_clients=3, alpha=0.5, min_size=10):
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise AttributeError("Dataset must have 'targets' or 'labels' attribute")

    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))

        splits = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        split_indices = np.split(class_indices, splits)

        for i, idx in enumerate(split_indices):
            client_indices[i].extend(idx)

    # Ensure minimum size
    for i in range(num_clients):
        if len(client_indices[i]) < min_size:
            largest_client = np.argmax([len(client_indices[j]) for j in range(num_clients)])
            needed = min_size - len(client_indices[i])
            if len(client_indices[largest_client]) > needed:
                client_indices[i].extend(client_indices[largest_client][:needed])
                client_indices[largest_client] = client_indices[largest_client][needed:]

    return client_indices

def get_dataloaders(dataset_name, alpha, num_clients, batch_size=64, min_size=10):
    full_trainset = get_dataset(dataset_name, train=True)
    testset = get_dataset(dataset_name, train=False)

    client_idx_lists = dirichlet_partition(full_trainset, num_clients=num_clients, alpha=alpha, min_size=min_size)

    client_dataloaders = [
        DataLoader(Subset(full_trainset, idxs),
                   batch_size=batch_size,
                   shuffle=True,
                   num_workers=4,
                   pin_memory=True)
        for idxs in client_idx_lists
    ]
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)
    return client_dataloaders, test_loader
