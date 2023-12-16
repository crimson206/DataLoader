import torch
import torchvision
import torchvision.transforms as transforms

def load_mnist(root="./data", batch_size=64, download=False, resize_to=32, dataset_type='mnist'):
    transform = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    if dataset_type == 'mnist':
        dataset = torchvision.datasets.MNIST
    elif dataset_type == 'fashion':
        dataset = torchvision.datasets.FashionMNIST
    else:
        raise ValueError("Invalid dataset type. Choose 'mnist' or 'fashion'.")

    # Download and load the training data
    trainset = dataset(root, download=download, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = dataset(root, download=download, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainset, trainloader, testset, testloader

def load_cifar(root="./data", batch_size=64, download=False, num_classes=10):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # CIFAR images are in color
    ])

    # Choose the dataset based on the number of classes
    if num_classes == 10:
        dataset = torchvision.datasets.CIFAR10
    elif num_classes == 100:
        dataset = torchvision.datasets.CIFAR100
    else:
        raise ValueError("num_classes must be 10 or 100")

    # Download and load the training data
    trainset = dataset(root, download=download, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Download and load the test data
    testset = dataset(root, download=download, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainset, trainloader, testset, testloader
