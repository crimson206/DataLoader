import torch
import torchvision
import torchvision.transforms as transforms

def load_mnist(root="./data", batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Download and load the training data
    trainset = torchvision.datasets.MNIST(root, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.MNIST(root, download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainset, trainloader, testset, testloader


transform = transforms.Compose([
    transforms.ToTensor(),
    # Add other transformations here if needed
    # Example: transforms.Resize((224, 224)),
    # Example: transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])