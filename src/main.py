import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from resnet import ResNet
from rich.progress import track
from torch.utils.data import DataLoader

BATCH_SIZE = 6400
LEARNING_RATE = 0.0002
NUM_EPOCHS = 100

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ## load dataset
    print("Loading dataset...")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    cifar10_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    cifar10_test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        cifar10_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    test_loader = DataLoader(
        cifar10_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    print("Done!")

    ## initialize model
    print("Compiling model...")
    resnet = ResNet(base_dim=64).to(device)
    print("Done!")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=LEARNING_RATE)

    print("Training model...")
    loss = 0.0
    acc = 0.0
    losses = []
    for i in range(NUM_EPOCHS):
        for j, batch in track(
            enumerate(train_loader),
            total=len(train_loader),
            description=f"Epoch {i}/{NUM_EPOCHS}, loss: {loss:.4f}, acc: {acc:.4f}",
        ):
            data, target = batch
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = resnet(data)
            loss = criterion(output, target)
            acc = (output.argmax(1) == target).float().mean()
            loss.backward()
            optimizer.step()

        if i % 10 == 0:
            print(f"Epoch: {i} | Loss: {loss}")
            losses.append(loss.cpu().detach().numpy())
    print(f"Final Loss: {loss}, Final Accuracy: {acc}")
