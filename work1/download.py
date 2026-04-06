import torchvision.datasets as datasets

train_dataset = datasets.FashionMNIST(
    root="./FashionMNIST",
    train=True,
    download=True
)

test_dataset = datasets.FashionMNIST(
    root="./FashionMNIST",
    train=False,
    download=True
)

print("Download finished")