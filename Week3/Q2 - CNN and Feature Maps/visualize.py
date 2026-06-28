import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

from model import CNNModel

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CNNModel().to(device)

model.load_state_dict(
    torch.load(
        "cnn_model.pth",
        map_location=device
    )
)

model.eval()

activations = {}

def activation_hook(name):

    def hook(module, inp, output):
        activations[name] = output.detach()

    return hook

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),
                         (0.5,0.5,0.5))
])

dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

image, _ = dataset[0]
image = image.unsqueeze(0).to(device)

model.conv_block1.register_forward_hook(activation_hook("conv_block1"))
model.conv_block2.register_forward_hook(activation_hook("conv_block2"))


with torch.no_grad():
    output = model(image)


# Plot feature maps
for layer_name, feature_maps in activations.items():

    feature_maps = feature_maps.squeeze(0)

    plt.figure(figsize=(12, 6))

    for i in range(min(8, feature_maps.shape[0])):
        plt.subplot(2, 4, i + 1)
        plt.imshow(feature_maps[i].cpu(), cmap="gray")
        plt.axis("off")

    plt.suptitle(layer_name)
    plt.show()