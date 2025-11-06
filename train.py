import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from utils import ESC50
from models.base_model import AudioNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ESC50("data/train")
valid_dataset = ESC50("data/valid")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)

audionet = AudioNet()
audionet = audionet.to(device)
features = list(audionet.backbone.features.children())

for layer in features[:len(features)//2]:
    for param in layer.parameters():
        param.requires_grad = False

mid_and_deep_layers = features[len(features)//2:]

mid_and_deep_params = []
for layer in mid_and_deep_layers:
    mid_and_deep_params += list(layer.parameters())

def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


optimizer = optim.Adam([
    {'params': mid_and_deep_params, 'lr': 1e-6},
    {'params': audionet.backbone.classifier.parameters(), 'lr': 1e-4} 
])
loss_fn = nn.CrossEntropyLoss()
num_epochs = 30

for epoch in range(num_epochs):
    audionet.train()
    train_loss = 0

    for img, target in train_loader:
        img = img.to(device)
        target = target.to(device)

        output = audionet(img)
        loss = loss_fn(output, target)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = calculate_accuracy(audionet, train_loader, device)
    valid_acc = calculate_accuracy(audionet, valid_loader, device)

    print(f"Epoch {epoch}")
    print(f"    Loss: {train_loss/len(train_loader)}")
    print(f"    Training Accuracy: {train_acc}, Validation Accuracy: {valid_acc}")

torch.save(
    {
    'audionet_state_dict': audionet.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }
    , "audionet_checkpoint.pth")
