import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1) Dataset + DataLoader (équivalent mnist.load_data)
transform = transforms.Compose([
    transforms.ToTensor(),  # (H,W) -> (1,H,W) et /255
])

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

# 2) Modèle
class BasicConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1 -> 32
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32 -> 64
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14 -> 7x7

            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BasicConvNet().to(device)

# 3) Loss + Optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4) Entrainement
epochs = 5
for epoch in range(epochs):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print(f"epoch={epoch+1}/{epochs} batch={batch_idx}, loss={loss.item():.4f}")

# 5) Evaluation
model.eval()
correct, total = 0, 0
loss_sum = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)

        loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_loss = loss_sum / total
test_acc = correct / total

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
