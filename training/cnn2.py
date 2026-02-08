import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

# -------------------------
# Model
# -------------------------
class BasicConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 -> 14x14
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

# -------------------------
# Common utils
# -------------------------
def evaluate(model, loader, device, criterion):
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return loss_sum / total, correct / total

def train_one_dataset(train_loader, test_loader, device, epochs=20, lr=1e-3, weight_decay=0.0):
    model = BasicConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    best = None

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

        # eval    chaque epoch (pour voir si   a sur-apprend)
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        print(f"  -> test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best is not None:
        model.load_state_dict(best)

    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    return test_loss, test_acc

# -------------------------
# 1) Train MNIST 
# -------------------------
def train_mnist(augment=False):
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.10, 0.10),
                scale=(0.90, 1.10),
                interpolation=InterpolationMode.BILINEAR,
                fill=0
            ),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([transforms.ToTensor()])

    transform_test = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loss, test_acc = train_one_dataset(train_loader, test_loader, device, epochs=5, lr=1e-3)

    tag = "AUG" if augment else "NOAUG"
    print(f"[MNIST-{tag}] Test loss: {test_loss:.4f}")
    print(f"[MNIST-{tag}] Test accuracy: {test_acc:.4f}")
    return test_loss, test_acc


# -------------------------
# 2) Train BDD perso (BMP 28x28) puis test sur la m  me BDD (split)
# -------------------------
class CustomDigitsDataset(Dataset):
    """
    Attend des fichiers dans data/custom_digits/ du type:
      0-0.bmp, 0-1.bmp, ..., 9-123.bmp
    Le label est pris AVANT le premier '-'.
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.files = sorted(glob.glob(os.path.join(root, "*.bmp")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"Aucun .bmp trouv   dans {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        filename = os.path.basename(path)

        # label = avant le "-"
        # ex: "3-12.bmp" -> 3
        label_str = filename.split("-")[0]
        label = int(label_str)

        # open image
        img = Image.open(path).convert("L")  # L = grayscale

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def train_personnal(augment=False):
    custom_root = "./data/custom_digits"
    if not os.path.isdir(custom_root):
        raise FileNotFoundError(f"Dossier introuvable: {custom_root}")

    # Test/val: jamais d'augmentation (juste preprocess)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),             # fond blanc
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    # Train: preprocess + (optionnel) augmentation
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.10, 0.10),
                scale=(0.90, 1.10),
                interpolation=InterpolationMode.BILINEAR,
                fill=255  # fond blanc avant inversion
            ),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        transform_train = transform_test

    # IMPORTANT: pour avoir train/test avec transforms diff  rents,
    # on charge deux datasets identiques puis on split les indices.
    full_train = CustomDigitsDataset(custom_root, transform=transform_train)
    full_test  = CustomDigitsDataset(custom_root, transform=transform_test)

    n_total = len(full_train)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train

    # indices reproductibles
    g = torch.Generator().manual_seed(42)
    train_ds, _ = random_split(full_train, [n_train, n_test], generator=g)
    _, test_ds  = random_split(full_test,  [n_train, n_test], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loss, test_acc = train_one_dataset(train_loader, test_loader, device, epochs=20, lr=1e-3)

    tag = "AUG" if augment else "NOAUG"
    print(f"[CUSTOM-{tag}] Test loss: {test_loss:.4f}")
    print(f"[CUSTOM-{tag}] Test accuracy: {test_acc:.4f}")
    return test_loss, test_acc


# -------------------------
# 3) Compare
# -------------------------
if __name__ == "__main__":
    mnist_no = train_mnist(augment=False)
    mnist_aug = train_mnist(augment=True)

    custom_no = train_personnal(augment=False)
    custom_aug = train_personnal(augment=True)

    print("\n=== COMPARAISON ===")
    print(f"MNIST  NOAUG: loss={mnist_no[0]:.4f} | acc={mnist_no[1]:.4f}")
    print(f"MNIST AUGMENT: loss={mnist_aug[0]:.4f}, acc={mnist_aug[1]:.4f}")
    print(f"CUSTOM NOAUG: loss={custom_no[0]:.4f}, acc={custom_no[1]:.4f}")
    print(f"CUSTOM AUGMENT: loss={custom_aug[0]:.4f} | acc={custom_aug[1]:.4f}")
    
    print("\nDelta (CUSTOM - MNIST):")
    print(f"NO AUG : loss: {mnist_no[0]:.4f - custom_no[0]:.4f}   ;   acc: {mnist_no[1]:.4f - custom_no[1]:.4f}")
    print(f"AUG : loss: {mnist_aug[0]:.4f - custom_aug[0]:.4f}   ;    acc: {mnist_aug[1]:.4f - custom_aug[1]:.4f}") 
