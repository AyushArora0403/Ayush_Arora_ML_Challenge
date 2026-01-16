

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")


my_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Downloading dataset...")
set1 = torchvision.datasets.Flowers102(root='./data', split='train', download=True, transform=my_transform)
set2 = torchvision.datasets.Flowers102(root='./data', split='val', download=True, transform=my_transform)
set3 = torchvision.datasets.Flowers102(root='./data', split='test', download=True, transform=my_transform)


full_dataset = ConcatDataset([set1, set2, set3])
total_images = len(full_dataset)
print(f"Total images found: {total_images}")


train_count = int(0.8 * total_images)
val_count = int(0.1 * total_images)
test_count = total_images - train_count - val_count

train_data, val_data, test_data = random_split(full_dataset, [train_count, val_count, test_count])


train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print("Loading ResNet18 model...")
model = torchvision.models.resnet18(pretrained=True)


number_of_inputs = model.fc.in_features
model.fc = nn.Linear(number_of_inputs, 102)


model = model.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)


train_losses = []
val_accuracies = []


epochs = 5

for epoch in range(epochs):
    print(f"\nStarting Epoch {epoch + 1}/{epochs}...")


    model.train()
    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(outputs, labels)


        loss.backward()


        optimizer.step()

        running_loss += loss.item()


    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"  > Average Training Loss: {avg_loss:.4f}")


    model.eval()
    correct_guesses = 0
    total_guesses = 0


    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)


            _, predicted_class = torch.max(outputs, 1)

            total_guesses += labels.size(0)
            correct_guesses += (predicted_class == labels).sum().item()

    accuracy = 100 * correct_guesses / total_guesses
    val_accuracies.append(accuracy)
    print(f"  > Validation Accuracy: {accuracy:.2f}%")


print("\nTraining Finished!")
torch.save(model.state_dict(), "level1_beginner_model.pth")
print("Model saved as 'level1_beginner_model.pth'")


plt.plot(val_accuracies)
plt.title("My Model Accuracy over Time")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.show()
