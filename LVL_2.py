print("\n LEVEL 2")


aug_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




print("Setting up a fresh ResNet18 model...")
model_2 = torchvision.models.resnet18(pretrained=True)
num_inputs = model_2.fc.in_features
model_2.fc = nn.Linear(num_inputs, 102)
model_2 = model_2.to(device)


criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(model_2.parameters(), lr=0.0001, weight_decay=0.0001)


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


epochs = 8
print(f"Training for {epochs} epochs")

train_accuracies = []

for epoch in range(epochs):
    print(f"Epoch {epoch+1}...")


    model_2.train()
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model_2(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    scheduler.step()

    train_acc = 100 * correct / total
    train_accuracies.append(train_acc)
    print(f"  > Training Accuracy: {train_acc:.2f}%")


torch.save(model_2.state_dict(), "level2_model.pth")
print(" Level 2 Model Saved.")
