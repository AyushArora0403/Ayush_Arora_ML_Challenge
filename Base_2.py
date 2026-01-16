
def train_my_model(model, name, epochs=5, lr=0.0001, weight_decay=0):
    print(f"\n Starting training for: {name}")


    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    train_accuracies = []
    val_accuracies = []


    for epoch in range(epochs):
        start_time = time.time()


        model.train()
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)


            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total


        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total


        time_taken = time.time() - start_time
        print(f"   Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Time: {time_taken:.0f}s")


        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)


    torch.save(model.state_dict(), f"{name}.pth")
    print(f" Saved model to {name}.pth")


    plt.figure(figsize=(8, 4))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title(f"Accuracy Curve: {name}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model
