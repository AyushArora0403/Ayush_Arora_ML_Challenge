print("\n LEVEL 3")


class MyCustomModel(nn.Module):
    def __init__(self):
        super(MyCustomModel, self).__init__()


        self.body = torchvision.models.resnet34(pretrained=True)


        number_of_features = self.body.fc.in_features
        self.body.fc = nn.Identity()


        self.my_head = nn.Sequential(
            nn.Linear(number_of_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 102)
        )

    def forward(self, x):

        x = self.body(x)
        x = self.my_head(x)
        return x


print("Building Custom ResNet34")
model_3 = MyCustomModel()
model_3 = model_3.to(device)


optimizer = optim.Adam(model_3.parameters(), lr=0.0001)


epochs = 10
print(f"Training Custom Model for {epochs} epochs")

for epoch in range(epochs):
    model_3.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model_3(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


torch.save(model_3.state_dict(), "level3_custom_model.pth")
print(" Level 3 Custom Model Saved.")
