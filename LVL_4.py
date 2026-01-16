
print("\n LEVEL 4: ENSEMBLE")

print("Loading Level 2 Model")
model_A = torchvision.models.resnet18(pretrained=False)
model_A.fc = nn.Linear(512, 102)
model_A.load_state_dict(torch.load("level2_model.pth"))
model_A = model_A.to(device)
model_A.eval()


print("Loading Level 3 Model...")
model_B = MyCustomModel()
model_B.load_state_dict(torch.load("level3_custom_model.pth"))
model_B = model_B.to(device)
model_B.eval()


print("Running prediction on Test Set...")
correct_guesses = 0
total_images = 0


with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)


        outputs_A = model_A(images)


        outputs_B = model_B(images)


        final_opinion = (outputs_A + outputs_B) / 2.0


        _, predicted_class = torch.max(final_opinion, 1)

        total_images += labels.size(0)
        correct_guesses += (predicted_class == labels).sum().item()


final_accuracy = 100 * correct_guesses / total_images


print(f" FINAL ENSEMBLE ACCURACY: {final_accuracy:.2f}%")
