# Trial 123

import matplotlib.pyplot as plt
import numpy as np


dataiter = iter(test_loader)
images, labels = next(dataiter)


images = images[:6]
labels = labels[:6]


images = images.to(device)


output_A = model_A(images)
output_B = model_B(images)


avg_output = (output_A + output_B) / 2.0


_, predicted_indices = torch.max(avg_output, 1)


def imshow(img):
    img = img.cpu().numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    plt.axis('off')


print("MODEL PREDICTIONS")
print("Top: True Label, Bottom: Model Prediction\n")

plt.figure(figsize=(15, 8))
for i in range(6):
    ax = plt.subplot(2, 3, i + 1)


    imshow(images[i])


    true_name = f"True: {labels[i].item()}"
    pred_name = f"Pred: {predicted_indices[i].item()}"


    color = 'green' if labels[i] == predicted_indices[i] else 'red'
    ax.set_title(f"{true_name}\n{pred_name}", color=color, fontsize=12)

plt.show()
