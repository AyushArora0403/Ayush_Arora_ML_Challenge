# Ayush_Arora_ML_Challenge

Machine Learning Challenge Submission
Name: Ayush Arora 
Date: 16-01-2026
Dataset: Option 3 (Flowers-102)
Project Overview:
This is my submission for the Terafac Machine Learning Hiring Challenge. I chose the Oxford Flowers-102 dataset. The goal was to build a model that can recognize 102 different types of flowers. I completed Levels 1, 2, 3, and 4.
Google Colab Link
Here is the code for all levels. The outputs are visible in the notebook.
Link: https://colab.research.google.com/drive/1I9aYrcVJuZrsNf-ikGjqQMXk4zqhkc81?usp=sharing
Dataset Setup
The Flowers-102 dataset has a weird default split where the training set is very small. The instructions said we must use an 80% Train, 10% Validation, and 10% Test split.
To fix this, I downloaded all the original parts of the dataset (train, val, test) and combined them into one big list. Then I used code to randomly split them again:
 * Training: 80% of images
 * Validation: 10% of images
 * Test: 10% of images
I also resized all images to 224x224 pixels because the original images were different sizes and the model needs them to be the same.
Level 1: Baseline Model
Goal: Build a basic classifier.
Model Used: ResNet18 (Pre-trained)
Approach:
I used the ResNet18 model which is already trained on ImageNet. Since ImageNet has things like dogs and cars, and I am training on flowers, I decided to unfreeze the whole model and train it again with a small learning rate (0.0001).
Results:
The model worked well and got good accuracy quickly. I saved the model as level1_model.pth.
Level 2: Intermediate Techniques
Goal: Improve the model using better techniques.
Approach:
I noticed the model might start memorizing the training images (overfitting). To stop this, I added two things:
 * Data Augmentation: I added random rotations (30 degrees) and horizontal flips to the images. This makes it harder for the model and helps it learn better.
 * Scheduler: I used a "StepLR" scheduler. This lowers the learning rate every 3 epochs. It helps the model settle down when it gets close to the best answer.
 * Regularization: I added Weight Decay in the optimizer.
Results:
The training was more stable. The validation accuracy was better than Level 1.
Level 3: Advanced Architecture
Goal: Design a custom architecture.
Approach:
Instead of just using a normal model, I created my own class called MyCustomModel.
 * Backbone: I used ResNet34 because it is bigger than ResNet18.
 * Custom Head: I removed the last layer of ResNet34 and added my own layers.
 * Layers Added: I added a Hidden Layer (512 units), a Batch Normalization layer, a ReLU activation, and a Dropout layer (50%).
Design Decision:
I added Dropout because it randomly turns off neurons. This forces the model to not rely on just one feature, making it stronger on new data.
Level 4: Expert Techniques
Goal: Use ensemble learning to get the best score.
Approach:
I used a technique called "Ensemble Voting." I loaded the trained model from Level 2 and the custom model from Level 3.
 * I gave the test images to both models.
 * I took the predictions from Model 2 and Model 3 and averaged them.
 * This is like asking two different people for the answer and taking the middle ground.
Results:
This gave the highest accuracy because if one model makes a mistake, the other one usually corrects it.
Failure Cases & Limitations
 * Sometimes the model confuses flowers that look very similar, like two different types of roses.
 * If the image is very blurry, the model struggles.
 * I only trained for a few epochs (5-10) to save time. If I trained for 50 epochs, it would probably be better.
