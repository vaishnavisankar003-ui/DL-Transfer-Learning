# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.
## Problem Statement and Dataset
Include the problem statement and Dataset

## Neural Network Model
Include the neural network model diagram.
<img width="1020" height="792" alt="image" src="https://github.com/user-attachments/assets/64d08f2f-1934-4b23-83bf-a288c2d06350" />

## DESIGN STEPS
### STEP 1: 
Import required libraries and define image transforms.

### STEP 2: 
Load training and testing datasets using ImageFolder.

### STEP 3: 
Visualize sample images from the dataset.

### STEP 4: 
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions

## PROGRAM
### Name: VAISHNAVI S
### Register Number: 212225230289

```python
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

model = models.vgg19(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False

num_classes = 10
model.classifier[6] = nn.Linear(
    model.classifier[6].in_features,
    num_classes
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
model = model.to(device)


print("Using Device:", device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

def train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=10):


    train_losses = []
    val_losses = []
for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:


            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = (
            running_loss /
            len(train_loader)
        )


        train_losses.append(train_loss)
        model.eval()

        validation_loss = 0.0



        with torch.no_grad():

            for images, labels in test_loader:


                images = images.to(device)

                labels = labels.to(device)



                outputs = model(images)



                loss = criterion(
                    outputs,
                    labels
                )


                validation_loss += loss.item()



        val_loss = (
            validation_loss /
            len(test_loader)
        )


        val_losses.append(val_loss)



        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Validation Loss: {val_loss:.4f}"
        )



    # Loss Graph

    plt.figure(figsize=(8,6))

    plt.plot(
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.show()



train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=10
)

def test_model(model,test_loader):

    model.eval()


    correct = 0
    total = 0


    all_preds = []
    all_labels = []



    with torch.no_grad():


        for images, labels in test_loader:


            images = images.to(device)
            labels = labels.to(device)



            outputs = model(images)



            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)


            correct += (
                predicted == labels
            ).sum().item()



            all_preds.extend(
                predicted.cpu().numpy()
            )


            all_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = correct / total


    print(
        f"Test Accuracy: {accuracy*100:.2f}%"
    )

    cm = confusion_matrix(
        all_labels,
        all_preds
    )


    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=train_dataset.classes,
        yticklabels=train_dataset.classes
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title(
        "Confusion Matrix"
    )

    plt.show()

    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=train_dataset.classes
        )
    )


test_model(
    model,
    test_loader
)

def predict_image(
        model,
        image_index,
        dataset):

    model.eval()


    image, label = dataset[image_index]

    with torch.no_grad():


        image_tensor = (
            image.unsqueeze(0)
            .to(device)
        )


        output = model(
            image_tensor
        )

        _, predicted = torch.max(
            output,
            1
        )


        predicted = predicted.item()



    class_names = dataset.classes

    img = image.permute(
        1,2,0
    )


    plt.figure(figsize=(4,4))


    plt.imshow(
        img
    )


    plt.title(
        f"Actual: {class_names[label]}\n"
        f"Predicted: {class_names[predicted]}"
    )


    plt.axis("off")

    plt.show()



    print(
        "Actual:",
        class_names[label]
    )

    print(
        "Predicted:",
        class_names[predicted]
    )




# Example Prediction

predict_image(
    model,
    image_index=55,
    dataset=test_dataset
)

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot
<img width="877" height="698" alt="image" src="https://github.com/user-attachments/assets/287499a8-cf52-4e38-8086-238d5fc78201" />

## Confusion Matrix
Include confusion matrix here
<img width="808" height="680" alt="image" src="https://github.com/user-attachments/assets/8c92ff22-9263-45ba-8c20-39ef1c70a003" />

## Classification Report
Include classification report here

<img width="546" height="228" alt="image" src="https://github.com/user-attachments/assets/09f91b80-f430-4774-874a-e10edfceac3f" />

### New Sample Data Prediction

<img width="489" height="515" alt="image" src="https://github.com/user-attachments/assets/7af6ff24-b12f-40e2-8ba7-4b7ae31b806e" />

## RESULT
Thus the python program to develop an image classification model using transfer learning with VGG19 architecture is executed successfully.
