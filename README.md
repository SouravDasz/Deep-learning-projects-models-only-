# Deep Learning Models Hub

Welcome to my collection of deep learning models! 🚀  
Here, I upload all the models I have created and fine-tuned for various tasks. Each model is clearly described so you can understand and use it directly in your projects.

---

## Models Overview

Each model in this repository is described by:

- **Name** – The official model architecture and task.
- **Performance** – Accuracy or other key metrics.
- **Usage** – How to load and use it in your own projects.

---

# 1️⃣ ResNet50 Fine-Tuned for Dog vs Cat Classification

### Overview
This is a fine-tuned **ResNet50** model designed for classifying images of **dogs** and **cats**.  
It uses **transfer learning**, allowing the model to achieve high accuracy even with a relatively smaller dataset.

### Performance
- Achieved **99.5% accuracy** on the validation set.
- Works well across different dog and cat breeds.

### How to Use

#### Load the model in PyTorch

```python
import torch
from torchvision import models

model = models.resnet50(pretrained=False)
model.load_state_dict(torch.load("resnet50_dog_cat.pth"))
model.eval()

from torchvision import transforms
from PIL import Image

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img = Image.open("your_image.jpg")
img_tensor = preprocess(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)

    print("Predicted class:", "Dog" if predicted.item() == 1 else "Cat")
```





# 2️⃣ Question Answering Model (RNN Based)

## Overview
This project implements a **Question Answering model using Recurrent Neural Networks (RNN)**.  
The model learns relationships between **questions and answers** and predicts the correct answer based on the input question.

It uses **word embeddings** and **sequence modeling** to understand the context of the question.

---

## Model Architecture

- **Embedding Layer** – Converts word indices into dense vectors.
- **RNN Layer** – Captures sequential dependencies in the question.
- **Fully Connected Layer** – Predicts the most probable answer from the vocabulary.

---

## Key Features

- Custom **vocabulary and tokenization pipeline**
- Implemented entirely in **PyTorch**
- Uses **CrossEntropyLoss** for multi-class answer prediction
Implemented completely in PyTorch

Uses CrossEntropyLoss for multi-class answer prediction

Example Model Structure
