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

## ResNet50 Fine-Tuned for Dog vs Cat Classification

**Overview:**  
This is a fine-tuned **ResNet50** model, designed specifically for classifying images of **dogs** and **cats**. It leverages transfer learning to achieve high accuracy with less training data.

**Performance:**  
- Achieved **99.5% accuracy** on the validation set.  
- Works well on diverse dog and cat breeds.

**How to Use:**  

1. **Load the model in PyTorch**
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
img_tensor = preprocess(img).unsqueeze(0)  # add batch dimension
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    print("Predicted class:", "Dog" if predicted.item() == 1 else "Cat")

