import torch
from torch.nn import *
from torchvision.transforms import *
from torchvision import datasets
from torch.utils.data import DataLoader

TRAIN = '/kaggle/input/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/train'
TEST = '/kaggle/input/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/test'

train_transform = Compose(
    Resize(224,224),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
)

train_data = datasets.ImageFolder(root=TRAIN,transform=train_transform)
test_data = datasets.ImageFolder(root=TEST)

trian_loader = DataLoader(
    dataset=train_data,
    shuffle=True,
    batch_size=32,
    num_workers=2,
    pin_memory=True
)
test_loader = DataLoader(
    dataset=train_data,
    shuffle=True,
    batch_size=32,
    num_workers=2,
    pin_memory=True
)

from torchvision.models import vgg16

base_model = vgg16(pretrained= True )

for params in base_model.parameters():
    params.requires_grad = False

print('successfully freezed the top cnns ')

base_model.classifier = Sequential(
    LazyLinear(128),
    ReLU(),
    Linear(128,1),
    Sigmoid()
)

criterian = BCELoss()
optimizer = torch.optim.Adam(params=base_model.parameters(),lr = 0.0001)
epochs = 100
train_losses = []
test_losses  = []
for i in range(epochs):
    train_loss = 0.0
    base_model.train()
    for image, label in trian_loader:
        optimizer.zero_grad()
        predictions = base_model(image)
        loss = criterian(predictions,label)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*image.size(0)
    avg_loss = train_loss/len(trian_loader.dataset)
    train_losses.append(avg_loss)
    base_model.eval()
    test_losses = []
    with optimizer.no_grad():
        test_loss
        for image,label in test_loader:
            preds = base_model(image)
            loss = criterian(preds,label)
            test_loss+=loss.item()*image.size(0)
    avg_test_loss = test_loss/len(test_loader.dataset)
    test_losses.append(avg_test_loss)
    print(f'epochs: {epochs+1} --- train_loss => {avg_loss} --- test_loss => {avg_test_loss}')


import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid') 
plt.figure(figsize=(10, 6), dpi=150) 
plt.plot(range(1, epochs + 1), train_losses, color='#1f77b4', linestyle='-', linewidth=2.5, marker='o', label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, color='#ff7f0e', linestyle='--', linewidth=2.5, marker='s', label='Testing Loss')
plt.title('VGG16 Model Loss Progression Over Epochs', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epochs', fontsize=12, fontweight='semibold')
plt.ylabel('Loss Value', fontsize=12, fontweight='semibold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=11, loc='upper right')
plt.xticks(range(1, epochs + 1))
plt.tight_layout()
plt.show()
