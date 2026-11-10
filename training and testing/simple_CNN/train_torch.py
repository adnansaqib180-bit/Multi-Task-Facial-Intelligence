import torch 
from torch.nn import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
TRAIN = '/kaggle/working/cleaned_dataset/train_dir'
TEST = '/kaggle/working/cleaned_dataset/test_dir'

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),     
    transforms.RandomHorizontalFlip(),     
    transforms.ToTensor(),         
    transforms.Normalize(                      
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
train_data = datasets.ImageFolder(root_dir =TRAIN,transform=train_transforms)
test_data = datasets.ImageFolder(root_dir =TEST)
train_loader = DataLoader(
    dataset= train_data,
    shuffle=True,
    batch_size=32,
    pin_memory=True,
    num_workers=2
)
test_loader = DataLoader(
    dataset= test_data,
    shuffle=True,
    batch_size=32,
    pin_memory=True,
    num_workers=2
)
class CNN(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = Sequential(
            Conv2d(64, kernel_size=(3,3), padding='same'),
            ReLU(),
            BatchNorm2d(),
            Conv2d(64, kernel_size=(3,3), padding='same'),
            ReLU(),
            BatchNorm2d(),
            MaxPool2d(pool_size=(2,2)),
            Dropout(),
        
            Conv2d(128, kernel_size=(3,3), padding='same'),
            BatchNorm2d(),
            ReLU(),
            Conv2d(128, kernel_size=(3,3), padding='same'),
            ReLU(),
            BatchNorm2d(),
            MaxPool2d(pool_size=(2,2)),
            Dropout(),
        
            Conv2d(256, kernel_size=(3,3), padding='same'),
            BatchNorm2d(),
            ReLU(),
            Conv2d(256, kernel_size=(3,3), padding='same'),
            ReLU(),
            BatchNorm2d(),
            MaxPool2d(pool_size=(2,2)),
            Dropout(0.4)
        )
        self.ann = Sequential(
            Flatten(),
            LazyLinear(out_features=256),
            ReLU(),
            BatchNorm1d(),
            Dropout(0.5),
            Linear(in_features=256,out_features=num_classes),
            Softmax()
        )
    def forward(self,x):
        x = self.cnn(x)
        return self.ann(x)

model = CNN(7)
criterian = CrossEntropyLoss()
optimizer  = torch.optim.Adam(model.parameters(),lr=0.001)
epochs = 100

for i in range (epochs):
    model.train()
    train_loss = 0.0
    for image , label in train_loader:
        optimizer.zero_grad()
        predictions = model(image)
        loss = criterian(predictions,label)
        loss.backkward()
        optimizer.step()
        train_loss+=loss.item()* image.size(0)
    model.eval()
    eval_loss = 0.0
    with optimizer.zero_grad():
        for image , label in test_loader:
            prediction = model(image)
            loss = criterian(prediction,label)
            eval_loss += loss.item()*image.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_val_loss = eval_loss / len(test_loader.dataset)
    
    print(f"Epoch {i+1}/{epochs} , Train Loss: {avg_train_loss:.4f} , Val Loss: {avg_val_loss:.4f}")
           