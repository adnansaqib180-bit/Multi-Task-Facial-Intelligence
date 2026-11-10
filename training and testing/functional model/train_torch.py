import torch
from torch.nn import *
from torchvision.transforms import *
from train import train_df_split,test_df_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class AgeGenderDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        self.transform = Compose([
            Resize((224, 224)),
            ToTensor()
        ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        path = self.df.iloc[index]['img_path']
        age = self.df.iloc[index]['age']
        gender = self.df.iloc[index]['gender']
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        age = torch.tensor(age, dtype=torch.float32)
        gender = torch.tensor(gender, dtype=torch.float32)
        return image, {
            'age_output': age,
            'gender_output': gender
        }
train_dataset = AgeGenderDataset(train_df_split)
test_dataset = AgeGenderDataset(test_df_split)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)
class MultiOutputCNN(Module):
    def __init__(self,):
        super().__init__()
        self.covolutions = Sequential(
            RandomHorizontalFlip(),
            RandomRotation(0.1),
            Resize((224,224,3)),
            Conv2d(64, 3,  padding='same'),
            MaxPool2d((2, 2),2),
            Conv2d(128, 3,padding='same'),
            Conv2d(128, 3,padding='same'),
            MaxPool2d((2, 2),2),
            Conv2d(256, 3, padding='same'),
            Conv2d(256, 3, padding='same'),
            Conv2d(256, 3,padding='same'),
            MaxPool2d((2, 2),2),
            Conv2d(512, 3,padding='same'),
            Conv2d(512, 3,padding='same'),
            Conv2d(512, 3,padding='same'),
            MaxPool2d((2, 2),2),
            Conv2d(512, 3,padding='same'),
            Conv2d(512, 3,padding='same'),
            Conv2d(512, 3,padding='same'),
            MaxPool2d((2, 2),2), 
        )
        self.age_ann  = Sequential(
            Flatten(),
            LazyLinear(128),
            ReLU(),
            Linear(128,1)
        )
        self.gender_ann = Sequential(
            Flatten(),
            LazyLinear(128),
            ReLU(),
            Linear(128,1),
            Sigmoid()
        )
    def forward(self,x):
        inputs = self.covolutions(x),
        age = self.age_ann(inputs)
        gender = self.gender_ann(inputs)
        return age,gender 
model = MultiOutputCNN()   
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
criterian_age =MSELoss()
criterian_gender = BCELoss()

epochs = 100
for i in range (epochs):
    train_age_loss = 0.0
    train_gender_loss = 0.0
    model.train()
    for image, labels in train_loader:
        age = labels['age']
        gender = labels['gender']
        age_pred,gender_pred = model(image)
        age_loss = criterian_age(age_pred,age)
        train_age_loss+=age_loss.item()
        gender_loss = criterian_gender(gender_pred,gender)
        train_gender_loss+=gender_loss.item()
        loss = age_loss+gender_loss
        loss.backward()
        optimizer.step()
    print(f'epochs ==> {i+1}, gender loss ==>{train_gender_loss}, age loss ==> {train_age_loss}')
    model.eval()
    test_age_loss = 0.0
    test_gender_loss = 0.0
    for image, labels in test_loader:
        age = labels['age']
        gender = labels['gender']
        age_pred,gender_pred = model(image)
        age_loss = criterian_age(age_pred,age)
        test_age_loss+=age_loss.item()
        gender_loss = criterian_gender(gender_pred,gender)
        total_gender_loss+=gender_loss.item()
    print(f'epochs ==> {i+1}, gender loss ==>{train_gender_loss}, age loss ==> {train_age_loss}')
