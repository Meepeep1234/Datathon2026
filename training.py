import timm
import torch
import torchvision
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image




device = torch.device("cpu")

model = timm.create_model('resnet50', pretrained=False, num_classes=102)
model.to(device)

training_file = pd.read_csv("train.csv")
img = Image.open("train/images/train_00000.jpg")
#label = training_file.loc[0,"label"] testing
#print(label) testing


#transformations so we could have more training data and instead of memorizing images it lets it look at images at "different angles"
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class FlowerDataset(Dataset):
    def __init__(self, df,image_dir,transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.image_dir}/{row["filename"]}"
        image = Image.open(img_path).convert('RGB')
        label = row["label"] - 1
        if self.transform:
            image = self.transform(image)
        return image, label

#data loaders
train_dataset = FlowerDataset(training_file,"train/images",transform=train_tfms)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


#now we work on the loss/reward optimize functions
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.00001)

#Training loop stuffs

for epoch in range(100):
    for images, labels in train_loader:
        imgs, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        print("Loss:",loss.item())
        optimizer.step()



torch.save(model.state_dict(), "model/model10.pth")
