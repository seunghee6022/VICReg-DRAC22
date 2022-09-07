import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os

from PIL import Image

challenge_default_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

class ChallengeDataset(Dataset):

    def __init__(self, root, csv_path, transform=challenge_default_transform):
        super(Dataset, self).__init__()
        self.root = root
        self.transform = transform
        self.paths = []
        for file in os.listdir(root):
            if file.endswith(".png"):
                self.paths.append(str(file))
        csv_df = pd.read_csv(csv_path, sep=',')
        if csv_df.isnull().values.any():
            print("Nan이 있습니다.")
        self.csv_df = csv_df

        
    def _loader(self, path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, index):
        csv_df =self.csv_df
        
        path = os.path.join(self.root, self.paths[index])
        image_name = self.paths[index]
        img = self._loader(path)
        target = csv_df.loc[lambda csv_df: csv_df['image name']== image_name]['DR grade'].item()

        img = self.transform(img)
        if csv_df.isnull().values.any():
            print(f"{index} - Nan이 있습니다.")
        
        return img, target

    def __len__(self):
        return len(self.paths)




