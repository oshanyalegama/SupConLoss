import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import torch
import os


class BiomarkerDatasetAttributes_MultiLabel(data.Dataset):
    def __init__(self,df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.img_dir + self.df.iloc[idx,0]
        image = Image.open(path).convert("L")
        image = np.array(image)
        image = Image.fromarray(image)
        image = self.transforms(image)
        B1 = self.df.iloc[idx,1]
        B2 = self.df.iloc[idx,2]
        B3 = self.df.iloc[idx,3]
        B4 = self.df.iloc[idx,4]
        B5 = self.df.iloc[idx,5]
        B6 = self.df.iloc[idx,6]
        
        bio_tensor = torch.tensor([B1, B2, B3, B4, B5, B6])
        return image, bio_tensor
