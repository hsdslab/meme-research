from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, Subset
from PIL import Image
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import pytorch_lightning as pl


pl.seed_everything(42)

class BinaryMemeDataset(Dataset):
    def __init__(self, data_df_path, transform=transforms.ToTensor()):
        self.data_df = pd.read_parquet(data_df_path)
        self.transform = transform
        self.X, self.y = self._get_X_y()
        
    def _get_X_y(self):
        X = self.data_df['path']
        y = self.data_df['label']
        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        try:
            img_path = self.X.iloc[idx]
            label = self.y.iloc[idx]
            # Open the image using PIL
            with Image.open(img_path) as img:
                # Convert to RGB if the image has an alpha channel or is not in RGB mode
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Apply any additional transforms
                if self.transform:
                    img = self.transform(img)
        except Exception as e:
            raise e
            return None, None
        
        return img, label

class MemesDataset(Dataset):
    def __init__(self, data_df_path, transform=transforms.ToTensor()):
        self.data_df = pd.read_parquet(data_df_path)
        self.transform = transform
        self.num_classes = self._get_num_classes()
        self.X, self.y = self._get_X_y()
        
    def _get_num_classes(self):
        print("Number of unique classes: ", self.data_df['template_name'].nunique())
        return self.data_df['template_name'].nunique()
    
    def _encode_templates(self):
        labelencoder = LabelEncoder()
        self.data_df['template_id'] = labelencoder.fit_transform(self.data_df['template_name'])
    
    def _get_X_y(self):
        self._encode_templates()
        X = self.data_df['path']
        y = self.data_df['template_id'].astype(float)
        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        try:
            img_path = self.X.iloc[idx]
            img = Image.open(img_path)
            label = self.y.iloc[idx]

            if self.transform:
                img = self.transform(img)
        except:
            print("Error loading image")
            return None, None
        
        return img, label
 

class MemesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_df_path, test_size=0.2):
        super().__init__()
        self.data_df_path = data_df_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.save_hyperparameters()

        # Augmentation policy for training set
        self.augmentation = transforms.Compose([
              transforms.Resize(size=(224,224)),
              transforms.RandomRotation(degrees=(0, 10)),
              transforms.ColorJitter(brightness=0.1, contrast=0.1),
              transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
              transforms.ToTensor(),
              transforms.Normalize([0.5325, 0.4980, 0.4715],[0.3409, 0.3384, 0.3465])
        ])
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose([
              transforms.Resize(size=(224,224)),
              transforms.ToTensor(),
              transforms.Normalize([0.5325, 0.4980, 0.4715],[0.3409, 0.3384, 0.3465])
        ])
        
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # build dataset
        dataset = MemesDataset(data_df_path=self.data_df_path)

        # Stratified Sampling for train and val
        train_idx, validation_idx = train_test_split(np.arange(len(dataset)),
                                                    test_size=self.test_size,
                                                    shuffle=True,
                                                    stratify=dataset.y)

        # Subset dataset for train and val
        self.train = Subset(dataset, train_idx)
        self.val = Subset(dataset, validation_idx)

        self.train.dataset.transform = self.augmentation
        self.val.dataset.transform = self.transform

        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=my_collate_fn)


    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4, collate_fn=my_collate_fn)

class KFoldMemeDataModule(pl.LightningDataModule):
    '''Reference: https://gist.github.com/ashleve/ac511f08c0d29e74566900fd3efbb3ec \n
    Example usage:
        results = []
        nums_folds = 10
        split_seed = 12345

        for k in range(nums_folds):
            datamodule = ProteinsKFoldDataModule(k=k, num_folds=num_folds, split_seed=split_seed, ...)
            datamodule.prepare_data()
            datamodule.setup()

            # here we train the model on given split...
            ...

            results.append(score)

        score = sum(results) / num_folds
    '''
    def __init__(self, batch_size:int, dataset:Dataset, k:int, num_splits:int, split_seed:int):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_splits = num_splits
        self.split_seed = split_seed
        self.save_hyperparameters()

        assert 0 <= self.k <= self.num_splits-1, "incorrect fold number"

        # Augmentation policy for training set
        self.augmentation = transforms.Compose([
              transforms.Resize(size=(224,224)),
              transforms.RandomRotation(degrees=(0, 10)),
              transforms.ColorJitter(brightness=0.1, contrast=0.1),
              transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
              transforms.ToTensor(),
              transforms.Normalize([0.5325, 0.4980, 0.4715],[0.3409, 0.3384, 0.3465])
        ])
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose([
              transforms.Resize(size=(224,224)),
              transforms.ToTensor(),
              transforms.Normalize([0.5325, 0.4980, 0.4715],[0.3409, 0.3384, 0.3465])
        ])
        
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        kf = StratifiedKFold(n_splits=self.num_splits, shuffle=True, random_state=self.split_seed)

        all_splits = list(kf.split(np.arange(len(self.dataset)), self.dataset.y))
        train_idx, validation_idx = all_splits[self.k]
        train_idx, validation_idx = train_idx.tolist(), validation_idx.tolist()

        # Subset dataset for train and val
        self.train = Subset(self.dataset, train_idx)
        self.val = Subset(self.dataset, validation_idx)

        self.train.dataset.transform = self.augmentation
        self.val.dataset.transform = self.transform

        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=my_collate_fn)


    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4, collate_fn=my_collate_fn)   

def my_collate_fn(batch):
    """ Filter out None samples """
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor()  # Empty tensors if batch is fully invalid
    return torch.utils.data.dataloader.default_collate(batch)
