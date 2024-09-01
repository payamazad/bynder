import os.path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class SingleLabelData:
    def __init__(self, label_name: str, images_dir: str, metadata_file_path: str, bath_size: int):
        self.label_name = label_name
        self.images_dir = images_dir
        self.metadata = pd.read_csv(metadata_file_path)
        self.label_map = self.get_label_map()
        self.bath_size = bath_size
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.create_data_loader()

    def get_label_map(self) -> dict:
        return {label_desc: label for label, label_desc in enumerate(self.metadata[self.label_name].unique())}

    def create_data_loader(self):
        dataset = ImageDataset(metadata=self.metadata,
                               img_dir=self.images_dir,
                               label_name=self.label_name,
                               label_map=self.label_map,
                               )

        generator = torch.Generator().manual_seed(42)   # setting the seed with guarantee the same split all the time
        train_data, val_data, test_data = torch.utils.data.random_split(dataset, [0.7, 0.1, 0.2], generator)

        train_dataloader = DataLoader(train_data, batch_size=self.bath_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=self.bath_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=self.bath_size, shuffle=True)

        return train_dataloader, val_dataloader, test_dataloader


class ImageDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, img_dir: str, label_name: str, label_map: dict):
        self.metadata = metadata
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert PIL image to tensor
        ])
        self.label_map = label_map
        self.label_name = label_name

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, str(self.metadata.loc[idx, "id"]) + ".jpg"))
        label = self.label_map[self.metadata.loc[idx, self.label_name]]

        if self.transform:
            image = self.transform(image)

        return image, label
