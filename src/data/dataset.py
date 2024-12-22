import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class HybridConvViTDataset(Dataset):
    def __init__(self, root_dir, dataset_type, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            dataset_type (string): Type of dataset to load (e.g., 'train', 'val').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, dataset_type)
        self.transform = transform
        self.grayscale_images = []
        self.color_images = []

        # Load images from the grayscale and color directories
        grayscale_dir = os.path.join(self.root_dir, 'grayscale')
        color_dir = os.path.join(self.root_dir, 'original')

        for img_name in os.listdir(grayscale_dir):
            if img_name.endswith(('jpg', 'jpeg', 'png')):
                grayscale_path = os.path.join(grayscale_dir, img_name)
                color_path = os.path.join(color_dir, img_name)

                if os.path.exists(color_path):
                    self.grayscale_images.append(grayscale_path)
                    self.color_images.append(color_path)

    def __len__(self):
        return len(self.grayscale_images)

    def __getitem__(self, idx):
        grayscale_img = Image.open(self.grayscale_images[idx]).convert("L")
        color_img = Image.open(self.color_images[idx]).convert("RGB")

        if self.transform:
            grayscale_img = self.transform(grayscale_img)
            color_img = self.transform(color_img)

        return grayscale_img, color_img