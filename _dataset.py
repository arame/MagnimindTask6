from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

class dataset(Dataset):
    def __init__(self, image_paths, dict_classes, IMAGE_SIZE, logging):
        self.image_paths = image_paths
        self.dict_classes = dict_classes
        self.logging = logging
        self.IMAGE_SIZE = IMAGE_SIZE
        
    #dataset length
    def __len__(self):
        return len(self.image_paths)
  
    #load an one of images
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.IMAGE_SIZE),
            transforms.RandomResizedCrop(self.IMAGE_SIZE[0])
        ])
        img_tensor = transform(img)
        _key = Path(img_path).parts[3]  
        label = self.dict_classes[_key]
        return img_tensor, label