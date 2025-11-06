from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from PIL import Image

class ESC50(Dataset):

    def __init__(self, path):
        files = Path(path).glob("*.png")
        self.items = [(f, int(f.name.split("-")[-1].replace(".png", ""))) for f in files]
        self.length = len(self.items)
        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img_file , label = self.items[index]
        img_tensor = self.transform(Image.open(img_file))
        return img_tensor, label

    def __len__(self):
        return self.length