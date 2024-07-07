import torch
from torch.utils.data import Dataset
from typing import List, Optional
from pathlib import Path
from PIL import Image
from torchvision import transforms


class MyDataPoint:
    def __init__(self):
        self.feat: Optional[torch.Tensor] = None
        self.label: Optional[int] = None


class MyToken(MyDataPoint):
    def __init__(self, text, label):
        super().__init__()
        self.text: str = text
        self.label = label

class MyToken_Caption(MyDataPoint):
    def __init__(self, text, label):
        super().__init__()
        self.text: str = text

class MySentence(MyDataPoint):
    def __init__(self, tokens: List[MyToken] = None, text: str = None):
        super().__init__()
        self.tokens: List[MyToken] = tokens if tokens is not None else []
        self.text = text

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index: int):
        return self.tokens[index]

    def __iter__(self):
        return iter(self.tokens)

    def __str__(self):
        return self.text if self.text else ' '.join([token.text for token in self.tokens])


class MyImage(MyDataPoint):
    def __init__(self, file_name: str):
        super().__init__()
        self.file_name: str = file_name
        self.data: Image = None


# class MyPair(MyDataPoint):
#     def __init__(self, sentence, image, label=-1):
#         super().__init__()
#         self.sentence: MySentence = sentence
#         self.image: MyImage = image
#         self.label = label

class MyPair(MyDataPoint):
    def __init__(self, sentence, image, caption, gpt1, gpt2): # [T, I, C, G]
        super().__init__()
        self.sentence: MySentence = sentence
        self.caption: MySentence = caption
        self.gpt1: MySentence = gpt1
        self.gpt2: MySentence = gpt2
        self.image: MyImage = image

class MyPair_Caption(MyDataPoint):
    def __init__(self, sentence, caption):
        super().__init__()
        self.sentence: MySentence = sentence
        self.caption: MySentence = caption

class MyPair_gpt1(MyDataPoint):
    def __init__(self, sentence, caption, gpt1):
        super().__init__()
        self.sentence: MySentence = sentence
        self.caption: MySentence = caption
        self.gpt1: MySentence = gpt1

class MyPair_gpt2(MyDataPoint):
    def __init__(self, sentence, caption, gpt1, gpt2):
        super().__init__()
        self.sentence: MySentence = sentence
        self.caption: MySentence = caption
        self.gpt1: MySentence = gpt1
        self.gpt2: MySentence = gpt2


class MyDataset(Dataset):
    def __init__(self, pairs: List[MyPair], path_to_images: Path, load_image: bool = True):
        self.pairs: List[MyPair] = pairs
        self.path_to_images = path_to_images
        self.load_image = load_image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index: int):
        pair = self.pairs[index]

        if self.load_image:
            image = pair.image

            if image.data is not None or image.feat is not None:
                return pair

            path_to_image = self.path_to_images / image.file_name
            image.data = Image.open(path_to_image).convert('RGB')
            image.data = self.transform(image.data)

        return pair

class MyDataset_Caption(Dataset):
    def __init__(self, pairs: List[MyPair], path_to_images: Path, path_to_caption: Path, load_image: bool = True):
        self.pairs: List[MyPair] = pairs
        self.path_to_images = path_to_images
        self.load_image = load_image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index: int):
        pair = self.pairs[index]
        return pair

class MyDataset_gpt1(Dataset):
    def __init__(self, pairs: List[MyPair], path_to_images: Path, path_to_caption: Path, load_image: bool = True):
        self.pairs: List[MyPair] = pairs
        self.path_to_images = path_to_images
        self.load_image = load_image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index: int):
        pair = self.pairs[index]
        return pair

class MyDataset_gpt2(Dataset):
    def __init__(self, pairs: List[MyPair], path_to_images: Path, path_to_caption: Path, load_image: bool = True):
        self.pairs: List[MyPair] = pairs
        self.path_to_images = path_to_images
        self.load_image = load_image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index: int):
        pair = self.pairs[index]
        return pair

class MyCorpus:
    def __init__(self, train=None, dev=None, test=None):
        self.train: MyDataset = train
        self.dev: MyDataset = dev
        self.test: MyDataset = test

class MyCorpus_Caption:
    def __init__(self, train=None, dev=None, test=None):
        self.train: MyDataset_Caption = train
        self.dev: MyDataset_Caption = dev
        self.test: MyDataset_Caption = test

class MyCorpus_gpt1:
    def __init__(self, train=None, dev=None, test=None):
        self.train: MyDataset_gpt1 = train
        self.dev: MyDataset_gpt1 = dev
        self.test: MyDataset_gpt1 = test

class MyCorpus_gpt2:
    def __init__(self, train=None, dev=None, test=None):
        self.train: MyDataset_gpt2 = train
        self.dev: MyDataset_gpt2 = dev
        self.test: MyDataset_gpt2 = test
