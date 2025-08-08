import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = 'flip' in self.feature_dir

        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        self.feature_files = sorted(os.listdir(feature_dir))
        self.label_files = sorted(os.listdir(label_dir))

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir
        
        while True:
            try:        
                # for whole npys
                feature_file = self.feature_files[idx]
                label_file = self.label_files[idx]
                features = np.load(os.path.join(feature_dir, feature_file))
                labels = np.load(os.path.join(label_dir, label_file))

                # for cls dirs
                # feature_file = self.feature_files[idx]
                # cls_dir = f"cls_{idx//1300}"
                # features = np.load(os.path.join(feature_dir, cls_dir, feature_file))
                # labels = idx//1300
                break
            except Exception as e:
                print(f"Error details: {str(e)}")
                print(f"Error path: {self.feature_files[idx]}")
                idx = np.random.randint(len(self))
        if self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            features = features[:, aug_idx]
        return torch.from_numpy(features), torch.from_numpy(labels)

def build_imagenet(data_path, transform, condition_frames):
    print("imagenet: building train...")
    return ImageFolder(data_path, transform=transform)

def build_imagenet_code(code_path):
    feature_dir = f"{code_path}/imagenet256_codes"
    label_dir = f"{code_path}/imagenet256_labels"
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDataset(feature_dir, label_dir)
