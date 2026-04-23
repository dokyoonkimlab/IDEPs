import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


class ImageDataset(Dataset):
    def __init__(self, image_folder, level_of_interest, transform=None):
        self.image_paths = sorted(
            glob.glob(os.path.join(image_folder, f"bev_image_eid-*_zoom-{level_of_interest}.png"))
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        eid = image_path.split("_eid-")[1].split("_")[0]

        if self.transform:
            image = self.transform(image)

        return image, eid


def define_transform(input_size):
    transform = transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform


class FeatureExtractor(nn.Module):
    def __init__(self, ckpt_path):
        super(FeatureExtractor, self).__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.backbone.load_state_dict(
            torch.load(ckpt_path, weights_only=True)["teacher"], strict=False
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)

        return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from images using a trained DINOv2 model.")
    parser.add_argument("--level_of_interest", type=int, choices=[15, 17], required=True,
                        help="Zoom level of interest for processing.")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size.")
    parser.add_argument("--feat_dim", type=int, default=768, help="Feature dimension size.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing BEV images.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--output_file", type=str, required=True, help="Output .npz file path.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of dataloader workers.")
    args = parser.parse_args()

    transform = define_transform(args.input_size)
    dataset = ImageDataset(args.image_dir, transform=transform, level_of_interest=args.level_of_interest)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = FeatureExtractor(args.checkpoint)
    model.to("cuda")

    feats, ids = np.empty((0, args.feat_dim)), np.empty((0, 1))
    for images, eids in tqdm(dataloader):
        images = images.to("cuda")
        features = model(images)
        feats = np.concatenate([feats, features.cpu().numpy()], axis=0)
        ids = np.concatenate([ids, np.array(eids).reshape(-1, 1)], axis=0)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.savez_compressed(args.output_file, ids=ids, feats=feats)
    print(f"Features saved to {args.output_file} | shape: {feats.shape}")
