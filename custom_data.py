import os
import json
import torch
from PIL import Image


# Custom Dataset (adapted from load_data.py)
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, images_root="", transform=None, cur_type="train"):
        """
        Expects a JSON file with structure:
        {
            "DFDCP": {
                "domain_name": {
                    "train" or "test" or "val": {
                        "image_identifier": {
                            "frames": {
                                "relative/path/to/image1.jpg": {},
                                "relative/path/to/image2.jpg": {}
                            },
                            "label": "Real" or "Fake"
                        },
                        ...
                    },
                    ...
                },
                ...
            }
        }
        """
        self.images_root = images_root
        self.transform = transform
        self.cur_type = cur_type
        assert self.cur_type in [
            "train",
            "test",
            "val",
        ], "cur_type must be train, test or val"

        with open(json_file, "r") as f:
            data = json.load(f)

        dfdcp_data = data.get("DFDCP", {})
        domains = dfdcp_data.keys()

        self.image_paths = []
        self.labels = []
        for domain in domains:
            domain_data = dfdcp_data.get(domain, {})
            partition = domain_data.get(self.cur_type, {})
            for image in partition:
                image_info = partition.get(image, {})
                paths = [path.replace("\\", "/") for path in image_info.get("frames", {})]
                label = image_info.get("label", "")
                for path in paths:
                    self.image_paths.append(path)
                    self.labels.append(label)
        if len(self.image_paths) != len(self.labels):
            raise RuntimeError("Mismatch in image paths and labels")
        if not self.image_paths:
            raise RuntimeError("No image paths found.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(os.path.join(self.images_root, img_path)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Convert textual labels into integers: 1 for real, 0 for fake.
        label_str = self.labels[index].lower()
        label = 1 if "real" in label_str else 0
        return image, label
