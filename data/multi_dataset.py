"""
Data loader for combining multiple change detection datasets.
"""

import os
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
from torchvision import transforms

class ChangeDetectionDataset(Dataset):
    """Change detection dataset loader"""

    def __init__(self, root_dir, split='train', transform=None, dataset_name='LEVIR-CD'):
        self.root_dir = root_dir
        self.split = split
        self.dataset_name = dataset_name
        self.transform = transform or self._default_transform()

        # Paths
        self.img_a_dir = os.path.join(root_dir, split, 'A')
        self.img_b_dir = os.path.join(root_dir, split, 'B')
        self.label_dir = os.path.join(root_dir, split, 'label')

        # Get image names
        self.image_names = sorted([
            f for f in os.listdir(self.img_a_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        print(f"Loaded {dataset_name} {split}: {len(self.image_names)} pairs")

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]

        # Load images
        img_a = Image.open(os.path.join(self.img_a_dir, name)).convert('RGB')
        img_b = Image.open(os.path.join(self.img_b_dir, name)).convert('RGB')

        # Transform
        img_a_t = self.transform(img_a)
        img_b_t = self.transform(img_b)

        return {
            'image_a': img_a_t,
            'image_b': img_b_t,
            'image_id': f"{self.dataset_name}_{name}",
            'dataset': self.dataset_name
        }


def create_multi_dataset(datasets_to_use=['LEVIR-CD'], split='train', transform=None):
    """
    Combined dataset from multiple sources.

    Args:
        datasets_to_use: List of dataset names
        split: 'train', 'val', or 'test'
        transform: data transformations

    Returns:
        Combined dataset
    """
    dataset_paths = {
        'LEVIR-CD': 'data/raw/LEVIR-CD',
        'S2Looking': 'data/raw/S2Looking'
    }

    individual_datasets = []

    for dataset_name in datasets_to_use:
        dataset_path = dataset_paths.get(dataset_name)

        if dataset_path and os.path.exists(dataset_path):
            ds = ChangeDetectionDataset(
                dataset_path,
                split=split,
                transform=transform,
                dataset_name=dataset_name
            )
            individual_datasets.append(ds)
        else:
            print(f"{dataset_name} not found at {dataset_path}, skipping")

    if len(individual_datasets) == 0:
        raise ValueError("No datasets found!")

    if len(individual_datasets) == 1:
        return individual_datasets[0]

    # Combine datasets
    combined = ConcatDataset(individual_datasets)
    print(f"\nCombined dataset: {len(combined)} total pairs")

    return combined


if __name__ == '__main__':
    
    dataset = create_multi_dataset(
        datasets_to_use=['LEVIR-CD', 'S2Looking'],
        split='train'
    )
    print(f"Dataset size: {len(dataset)}")

    # Test loading
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image A shape: {sample['image_a'].shape}")
    print(f"Dataset source: {sample['dataset']}")
