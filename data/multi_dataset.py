import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultiTemporalDataset(Dataset):

    def __init__(self, split='train', datasets=['levir-mci', 's2looking'], data_root='./data/raw'):
        self.split = split
        self.data_root = Path(data_root)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
        ])

        self.samples = []

        for dataset_name in datasets:
            if dataset_name in ['levir-cc', 'levir-mci']:
                self._load_levir_cc()
            elif dataset_name == 's2looking':
                self._load_s2looking()

    def _load_levir_cc(self):
        dataset_root = self.data_root / 'Levir-MCI-dataset'
        captions_file = dataset_root / 'LevirCCcaptions.json'

        with open(captions_file, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'images' in data:
            images_list = data['images']
        else:
            return

        for img_data in images_list:
            if img_data['split'] != self.split:
                continue

            filename = img_data['filename']
            img_id = filename.replace('.png', '')

            for sent_obj in img_data['sentences']:
                caption = sent_obj['raw'].strip()

                self.samples.append({
                    'dataset': 'levir-mci',
                    'img_id': img_id,
                    'image_a': dataset_root / 'images' / self.split / 'A' / filename,
                    'image_b': dataset_root / 'images' / self.split / 'B' / filename,
                    'caption': caption
                })

    def _load_s2looking(self):
        dataset_root = self.data_root / 'S2Looking-dataset'
        captions_file = dataset_root / 'S2LookingCaptions.json'

        with open(captions_file, 'r') as f:
            captions = json.load(f)

        for img_id, data in captions.items():
            if data['split'] != self.split:
                continue

            for caption in data['sentences']:
                self.samples.append({
                    'dataset': 's2looking',
                    'img_id': img_id,
                    'image_a': dataset_root / 'images' / self.split / 'A' / data['filename_a'],
                    'image_b': dataset_root / 'images' / self.split / 'B' / data['filename_b'],
                    'caption': caption
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_a = Image.open(sample['image_a']).convert('RGB')
        img_b = Image.open(sample['image_b']).convert('RGB')

        img_a = self.transform(img_a)
        img_b = self.transform(img_b)

        return {
            'image_a': img_a,
            'image_b': img_b,
            'caption': sample['caption'],
            'dataset': sample['dataset']
        }
