import torch
from nuimages import NuImages
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO


class NuImagesCoCoDataset(Dataset):
    def __init__(self, dataset_root, dataset_version, transform=None, target_transform=None):
        self.dataset_root = dataset_root
        self.dataset_version = dataset_version
        self.transform = transform
        self.target_transform = target_transform

        self.nuImg = NuImages(dataroot=self.dataset_root,
                              version=self.dataset_version,
                              verbose=False, lazy=True)
        self.categories_list = ['animal',
                                'flat.driveable_surface',
                                'human.pedestrian.adult',
                                'human.pedestrian.child',
                                'human.pedestrian.construction_worker',
                                'human.pedestrian.personal_mobility',
                                'human.pedestrian.police_officer',
                                'human.pedestrian.stroller',
                                'human.pedestrian.wheelchair',
                                'movable_object.barrier',
                                'movable_object.debris',
                                'movable_object.pushable_pullable',
                                'movable_object.trafficcone',
                                'static_object.bicycle_rack',
                                'vehicle.bicycle',
                                'vehicle.bus.bendy',
                                'vehicle.bus.rigid',
                                'vehicle.car',
                                'vehicle.construction',
                                'vehicle.ego',
                                'vehicle.emergency.ambulance',
                                'vehicle.emergency.police',
                                'vehicle.motorcycle',
                                'vehicle.trailer',
                                'vehicle.truck']  # list(map(lambda x: x['name'], self.nuImg.category))
        self.num_classes = len(self.categories_list)
        self.categories_idx_map = {}
        for idx, category in enumerate(self.categories_list):
            self.categories_idx_map[category] = idx

    def __len__(self):
        return len(self.nuImg.sample)

    def __getitem__(self, sample_idx):
        sample = self.nuImg.sample[sample_idx]

        sample_camera_data = self.nuImg.get('sample_data', sample['key_camera_token'])
        image_path = os.path.join(self.dataset_root, sample_camera_data['filename'])
        image = Image.open(image_path, 'r')

        # Load object instances.
        sd_token = sample['key_camera_token']
        object_anns = [o for o in self.nuImg.object_ann if o['sample_data_token'] == sd_token]

        # convert to CoCo format
        target_bbox = []
        target_labels = []
        target_areas = []
        target_crowds = []

        for ann in object_anns:
            # Get color, box, mask and name.
            category_token = ann['category_token']
            category_name = self.nuImg.get('category', category_token)['name']
            # color = self.nuImg.color_map[category_name]
            x1, y1, x2, y2 = ann['bbox']
            w = x2 - x1
            h = y2 - y1
            # attr_tokens = ann['attribute_tokens']
            # attributes = [self.nuImg.get('attribute', at) for at in attr_tokens]

            target_bbox.append([x1, y1, x2, y2])
            target_areas.append(w*h)
            target_crowds.append(0)
            target_labels.append(self.categories_idx_map[category_name])

        # COCO format
        target = {
            "image_id": int(sample_idx),
            "boxes": torch.FloatTensor(target_bbox) if len(object_anns) > 0 else torch.zeros((0, 4),
                                                                                             dtype=torch.float32),
            "labels": torch.IntTensor(np.array(target_labels)),
            "area": torch.FloatTensor(np.array(target_areas)),
            "iscrowd": torch.IntTensor(np.array(target_crowds))
        }
        if self.transform:
            image, target = self.transform(image, target)

        return image, target


if __name__ == '__main__':
    dataset_root = './dataset/nuimages'
    version = 'v1.0-mini'
    dataset = NuImagesCoCoDataset(dataset_root, version)
    # dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    _img, _target = dataset[1]
    _img.show()
    print(_target.keys())
    print(f"Dataset size = ", len(dataset))


