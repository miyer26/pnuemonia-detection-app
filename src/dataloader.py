import os
from torchvision.datasets import ImageFolder
import torch
import random
from src.image_preprocessing import Transformation


class DataSplitter:
    def __init__(self, input_folder, val_split_perc):
        self.input_folder = input_folder
        self.val_split_perc = val_split_perc

    def extract_patient_ids(self, filename):
        patient_id = filename.split("_")[0].replace("person", "")
        return patient_id

    def split_file_names(self):
        pneumonia_patient_ids = set(
            [
                self.extract_patient_ids(fn)
                for fn in os.listdir(os.path.join(self.input_folder, "PNEUMONIA"))
            ]
        )
        pneumonia_val_patient_ids = random.sample(
            list(pneumonia_patient_ids),
            int(self.val_split_perc * len(pneumonia_patient_ids)),
        )

        pneumonia_val_filenames = []
        pneumonia_train_filenames = []

        for filename in os.listdir(os.path.join(self.input_folder, "PNEUMONIA")):
            patient_id = self.extract_patient_ids(filename)
            if patient_id in pneumonia_val_patient_ids:
                pneumonia_val_filenames.append(
                    os.path.join(self.input_folder, "PNEUMONIA", filename)
                )
            else:
                pneumonia_train_filenames.append(
                    os.path.join(self.input_folder, "PNEUMONIA", filename)
                )

        normal_filenames = [
            os.path.join(self.input_folder, "NORMAL", fn)
            for fn in os.listdir(os.path.join(self.input_folder, "NORMAL"))
        ]
        normal_val_filenames = random.sample(
            normal_filenames, int(self.val_split_perc * len(normal_filenames))
        )
        normal_train_filenames = list(set(normal_filenames) - set(normal_val_filenames))

        train_filenames = pneumonia_train_filenames + normal_train_filenames
        val_filenames = pneumonia_val_filenames + normal_val_filenames

        return train_filenames, val_filenames


class ExtendedImageFolder(ImageFolder):
    def add_items(self, new_data_dir, transform=None):
        # Update class_to_idx mapping
        classes = [d.name for d in os.scandir(new_data_dir) if d.is_dir()]
        for cls in classes:
            if cls not in self.class_to_idx:
                self.class_to_idx[cls] = len(self.class_to_idx)

        # Update samples list with new data
        new_samples = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(new_data_dir, target_class)
            if os.path.isdir(target_dir):
                new_samples += [
                    (os.path.join(target_dir, file), class_index)
                    for file in os.listdir(target_dir)
                ]

        self.samples += new_samples
        self.targets += [s[1] for s in new_samples]


class Datasets(ExtendedImageFolder):
    def __init__(
        self,
        train_data_dir,
        eval_data_dir,
        test_data_dir,
        train_filenames,
        val_filenames,
        transforms: Transformation,
    ):
        self.train_data_dir = train_data_dir
        self.val_data_dir = eval_data_dir
        self.test_data_dir = test_data_dir
        self.train_filenames = train_filenames
        self.val_filenames = val_filenames
        self.transforms = transforms

    def get_train_dataset(self):
        return ExtendedImageFolder(
            root=self.train_data_dir,
            transform=self.transforms.augmented_transforms,
            is_valid_file=lambda x: x in self.train_filenames,
        )

    def get_eval_dataset(self):
        return ExtendedImageFolder(
            root=self.eval_data_dir,
            transform=self.transforms.base_transforms,
            is_valid_file=lambda x: x in self.eval_filenames,
        )

    def get_test_dataset(self):
        return ExtendedImageFolder(
            root=self.test_data_dir, transform=self.transforms.base_transforms
        )

    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples])
        labels = torch.tensor([example[1] for example in examples])
        return {"x": pixel_values, "y": labels}

    def train_dataloader(self, batch_size=32):
        return torch.utils.data.DataLoader(
            self.get_train_dataset(),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def eval_datalaoder(self, batch_size=32):
        return torch.utils.data.DataLoader(
            self.get_eval_dataset(),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self, batch_size=32):
        return torch.utils.data.DataLoader(
            self.get_test_dataset(),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
