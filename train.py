import torch
from torchvision import transforms
import yaml
import os
import argparse
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim

from src.dataloader import DataSplitter, Datasets
from src.image_preprocessing import Transformation
from src.model_training import PeftModel, Trainer


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def build_transforms(transform_list):
    transform_ops = []
    for transform in transform_list:
        operation, params = list(transform.items())[0]
        if operation == "RandomRotation":
            transform_ops.append(transforms.RandomRotation(params))
        elif operation == "RandomResizedCrop":
            transform_ops.append(transforms.RandomResizedCrop(**params))
        elif operation == "RandomApply":
            nested_transforms = build_transforms(params['transformations'])
            transform_ops.append(transforms.RandomApply(nested_transforms, p=params['p']))
        elif operation == "ToTensor":
            transform_ops.append(transforms.ToTensor())
        elif operation == "RandomPerspective":
            transform_ops.append(transforms.RandomPerspective(**params))
        elif operation == "RandomAffine":
            transform_ops.append(transforms.RandomAffine(**params))
    return transforms.Compose(transform_ops)

def main(config_path):
    config = load_config(config_path)

    root = "/kaggle/input/chest-xray-pneumonia/chest_xray"

    train_data_dir = os.path.join(config['root_directory'], "train")
    eval_data_dir = os.path.join(config['root_directory'], "val")
    test_data_dir = os.path.join(config['root_directory'], "test")

    #Datasplitter
    val_split = 0.2
    data_splitter = DataSplitter(config['train_data_dir'], config['val_split'])
    train_filenames, val_filenames = data_splitter.split_file_names()

    #Transformations
    augmentation_transforms_list = build_transforms(config['transformations'])
    transforms = Transformation(config['model_id_timm'],
                                augmentation_transforms_list=augmentation_transforms_list)

    #Datasets
    datasets = Datasets(train_data_dir,
                        eval_data_dir,
                        test_data_dir,
                        train_filenames,
                        val_filenames,
                        transforms)
    
    train_dataloader = datasets.get_train_dataloader(batch_size=config['batch_size'])
    eval_dataloader = datasets.get_eval_dataloader(batch_size=config['batch_size'])
    test_dataloader = datasets.get_test_dataloader(batch_size=config['batch_size'])

    #define criterion
    criterion = torch.nn.CrossEntropyLoss()

    #define model
    if config["peft_model"]["fine_tuned"]:
        model = torch.load_model(config["model_path"])
    else:
        model = PeftModel(config['peft_model']['model_id_timm']).get_peft_lora_model(
            modules_to_save=["classifier"]
            )

    # Define the optimizer and scheduler outside the Trainer class
    optimizer = optim.AdamW(model.parameters(), **config['optimizer_params'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **config['scheduler_params'])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pass the scheduler to the Trainer class
    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      eval_dataloader=eval_dataloader,
                      optimizer=optimizer,
                      criterion=criterion,
                      device=device,
                      patience=5,
                      num_epochs=config['num_epochs'],
                      lr_scheduler=scheduler)

    trainer.train()


    trainer.test(test_dataloader, "best_model_elastic.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on specified dataset")
    parser.add_argument('config_path', type=str, help='Path to configuration file')
    args = parser.parse_args()
    main(args.config_path)