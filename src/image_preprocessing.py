from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import timm
from timm.data import resolve_data_config, create_transform


class Transformation:
    def __init__(self, model_id_timm, augmentation_transforms_list=None):
        self.model_id_timm = model_id_timm
        self.base_transform = self._get_base_transforms()
        self._augmentation_transforms_list = augmentation_transforms_list

    def _get_timm_model(self):
        return timm.create_model(self.model_id_timm, pretrained=True, num_classes=2)

    def _get_base_transforms(self):
        model = self._get_timm_model()
        transform_seq = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        return transform_seq

    @property
    def augmented_transforms(self):
        if self._augmentation_transforms_list is not None:
            return transforms.Compose(
                [self.base_transform] + self._augmentation_transforms_list
            )
        else:
            return self.base_transform

    def show_images_with_transformations(self, img_path, num_images=5):
        # Open the image using PIL
        image = Image.open(img_path)

        # Convert the image to RGB format if it's not already in RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Create a figure with subplots for each transformed image
        _, axes = plt.subplots(1, num_images + 1, figsize=(15, 5))

        # Plot the original image
        axes[0].imshow(image)
        axes[0].axis("off")
        axes[0].set_title("Original Image")

        # Apply the transformation to the original image and plot the transformed images
        for i in range(1, num_images + 1):
            transformed_image = self.augmented_transforms(image)
            transformed_image_array = transformed_image.permute(1, 2, 0).numpy()
            axes[i].imshow(transformed_image_array)
            axes[i].axis("off")
            axes[i].set_title(f"Transformed Image {i}")

        # Show the plot
        plt.show()
