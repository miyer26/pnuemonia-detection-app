import pytest
from unittest.mock import Mock, patch
from PIL import Image
import torch
from src.image_preprocessing import Transformation
from src.model_inference import Inference

# Fixture to handle the setup of an Image object
@pytest.fixture
def sample_image():
    return Image.new('RGB', (256, 256))

# Fixture for transformation
@pytest.fixture
def mock_transformation():
    transform = Mock(spec=Transformation)
    # Mocking the base_transform attribute to return a Mock object
    transform.base_transform = Mock(return_value=torch.zeros((3, 224, 224)))
    return transform

# Fixture for model loading
@pytest.fixture
def mock_model():
    model = Mock()
    model.eval.return_value = None
    model.return_value = torch.tensor([[0.7, 0.3]])
    return model

@pytest.fixture
def inference_instance(mock_model, mock_transformation):
    with patch('torch.load', return_value=mock_model):
        return Inference(model_path="dummy/path/model.pth", transforms=mock_transformation)

def test_error_handling_model_loading():
    # Test error handling when model loading fails
    with pytest.raises(Exception):  # Adjust the exception type as per your actual implementation
        inference = Inference(model_path="nonexistent/path/model.pth", transforms=None)

def test_device_handling_cuda_available():
    # Test device handling when CUDA is available
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.load', return_value=Mock()):  # Mocking torch.load to avoid actual file loading
            inference = Inference(model_path="dummy/path/model.pth", transforms=None)
            assert inference.device.type == 'cuda'


def test_device_handling_cuda_not_available():
    # Test device handling when CUDA is not available
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.load', return_value=Mock()):  # Mocking torch.load to avoid actual file loading
            inference = Inference(model_path="dummy/path/model.pth", transforms=None)
            assert inference.device.type == 'cpu'

def test_transformations(sample_image, inference_instance):
    # Test transformations applied to the input image
    with patch('PIL.Image.open', return_value=sample_image):
        inference_instance.predict("dummy/path/image.jpg")
        inference_instance.transforms.base_transform.assert_called_once_with(sample_image)

def test_output_label_mapping_normal(sample_image, inference_instance):
    # Test output label mapping for a normal prediction
    inference_instance.model.return_value = torch.tensor([[0.9, 0.1]])  # High confidence for "Normal" class
    with patch('PIL.Image.open', return_value=sample_image):
        label = inference_instance.predict("dummy/path/image.jpg")
        assert label == "Normal"

def test_output_label_mapping_pneumonia(sample_image, inference_instance):
    # Test output label mapping for a pneumonia prediction
    inference_instance.model.return_value = torch.tensor([[0.1, 0.9]])  # High confidence for "Pneumonia" class
    with patch('PIL.Image.open', return_value=sample_image):
        label = inference_instance.predict("dummy/path/image.jpg")
        assert label == "Pneumonia"