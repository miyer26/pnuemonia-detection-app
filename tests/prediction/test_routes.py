# test_routes.py
import pytest
import torch

from unittest.mock import Mock, patch
from flask_testing import TestCase
from PIL import Image
from app import app  # Adjust this import to fit your project structure.


from src.model_inference import Inference

@pytest.fixture
def sample_image():
    return Image.new('RGB', (256, 256))

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
 
class TestFlaskRoutes(TestCase):
    def create_app(self):
        app.config['TESTING'] = True
        return app

    @pytest.mark.parametrize("inference_instance, sample_image", [(inference_instance, sample_image)])
    def test_predict(self):
        with patch('PIL.Image.open', return_value=sample_image):
            with patch('torch.load', return_value=inference_instance):
                response = self.client.post(
                    '/predict',
                    data={'file': (sample_image, 'test_image.jpg')},
                    content_type='multipart/form-data'
                )

        self.assertEqual(response.status_code, 200)
# Run the tests
if __name__ == "__main__":
    pytest.main()
