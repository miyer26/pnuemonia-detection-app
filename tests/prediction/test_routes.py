# test_routes.py
from flask_testing import TestCase
from app import app  # Adjust this import to fit your project structure.
import os
import pytest

class TestFlaskRoutes(TestCase):
    def create_app(self):
        app.config['TESTING'] = True
        return app

    def test_predict(self):
        path_to_image = os.path.join('tests', 'prediction','images', 'image_1.jpeg')
        with open(path_to_image, 'rb') as img:
            response = self.client.post(
                '/predict',
                data={'file': (img, 'test_image.jpg')},
                content_type='multipart/form-data'
            )
        self.assertEqual(response.status_code, 200)
        # Add more assertions based on your application's response structure.

# Run the tests
if __name__ == "__main__":
    pytest.main()
