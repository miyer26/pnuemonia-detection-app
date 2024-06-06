[![Run pytest](https://github.com/miyer26/pnuemonia-detection-app/actions/workflows/pytest.yml/badge.svg)](https://github.com/miyer26/pnuemonia-detection-app/actions/workflows/pytest.yml)

# Pneumonia Detection App

This API is designed to be used by pulmonologists to detect a potential case of pneumonia. The doctor can upload an xray image using the API, which runs the inference on the backend and returns the prediction.


## Usage

## Prerequisites

- Ensure you have [Docker](https://www.docker.com/products/docker-desktop) installed on your machine.
- Optionally, Docker Compose if your project uses multiple containers.

## Building the Docker Image

1. Clone the repository:

    ```sh
    git clone https://github.com/miyer26/pnuemonia-detection-app.git
    cd pnuemonia-detection-api
    ```

2. Build the Docker image:

    ```sh
    docker build -t pneumonia-detection-app .
    ```

    This command will create a Docker image with the tag `pneumonia-detection-app`.

## Running the Docker Container

1. Run the Docker container:

    ```sh
    docker run -p 8080:5000 pneumonia-detection-app
    ```

    This command will run the container and map port 8080 on your host to port 8080 in the container. Adjust the ports as necessary for your application.

2. Access the application:

    Open your web browser and go to `http://localhost:8080`.
