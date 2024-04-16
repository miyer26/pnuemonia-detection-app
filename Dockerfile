# Use the official Python image as the base image
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Copy the requirements file into the container at /app
COPY . /usr/app/

# Set the working directory in the container
WORKDIR /usr/app/

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the Flask application
CMD python app.py

# Expose port 5000 to allow communication to/from the Flask application
EXPOSE 5000