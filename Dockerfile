# Use the official Python image as the base image
FROM continuumio/anaconda3:4.4.0

# Copy the requirements file into the container at /app
COPY . /usr/app/

# Expose port 5000 to allow communication to/from the Flask application
EXPOSE 5000

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory in the container
WORKDIR /usr/app/

# Command to run the Flask application
CMD python app.py