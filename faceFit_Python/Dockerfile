# Use an official Python runtime as a parent image
FROM python:3.9.12

# Set the working directory to /app
WORKDIR /app

# Copy only requirement file in order to optmizize build process
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install -y python3-opencv
RUN apt-get install -y libmtdev-dev
RUN pip3 install opencv-python
RUN pip3 install kivy[full]
ENV DISPLAY=:0
# Copy the current directory contents into the container at /app
COPY . .

# Define environment variable
ENV NAME FaceFit

# Run app.py when the container launches
CMD [ "python3", "FFApp.py"]