# Use the latest tf GPU image
FROM tensorflow/tensorflow:latest-gpu

# NOTE: if error, remove. Add files to docker image, handle that the image has the src code when pulled
# ADD . /app

# Set working directory to app
WORKDIR /app

# Copy local files to container
COPY . .

# Update linux dist. repositories
RUN apt-get update

# Install git
RUN apt-get install git -y

# Install SQLite
RUN apt-get install sqlite3 -y

# Install 7zip
RUN apt-get install p7zip-full -y

# Install pip3 (parent image only comes with python 2 image)
RUN apt-get install python3-pip -y

# Install Python3-tk (Maybe not needed as long as we use pdf backend?)
RUN apt-get install python3-tk -y

# Update pip3
RUN pip3 install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip3 install -r ./requirements.txt
