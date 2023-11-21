# Base image for the Python server
FROM nikolaik/python-nodejs:python3.10-nodejs20

# Create a working directory
WORKDIR /app

# Install Python packages
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install -y python3-opencv \
        libxcb-xinerama0 \
        libxkbcommon-x11-0


COPY package*.json ./
RUN npm install
# Copy the rest of the code
COPY . .

# Run the server
ENV HOST 0.0.0.0
ENV VOLUME morphs-volume
#ENV MEDIAPIPE_DISABLE_GPU=1
#ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/cv2/qt/plugins
#ENV QT_QPA_PLATFORM offscreen
CMD /app/start.sh
#CMD ["dumb-init", "node", "/app/index.js"]

EXPOSE 8050
EXPOSE 8000
