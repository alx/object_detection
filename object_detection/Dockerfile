# Use a Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for Darknet compilation and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    libopencv-dev \
    # Add any other specific dependencies for hank-ai/darknet if needed (e.g., libomp-dev for OpenMP)
    && rm -rf /var/lib/apt/lists/*

# Clone the hank-ai/darknet repository
RUN git clone https://github.com/hank-ai/darknet.git /opt/darknet

# Build Darknet
# Navigate to the darknet directory
WORKDIR /opt/darknet
# Modify Makefile: Enable OpenCV, LIBSO (for libdarknet.so), disable GPU (for CPU-only, adjust if GPU needed)
# Ensure these sed commands correctly target the lines in the Makefile of hank-ai/darknet
RUN sed -i 's/GPU=1/GPU=0/' Makefile \
    && sed -i 's/CUDNN=1/CUDNN=0/' Makefile \
    && sed -i 's/CUDNN_HALF=1/CUDNN_HALF=0/' Makefile \
    && sed -i 's/OPENCV=0/OPENCV=1/' Makefile \
    && sed -i 's/LIBSO=0/LIBSO=1/' Makefile \
    && sed -i 's/OPENMP=0/OPENMP=1/' Makefile # Optional: Enable OpenMP for CPU acceleration
# Compile Darknet
RUN make -j$(nproc)

# Copy the compiled shared library and Python wrapper to the app directory
WORKDIR /app
COPY /opt/darknet/libdarknet.so .
COPY /opt/darknet/python/darknet.py .
# If darknet.py from hank-ai/darknet has other dependencies in its python/ folder, copy them too.
# Typically, darknet.py and libdarknet.so in the same directory are sufficient.

# Set LD_LIBRARY_PATH so the Python wrapper can find libdarknet.so
ENV LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH

# Copy the application requirements file
COPY object_detection/requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install yt-dlp and ffmpeg (if not already in base image or provided differently)
RUN pip install --no-cache-dir yt-dlp
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY object_detection/app.py .

# Make sure the output directories will be writable by the app
# The user running the app inside the container might not have write access by default.
# One way is to create them and chmod them.
# Alternatively, manage volumes and permissions from docker-compose.
RUN mkdir -p output/fullframe output/cropped output_images \
    && chmod -R 777 output output_images

# Environment variables for Darknet model paths (to be set at runtime or in docker-compose)
# Example placeholders:
# ENV DARKNET_DATA_PATH="/models/obj.data"
# ENV DARKNET_CFG_PATH="/models/yolov4-tiny.cfg"
# ENV DARKNET_WEIGHTS_PATH="/models/yolov4-tiny.weights"
# ENV DETECTION_THRESHOLD="0.25"

# Default command to run the application
CMD ["python", "app.py"]
