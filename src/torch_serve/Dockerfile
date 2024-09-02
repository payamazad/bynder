# Use the official TorchServe image as the base image
FROM pytorch/torchserve:latest

# Copy your model store and configuration files into the container
COPY model_store/ /home/model-server/model-store/
COPY config.properties /home/model-server/config.properties
COPY custom_handler.py /home/model-server/custom_handler.py

# Set environment variables
ENV TORCHSERVE_CONFIG=/home/model-server/config.properties

# Expose the ports
EXPOSE 8080 8081

# Run TorchServe
CMD ["torchserve", "--start", "--ncs", "--model-store", "/home/model-server/model-store", "--ts-config", "/home/model-server/config.properties"]