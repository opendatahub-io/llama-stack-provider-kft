FROM registry.access.redhat.com/ubi9/python-311

# Set working directory inside the container
WORKDIR /app

# Copy and install runtime dependencies and pre-built python wheel
COPY requirements.txt .
COPY dist/*.whl .
RUN pip install --no-cache-dir -r requirements.txt *.whl

# Copy config files
COPY run.yaml .
COPY providers.d/ ./providers.d/

# Expose the server port
EXPOSE 8321

# Set the user to a non-root user
USER 1001

# Command to start llama stack server
CMD ["llama", "stack", "run", "--image-type", "venv", "run.yaml"]