# Stage 1: Builder
FROM registry.access.redhat.com/ubi9/python-311 AS builder

# Use root to avoid permission issues during build
USER 0

# Set working directory inside the container
WORKDIR /app

# Copy project metadata and source code
COPY pyproject.toml .
COPY src ./src/
COPY utils ./utils/

# Install the build module and pre-build the Python wheel
RUN pip install --no-cache-dir build
RUN python -m build --wheel

# Stage 2: Runtime
FROM registry.access.redhat.com/ubi9/python-311

# Set working directory inside the container
WORKDIR /app

# Copy pre-built wheel from builder
COPY --from=builder /app/dist/*.whl .

# Copy and install runtime dependencies and pre-built python wheel
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt *.whl

# Copy config files
COPY run.yaml .
COPY providers.d/ ./providers.d/

# Expose the server port
EXPOSE 8321

# Set the user to a non-root user
USER 1001

# Command to start llama stack server
CMD ["llama", "stack", "run", "run.yaml"]
