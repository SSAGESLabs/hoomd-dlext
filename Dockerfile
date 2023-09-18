FROM ssages/pysages-base:latest
WORKDIR /hoomd-dlext

# Install python dependencies
# hadolint ignore=DL3013
RUN python3 -m pip install --no-cache-dir --upgrade pip "setuptools-scm==7.1.0"

# Build the plugin
COPY . .
RUN cmake -S . -B build && cmake --build build --target install && rm -rf build

# Test it can be loaded
RUN python3 -c "import hoomd; import hoomd.dlext"
