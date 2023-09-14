FROM ssages/pysages-base:latest
WORKDIR /hoomd-dlext/.docker_build

# Install python dependencies
# hadolint ignore=DL3013
RUN python3 -m pip install --no-cache-dir --upgrade pip
# hadolint ignore=DL3059
RUN python3 -m pip install --no-cache-dir "setuptools-scm==7.1.0"

# Build the plugin
COPY . ../
RUN cmake .. && make install

# Test it can be loaded
RUN python3 -c "import hoomd; import hoomd.dlext"
