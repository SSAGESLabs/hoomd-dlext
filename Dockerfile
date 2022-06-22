FROM ssages/pysages-base:latest
WORKDIR /hoomd-dlext/.docker_build

COPY . ../
RUN cmake .. && make install
RUN python3 -c "import hoomd; import hoomd.dlext"
