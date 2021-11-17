FROM ssages/pysages-base:latest

COPY . hoomd-dlext
RUN  cd hoomd-dlext && mkdir build && cd build && cmake .. && make install
RUN python3 -c "import hoomd; import hoomd.dlext"