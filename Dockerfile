FROM innocentbug/pysages-base:latest


#HOOMD-blue dlext plugin
RUN git clone https://github.com/SSAGESLabs/hoomd-dlext.git && cd hoomd-dlext && mkdir build && cd build && cmake .. && make install

