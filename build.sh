#!/bin/bash

#CUDA_PATH=/usr/lib/cuda-10.2/
PYTHON_EXECUTABLE=$(which python3)
HOOMD_ROOT=$(${PYTHON_EXECUTABLE} -c 'import site; print(site.getsitepackages()[0])')/hoomd/

cmake -S . -B build \
    -DBUILD_TESTING=OFF \
    -DCOPY_HEADERS=ON \
    -DHOOMD_ROOT=${HOOMD_ROOT}
    #-DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} \
    #-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH} \

cmake --build build --target install
