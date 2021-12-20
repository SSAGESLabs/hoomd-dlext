#!/bin/bash

PYTHON_EXECUTABLE=$(which python3)
HOOMD_ROOT=$(${PYTHON_EXECUTABLE} -c 'import site; print(site.getsitepackages()[0])')/hoomd/

cmake -S . -B build -DHOOMD_ROOT=${HOOMD_ROOT}
cmake --build build --target install
