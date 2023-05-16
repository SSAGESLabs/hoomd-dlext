#!/bin/bash

PYTHON_EXECUTABLE=$(which python3)
PYTHON_SITELIB=$(${PYTHON_EXECUTABLE} -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
HOOMD_ROOT="${PYTHON_SITELIB}/hoomd"

cmake -S . -B build -DCMAKE_FIND_ROOT_PATH="${HOOMD_ROOT}" -DHOOMD_ROOT="${HOOMD_ROOT}"
cmake --build build --target install
