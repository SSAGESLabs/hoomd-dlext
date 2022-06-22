# HOOMD-dlext

Provides access to [HOOMD-blue](https://hoomd-blue.readthedocs.io/en/v2.9.7/) simulation data on CPU or GPU via [DLPack](https://github.com/dmlc/dlpack)
This project is primarily designed to enable access to HOOMD-blue for the [PySAGES](https://pysages.readthedocs.io/en/latest/) project.
At the moment, we only support HOOMD-blue version 2.
Support HOOMD-blue version 3 is planned and will be released later.

## Installation

Follow the [Plugins and Components Guide](https://hoomd-blue.readthedocs.io/en/v2.9.7/developer.html) from the HOOMD-blue reference documentation site.
For system requirements, check HOOMD-blue's install [requirements](https://hoomd-blue.readthedocs.io/en/v2.9.7/installation.html#compiling-from-source).
At the moment we only support installations on Linux and Mac.
For GPU support, the base HOOMD-blue installation must be installed for CUDA GPUs.

For example installation as an external component can be performed after HOOMD-blue is already installed on the system.

The next step is to obtain a copy of this plugin preferably via `git clone`.

```shell
cd /path/to/hoomd-dlext
```

Now we configure the installation with CMake. Here it is important, that the python version detected by CMake can successfully `import hoomd`.

```shell
mkdir build && cd build
cmake ..
```

And finally we compile the plugin on your target machine
```shell
make
```
and install it into the HOOMD-blue installation.
It is important to have a writeable installation for this to work.
```shell
make install
```

For a quick and simple test run:
```shell
cd ~
python -c "import hoomd; import hoomd.dlext"
```