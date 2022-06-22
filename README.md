# HOOMD-dlext

Provides access to [HOOMD-blue](https://hoomd-blue.readthedocs.io/en/v2.9.7/) simulation data on CPU or GPU via [DLPack](https://github.com/dmlc/dlpack)
This project is primarily designed to enable access to HOOMD-blue for the [PySAGES](https://pysages.readthedocs.io/en/latest/) project.
At the moment, only HOOMD-blue version 2 is supported (support HOOMD-blue version 3 is in the works).

## Installation

Follow the [Plugins and Components Guide](https://hoomd-blue.readthedocs.io/en/v2.9.7/developer.html) from the HOOMD-blue reference documentation site.
For system requirements, check HOOMD-blue's install [requirements](https://hoomd-blue.readthedocs.io/en/v2.9.7/installation.html#compiling-from-source).
At the moment we only support installations on Linux and Mac.
For GPU support, the base HOOMD-blue installation must be built for CUDA GPUs.

Assuming HOOMD-blue is already installed on the system, the plugin can be installed as an external component

First, we obtain a copy of this plugin, for example via `git clone`.

```shell
git clone https://github.com/SSAGESLabs/hoomd-dlext.git
cd hoomd-dlext

We then configure the installation with CMake. It is important, that the python version detected by CMake can successfully `import hoomd`.

```shell
mkdir build && cd build
cmake ..
```

And finally we compile the plugin on the target machine
```shell
make
```
and install it into the HOOMD-blue installation path (the latter must be writeable for this to work).
```shell
make install
```

For a quick and simple test run:
```shell
cd ~
python -c "import hoomd; import hoomd.dlext"
```