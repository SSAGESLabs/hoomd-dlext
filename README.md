# HOOMD-dlext

Provides access to [HOOMD-blue](https://hoomd-blue.readthedocs.io) simulation data on CPU
or GPU via [DLPack](https://github.com/dmlc/dlpack) This project is primarily designed to
enable access to HOOMD-blue for the [PySAGES](https://pysages.readthedocs.io) project.
HOOMD-blue versions 2, 3, and 4 are supported (support HOOMD-blue v4 has not been
thoroughly tested).

## Installation

The latest version of `hoomd-dlext` can be installed via conda:

```shell
conda install -c conda-forge hoomd-dlext
```

## Building from source

The following instructions are similar for all HOOMD-blue versions.

For HOOMD-blue v2, follow the [Plugins and Components
Guide](https://hoomd-blue.readthedocs.io/en/v2.9.7/developer.html) from the HOOMD-blue
reference documentation site; and check [HOOMD-blue's install
requirements](https://hoomd-blue.readthedocs.io/en/v2.9.7/installation.html#compiling-from-source).
At the moment we only support installations on Linux and Mac. For GPU support, the base
HOOMD-blue installation must be built for CUDA GPUs.

Assuming HOOMD-blue is already installed on the system, the plugin can be installed as an
external component

First, we obtain a copy of this plugin, for example via `git clone`.

```shell
git clone https://github.com/SSAGESLabs/hoomd-dlext.git
cd hoomd-dlext
```

We then configure the installation with CMake. It is important, that the python version
detected by CMake can successfully `import hoomd`.

```shell
mkdir build && cd build
cmake -S . -B build
```

And finally we compile the plugin on the target machine

```shell
cmake --build build -j8
```

and install it into the HOOMD-blue installation path (the latter must be writeable for
this to work).

```shell
cmake --build build -j8 --target install
```

For a quick and simple test run:

```shell
cd ~
python -c "import hoomd; import hoomd.dlext"
```
