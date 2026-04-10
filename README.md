# Opera_tools

```bash
conda create -n opera_tools -y
conda activate opera_tools
conda install -c conda-forge python==3.11.13 gdal libgdal-netcdf
```

```bash
git clone git@github.com:geodesymiami/Opera_tools.git
```

```bash
cd Opera_tools
export OPERA_TOOLS=${PWD}
export PATH=${OPERA_TOOLS}/:$PATH
export PYTHONPATH=${OPERA_TOOLS}/:$PYTHONPATH
```
```bash
python -c "from setup_env import install_dependencies; install_dependencies()"
```
