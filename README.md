# Opera_tools

```bash
conda create -n opera_tools -y
conda activate opera_tools
conda install -c conda-forge python==3.11.13 gdal libgdal-netcdf
```

```bash
cd Opera_tools
python -c "from setup_env import install_dependencies; install_dependencies()"
```
