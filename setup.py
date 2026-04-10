from setuptools import setup, find_packages

setup(
    name="opera-tools",
    version="0.1.0",
    description="Tools for working with OPERA SAR/InSAR product files",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="geodesymiami",
    url="https://github.com/geodesymiami/Opera_tools",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "rasterio",
        "h5py",
    ],
    entry_points={
        "console_scripts": [
            "opera_subset=opera_tools.subset:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
