import setuptools

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="tensorboard-plotter",   # The name of your PyPI package
    version="0.1.0",
    description="A tool to plot aggregate TensorBoard logs from the command line.",
    author="Giulio Schiavi",
    url="https://github.com/giuschio/tensorboard-plotter",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            # console_script_name = package.module:function
            "tensorboard-plot = tensorboard_plotter.plot:main"
        ],
    },
    # If you have data files (like templates), you might need:
    # include_package_data=True,
    # package_data={"tensorboard_plotter": ["some_data_file"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
