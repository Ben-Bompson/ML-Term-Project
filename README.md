﻿# ML-Term-Project

This guide will walk you through the steps to install the required dependencies and run the PC Build Optimizer script, which helps you build the most optimized PC system within a given budget using machine learning.

*Python (3.6 at least) must be installed along with pip for package dependencies.*

*Ensure the entire repository is run and terminal directory is the root of the repository or else datasets may not be read correctly.*

## Installing Dependencies

To install dependencies, run the following command:
```
pip install pandas scikit-learn streamlit numpy
```

Setting up a virtual environment to install these in *may* be necessary. If so, please refer to [python.org's official guide on virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) prior to running the pip install command.

## Running Program

You can then run the script with:
```
streamlit run pc_builder_app.py
```

You will be prompted to enter total budget. The program will then generate a list of parts that would make an ideal build with this budget.

Program ends with ctrl-c in the terminal as with any program.
