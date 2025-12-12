============
Installation
============

The following instructions will guide you through the installation process specifically for linux based systems.

System Requirements
===================

- CUDA-compatible GPU (recommended for deep learning models)
- 8+ GB RAM recommended for large WSI processing
- Conda package manager

Prerequisites
=============

Before installing, ensure you have the following:

1. **Conda Environment Manager**
   
   Download and install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `Anaconda <https://www.anaconda.com/products/distribution>`_.

2. **CUDA Toolkit** (Optional but recommended)
   
   For GPU acceleration, install the appropriate CUDA toolkit version for your system from the `NVIDIA website <https://developer.nvidia.com/cuda-toolkit>`_.

Installation Steps
==================

1. Clone the Repository
-----------------------

.. code-block:: bash

   git clone https://github.com/CamiloSinningUN/Thesis.git
   cd Thesis

2. Create Conda Environment (Linux)
----------------------------

.. code-block:: bash

   # Create environment from the provided environment file
   conda env create -f environments/environment_cellmil.yml

   # Activate the environment
   conda activate cellmil

2. Create Conda Environment (Windows)
----------------------------

.. code-block:: bash

   # Create environment from the provided environment file
   conda env create -f environments/environment_cellmil_win.yml

   # Activate the environment
   conda activate cellmil_win

3. Install Python Dependencies
------------------------------

.. code-block:: bash

   # Install all Python dependencies using Poetry
   poetry install

1. Install Pytorch Geometric Dependencies
------------------------------

.. code-block:: bash

   # Install Pytorch Geomatric Dependencies (Issues could arise when installing)
   pip install --no-binary :all: torch-cluster torch-scatter pyg-lib 

Verification
============

To verify that the installation was successful, try running:

.. code-block:: bash

   # Check if the CLI tools are available
   poetry run patch_extraction --help

You should see help messages for each command without any errors.

Environment Files
=================

The project includes several environment files for different use cases:

- ``environment_cellmil.yml`` - Main environment for this package
- The other environment files located in the `environments/` directory are for specific experiments or configurations.

GPU Support
===========

For optimal performance, especially when working with deep learning models, GPU support is highly recommended:

1. **NVIDIA GPU**: Ensure you have a CUDA-compatible NVIDIA GPU
2. **CUDA Drivers**: Install the latest NVIDIA drivers
3. **PyTorch with CUDA**: The environment file includes PyTorch with CUDA support

To verify GPU support:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")

Troubleshooting
===============

Common Issues
-------------

**CUDA Issues**

If you have CUDA-related issues:

1. Verify your NVIDIA driver version: ``nvidia-smi``
2. Check CUDA toolkit version: ``nvcc --version``
3. Reinstall PyTorch with the correct CUDA version

**Memory Issues**

For large WSI processing:

1. Reduce patch size or batch size
2. Use CPU processing for initial testing
3. Ensure sufficient disk space for temporary files

**Environment Conflicts**

If you encounter package conflicts:

.. code-block:: bash

   # Remove and recreate the environment
   conda env remove -n cellmil
   conda env create -f environments/environment_cellmil.yml

