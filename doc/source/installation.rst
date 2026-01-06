============
Installation
============

System Requirements
===================

- CUDA-compatible GPU (recommended for deep learning models)
- Conda package manager or Poetry (for development setup)

Development Setup
=================

For users who want to modify the code or contribute to the project. This is the **recommended** approach for researchers and developers.

1. Clone the Repository
-----------------------

.. code-block:: bash

   git clone https://github.com/CamiloSinningUN/CellMIL.git
   cd CellMIL

2. Create Conda Environment
---------------------------

.. note::

   This step can be skipped if you already have Python 3.10 and Poetry installed on your machine.

If you don't have conda installed, follow the instructions from the `official documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

.. code-block:: bash

   # Create environment
   conda env create -f environment.yml

   # Activate environment
   conda activate cellmil

3. Install Python Dependencies
------------------------------

.. code-block:: bash

   poetry install

.. note::

   Poetry will install PyTorch with CUDA 11.8 support. If you have a GPU with an older CUDA version, install PyTorch manually first before running ``poetry install``. Visit `pytorch.org <https://pytorch.org/get-started/locally/>`_ to get the correct command for your system.

4. Install Additional Dependencies
----------------------------------

.. code-block:: bash

   # Pyradiomics (incompatible with Poetry resolver)
   pip install pyradiomics

.. note::

   **Optional:** Install cucim for GPU-accelerated image loading from their `official documentation <https://docs.rapids.ai/api/cucim/stable/>`_

Verification
============

To verify that the installation was successful, try running:

.. code-block:: bash

   # Check if the CLI tools are available
   patch_extraction --help

You should see help messages for each command without any errors.

GPU Support
===========

For optimal performance, especially when working with deep learning models, GPU support is highly recommended:

1. **NVIDIA GPU**: Ensure you have a CUDA-compatible NVIDIA GPU
2. **CUDA Drivers**: Install the latest NVIDIA drivers
3. **PyTorch with CUDA**: Install PyTorch with the correct CUDA version

To verify GPU support:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")

Troubleshooting
===============

Common Issues
-------------

**Windows DataLoader Workers**

If you encounter errors or duplicate runs, Windows may have issues with ``num_workers`` in DataLoader.

**Solution:** Comment out all ``num_workers`` arguments in the codebase.

**PyTorch Geometric Libraries**

Errors with ``torch-sparse``, ``torch-scatter``, or ``pyg-lib`` may occur due to pre-built binary incompatibilities.

.. note::

   This is only necessary if your GPU has a CUDA version older than 11.8 or you are experiencing specific errors. Otherwise, the default installation should work fine.

**Solution:** Compile the libraries from source:

.. code-block:: bash

   pip install --no-binary :all: torch-sparse
   pip install --no-binary :all: torch-scatter
   pip install --no-binary :all: pyg-lib

**Pyradiomics Installation**

When installing ``pyradiomics`` with pip, you may encounter build errors like:

.. code-block:: text

   ModuleNotFoundError: No module named 'numpy'
   ModuleNotFoundError: No module named 'versioneer'

This happens because pyradiomics has undeclared build dependencies, and pip's build isolation creates a clean environment without access to your installed packages.

**Solution:** Install the missing build dependencies first, then install pyradiomics with ``--no-build-isolation``:

.. code-block:: bash

   pip install versioneer cython
   pip install pyradiomics --no-build-isolation

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
   conda env create -f environment.yml

