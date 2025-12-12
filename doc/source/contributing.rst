============
Contributing
============

This guide will help you get started with contributing to the project.

Getting Started
===============

1. **Fork the Repository**

   Fork the repository on GitHub and clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/Thesis.git
      cd Thesis

2. **Set Up Development Environment**

   Follow the installation instructions in :doc:`installation`, then install development dependencies:

   .. code-block:: bash

      # Create and activate environment
      conda env create -f environments/environment_cellmil.yml
      conda activate cellmil

      # Install
      poetry install

3. **Create a Branch**

   Create a new branch for your feature or bug fix:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

Development Guidelines
=====================

Code Style
----------

- Follow PEP 8-ish style guidelines
- Use type hints for all function parameters and return values
- Use descriptive variable and function names


Documentation
-------------

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Update documentation when adding new features
- Build docs locally to test:

.. code-block:: bash

   cd src/doc/
   make html


Code Architecture
================

Understanding the codebase structure:

.. code-block:: text

   src/cellmil/
   ├── cli/                    # Command-line interfaces
   ├── models/                 # Deep learning models
   │   ├── segmentation/       # Cell segmentation models  
   │   └── mil/               # MIL models
   ├── data/                  # Data processing
   ├── features/              # Feature extraction
   ├── graph/                 # Graph creation
   ├── segmentation/          # Cell segmentation
   ├── visualization/         # Plotting and visualization
   ├── interfaces/            # Configuration classes
   ├── explainability/        # Explainability methods
   ├── statistics/            # Statistical analysis
   ├── utils/                 # Utility functions
   └── __tests__/                 # Test suite

Design Principles
-----------------

- **Modularity**: Each component should be self-contained
- **Configuration**: Use Pydantic models for configuration
- **Type Safety**: Use type hints throughout
- **Error Handling**: Provide clear error messages
- **Logging**: Use structured logging for debugging

Adding New Models
================

To add a new segmentation model:

1. Create a new file in ``cellmil/models/segmentation/``
2. Add model to ``__init__.py``
3. Update CLI to include new model
4. Add documentation

To add a new MIL model:

1. Create a new file in ``cellmil/models/mil/``
2. Follow existing model patterns
3. Add model to ``__init__.py``
4. Update CLI options


Documentation Guidelines
=======================

**Docstring Format:**

.. code-block:: python

   def extract_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
       """Extract features from segmented cells.
       
       Args:
           image: Input histological image
           mask: Binary segmentation mask
           
       Returns:
           Dictionary mapping feature names to values
           
       Raises:
           ValueError: If image and mask shapes don't match
           
       Example:
           >>> extractor = FeatureExtractor()
           >>> features = extractor.extract_features(image, mask)
           >>> print(features.keys())
       """

**API Documentation:**

- Document all public APIs
- Include usage examples
- Explain parameter choices
- Document return value formats
