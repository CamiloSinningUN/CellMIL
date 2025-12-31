===================================
Training
===================================

.. contents:: Table of Contents
   :depth: 2
   :local:

This section describes how to train MIL models using k-fold cross-validation, the recommended approach for robust model evaluation.

K-Fold Cross-Validation
========================

Overview
--------

The recommended approach for training and evaluating MIL models is using k-fold cross-validation. This provides more robust performance estimates than a single train/validation split and automatically handles data splitting and transform fitting.


Model Creator Function
----------------------

The ``lit_model_creator`` function is called for each fold to create a fresh model instance:

.. code-block:: python

    def lit_model_creator(input_dim: int, use_lr_scheduler: bool = True) -> Pl.LightningModule:
         """
         Create a LightningModule for training.
         
         Args:
              input_dim: Input feature dimension (determined automatically)
              use_lr_scheduler: Whether to use learning rate scheduler
              
         Returns:
              LightningModule instance ready for training
         """
         # Create model architecture
         model = AttentionDeepMIL(
              embed_dim=input_dim,
              size_arg=[256, 128],
              n_classes=2,
         )
         
         # Define optimizer
         optimizer = AdamW(model.parameters(), lr=1e-4)
         
         # Optional learning rate scheduler
         lr_scheduler = None
         if use_lr_scheduler:
              lr_scheduler = ReduceLROnPlateau(
                    optimizer, mode="min", patience=5, factor=0.8
              )
         
         # Wrap in LightningModule
         lit_model = LitAttentionDeepMIL(
              model=model,
              optimizer=optimizer,
              loss=FocalLoss(alpha=0.5, gamma=2.0),
              lr_scheduler=lr_scheduler,
         )
         
         return lit_model

**Key points:**

- Takes ``input_dim`` as parameter (automatically determined from data)
- Takes ``use_lr_scheduler`` to optionally enable learning rate scheduling
- Returns a ``LightningModule`` instance
- Creates fresh model for each fold (no weight sharing)
- Defines optimizer, loss, and optional scheduler

K-Fold Parameters
=================

**Constructor Parameters (KFoldCrossValidation):**

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``k``
     - ``5``
     - Number of cross-validation folds
   * - ``random_state``
     - ``42``
     - Random seed for reproducibility

**Evaluate Method Parameters:**

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``name``
     - Required
     - Experiment name for logging and checkpoints
   * - ``lit_model_creator``
     - Required
     - Function that takes ``input_dim`` and returns a ``LightningModule``
   * - ``dataset``
     - Required
     - The full MIL dataset (no split)
   * - ``output_dir``
     - Required
     - Directory to store all results and checkpoints
   * - ``wandb_project``
     - Required
     - Weights & Biases project name for logging
   * - ``transforms``
     - ``None``
     - Transform pipeline (fit on each fold's training data)
   * - ``label_transforms``
     - ``None``
     - Label transforms (e.g., ``TimeDiscretizerTransform`` for survival)
   * - ``balance_cell_counts``
     - ``False``
     - Jointly stratify by label and cell-count quantiles
   * - ``cell_balance_bins``
     - ``5``
     - Number of quantile bins when balancing cell counts
   * - ``early_stopping_patience``
     - ``30``
     - Epochs without improvement before early stopping

Advanced Features
=================

Cell-Count Balanced Stratification
-----------------------------------

When slides have varying numbers of cells, you can balance folds by both label and cell count:

.. code-block:: python

   k_fold = KFoldCrossValidation(k=5, random_state=42)
   
   model_storage = k_fold.evaluate(
       name="balanced_experiment",
       lit_model_creator=lit_model_creator,
       dataset=dataset,
       output_dir=Path("./results"),
       wandb_project="my_project",
       transforms=transforms,
       balance_cell_counts=True,    # Enable cell-count balancing
       cell_balance_bins=5,          # Number of quantile bins
   )

This creates a combined stratification target from labels and cell-count quantile bins, ensuring each fold has similar distributions of both class labels and cell counts.

Class Imbalance Handling
-------------------------

For imbalanced datasets, use ``FocalLoss`` with computed class weights:

.. code-block:: python

   from cellmil.utils.train import FocalLoss
   from cellmil.utils.train.dataset import complementary_frequencies
   
   # Get class frequencies from your data
   _, alpha = complementary_frequencies(df, 'label')
   
   def lit_model_creator(input_dim: int) -> Pl.LightningModule:
       model = AttentionDeepMIL(embed_dim=input_dim, ...)
       
       lit_model = LitAttentionDeepMIL(
           model=model,
           optimizer=optimizer,
           loss=FocalLoss(
               alpha=alpha,    # Class weight (higher for minority class)
               gamma=2.0,      # Focus parameter
           ),
       )
       
       return lit_model

``complementary_frequencies`` returns ``(frequencies, alpha)`` where ``alpha`` is the weight for the positive class.

Learning Rate Scheduling
-------------------------

Use learning rate schedulers to adjust learning rate during training:

.. code-block:: python

   from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
   
   def lit_model_creator(input_dim: int, use_lr_scheduler: bool = True) -> Pl.LightningModule:
       model = AttentionDeepMIL(embed_dim=input_dim, ...)
       optimizer = AdamW(model.parameters(), lr=1e-4)
       
       # Option 1: Reduce on plateau
       scheduler = ReduceLROnPlateau(
           optimizer, 
           mode="min",       # Minimize validation loss
           patience=5,       # Wait 5 epochs before reducing
           factor=0.8,       # Multiply LR by 0.8
       )
       
       # Option 2: Cosine annealing
       scheduler = CosineAnnealingLR(
           optimizer,
           T_max=50,         # Maximum epochs
           eta_min=1e-6,     # Minimum LR
       )
       
       lit_model = LitAttentionDeepMIL(
           model=model,
           optimizer=optimizer,
           lr_scheduler=scheduler if use_scheduler else None,
       )
       
       return lit_model

What K-Fold Does Internally
============================

The ``KFoldCrossValidation.evaluate()`` method:

1. **Splits data** using ``StratifiedKFold`` to maintain class balance
2. **Creates train/val subsets** using ``dataset.create_train_val_datasets()``
3. **Fits transforms** only on the training subset
4. **Applies transforms** to both train and validation
5. **Fits label transforms** (if provided) on training labels
6. **Creates fresh model** using ``lit_model_creator``
7. **Trains** with early stopping and checkpointing
8. **Evaluates** and logs metrics to wandb
9. **Aggregates** results across all folds
10. **Trains final model** on full dataset with average best epochs
11. **Returns** ``ModelStorage`` object with all results

Next Steps
==========

- :doc:`models` - Explore different model architectures
- :doc:`../explainability` - Interpret model predictions
