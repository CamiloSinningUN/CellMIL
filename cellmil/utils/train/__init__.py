import lightning as Pl
import numpy as np
import time
import wandb
from pathlib import Path
from lightning import Trainer
from typing import cast, Any, Union, Callable
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split  # type: ignore

from torch.utils.data import DataLoader as DataLoaderTorch
from torch_geometric.loader import DataLoader as DataLoaderPyG  # type: ignore
from cellmil.utils import logger

from cellmil.datamodels.datasets.cell_mil_dataset import CellMILDataset
from cellmil.datamodels.datasets.cell_gnn_mil_dataset import (
    CellGNNMILDataset,
    SubsetCellGNNMILDataset,
)
from cellmil.datamodels.datasets.patch_gnn_mil_dataset import (
    PatchGNNMILDataset,
    SubsetPatchGNNMILDataset,
)
from cellmil.datamodels.datasets.patch_mil_dataset import PatchMILDataset
from cellmil.datamodels.transforms import Transform, TransformPipeline

from .evals import get_report, plot_sample_features
from .dataset import split_dataset
from .losses import FocalLoss

__all__ = ["get_report", "split_dataset", "FocalLoss"]


def train_eval(
    name: str,
    lit_model_creator: Callable[[int], Pl.LightningModule],
    dataset: Union[
        CellMILDataset, CellGNNMILDataset, PatchGNNMILDataset, PatchMILDataset
    ],
    transforms: Union[Transform, TransformPipeline, None] = None,
    train_size: float = 0.8,
    random_state: int = 42,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Train and evaluate a model using PyTorch Lightning.
    Args:
        name (str): Name of the experiment/model.
        lit_model_creator (Callable[[int], Pl.LightningModule]): A callable that takes the number of classes
            as input and returns a PyTorch Lightning module.
        dataset (Union[CellMILDataset, CellGNNMILDataset, PatchGNNMILDataset, PatchMILDataset]):
            The dataset to be used for training and evaluation.
        transforms (Union[Transform, TransformPipeline, None], optional): Transformations to be applied to the dataset.
            Defaults to None.
        train_size (float, optional): Proportion of dataset to use for training. Defaults to 0.8.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        debug (bool, optional): If True, enables debug mode with additional logging. Defaults to False.
    Returns:
        dict[str, Any]: A dictionary containing training and evaluation results.
    """
    # Extract labels for stratified split
    y = np.array([dataset.labels[slide] for slide in dataset.slides])
    indices = np.arange(len(dataset))

    logger.info(f"Starting training with train_size={train_size}...")

    # Split dataset into train and test indices using stratified split
    train_idx, test_idx = cast(
        tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
        train_test_split(
            indices,
            train_size=train_size,
            random_state=random_state,
            stratify=y,
        ),
    )

    logger.info(f"Train indices: {len(train_idx)}, Test indices: {len(test_idx)}")

    # Create train and test datasets with proper transform fitting
    train_dataset, test_dataset = dataset.create_train_val_datasets(  # type: ignore
        train_indices=train_idx.tolist(),
        val_indices=test_idx.tolist(),
        transforms=transforms,
    )

    # Create dataloaders based on dataset type
    if isinstance(dataset, (CellGNNMILDataset, PatchGNNMILDataset)):
        train_loader = DataLoaderPyG(
            train_dataset,  # type: ignore
            batch_size=1,
            shuffle=True,
            num_workers=8
        )
        test_loader = DataLoaderPyG(
            test_dataset,  # type: ignore
            batch_size=1,
            shuffle=False,
            num_workers=8
        )
    else:
        train_loader = DataLoaderTorch(
            train_dataset,  # type: ignore
            batch_size=1,
            shuffle=True,
            num_workers=8
        )
        test_loader = DataLoaderTorch(
            test_dataset,  # type: ignore
            batch_size=1,
            shuffle=False,
            num_workers=8
        )

    # Plot sample features if in debug mode
    if debug:
        logger.info("Debug mode enabled - plotting sample features")
        plot_sample_features(train_dataset, test_dataset, name)
        input("Press Enter to continue...")

    # Create model instance
    if isinstance(
        dataset,
        (
            CellGNNMILDataset,
            SubsetCellGNNMILDataset,
            PatchGNNMILDataset,
            SubsetPatchGNNMILDataset,
        ),
    ):
        model = lit_model_creator(train_dataset[0].x.shape[-1])  # type: ignore
    else:
        model = lit_model_creator(train_dataset[0][0].shape[-1])  # type: ignore

    # Setup trainer with checkpoint callback
    early_stopping = EarlyStopping(monitor="val/total_loss", patience=10, mode="min")

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(f"./checkpoints/{name}"),
        filename="best_model",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
    )

    # Model training
    wandb.login()

    wandb_logger = WandbLogger(
        project="CELLMIL" if not debug else "(TEST) CELLMIL",
        name=f"{name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
    )

    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=[0],
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        accumulate_grad_batches=8,
    )

    # Train the model
    trainer.fit(model, train_loader, test_loader)

    # Get evaluation report
    report = get_report(trainer, model, test_loader)
    wandb.log(report)

    # Save the transforms
    checkpoint_dir = Path(f"./checkpoints/{name}")
    if (
        transforms is not None
        and hasattr(train_dataset, "transforms")
        and isinstance(
            train_dataset, (CellMILDataset, CellGNNMILDataset, SubsetCellGNNMILDataset)
        )
    ):
        # Get the fitted transforms from the train dataset if available
        fitted_transforms = train_dataset.transforms

        if isinstance(fitted_transforms, TransformPipeline):
            transforms_path = checkpoint_dir / "transforms"
            fitted_transforms.save(transforms_path)
            logger.info(f"Saved transforms to: {transforms_path}")
        elif isinstance(fitted_transforms, Transform):
            transforms_path = checkpoint_dir / "transform.json"
            fitted_transforms.save(transforms_path)
            logger.info(f"Saved transform to: {transforms_path}")

    logger.info("Training completed successfully")
    wandb.finish()

    return report
