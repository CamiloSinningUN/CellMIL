import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Any, cast

from cellmil.explainability.core.attention_extractor import AttentionResult
from cellmil.interfaces.ExplainerCreatorConfig import ExplainerCreatorConfig
from cellmil.utils import logger

class AttentionHeatmapVisualizer:
    """Creates heatmap visualizations for attention weights."""

    def __init__(self, config: ExplainerCreatorConfig):
        self.config = config

    def create_visualization(
        self,
        attention_result: AttentionResult,
        output_path: Path,
    ) -> List[Path]:
        """
        Create heatmap visualizations for the given attention result.

        Args:
            attention_result (AttentionResult): The result containing attention weights and related data.
            output_dir (Path): Directory to save the generated heatmap files.

        Returns:
            List[Path]: A list of paths to the generated heatmap files.
        """
        
        logger.info(f"Creating heatmap visualizations in: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        created_files: list[Path] = []

        for (
            attention_key,
            attention_weights,
        ) in attention_result.attention_weights.items():
            try:
                logger.info(f"Creating heatmap for: {attention_key}")
                # Convert to numpy
                attention_np = cast(
                    np.ndarray[Any, Any],
                    attention_weights.cpu().detach().numpy(),  # type: ignore
                )

                logger.info(f"Attention shape: {attention_np.shape}")

                # Create heatmap
                plt.figure(figsize=(12, 8))  # type: ignore

                if len(attention_np.shape) == 2:
                    logger.info("Creating 2D heatmap...")
                    # 2D attention (e.g., [classes, instances] or [heads, instances])
                    sns.heatmap(  # type: ignore
                        attention_np,
                        cmap="Reds",
                        annot=False,
                        cbar_kws={"label": "Attention Weight"},
                    )
                    plt.title(  # type: ignore
                        f"Attention Heatmap: {attention_key.replace('_', ' ').title()}"
                    )

                    if attention_np.shape[0] <= 10:  # Label axes if not too many
                        plt.ylabel("Attention Head/Class")  # type: ignore
                    plt.xlabel("Instance")  # type: ignore

                else:
                    logger.info("Creating 1D bar plot...")
                    # 1D attention - show as bar plot
                    plt.bar(range(len(attention_np)), attention_np)  # type: ignore
                    plt.title(  # type: ignore
                        f"Attention Weights: {attention_key.replace('_', ' ').title()}"
                    )
                    plt.xlabel("Instance")  # type: ignore
                    plt.ylabel("Attention Weight")  # type: ignore

                # Save plot
                plot_path = output_path / f"{attention_key}_heatmap.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # type: ignore
                plt.close()

                created_files.append(plot_path)
                logger.info(f"Saved heatmap: {plot_path}")

            except Exception as e:
                logger.error(f"Error creating heatmap for {attention_key}: {e}")

        logger.info(f"Heatmap creation completed - {len(created_files)} files created")
        return created_files