from cellmil.interfaces.AttentionExplainerConfig import (
    AttentionExplainerConfig,
    VisualizationMode,
    Normalization
)
from cellmil.datamodels.model import ModelStorage
from cellmil.explainability.attention.attention_explainer import AttentionExplainer
from pathlib import Path

MODEL_CHECKPOINT = "../../results/PFS+ALL+ABMIL+*+*"
SLIDE_PATH = "../../dataset/SLIDE_1"

config = AttentionExplainerConfig(
    output_path=Path(f"./{Path(SLIDE_PATH).name}"),
    visualization_mode=VisualizationMode.graph,
    normalization=Normalization.min_max,
)

explainer = AttentionExplainer(config)

results = explainer.generate_explanation(
    model_storage=ModelStorage.from_directory(MODEL_CHECKPOINT),
    slide_path=Path(SLIDE_PATH),
)