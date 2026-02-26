import pandas as pd
from cellmil.interfaces.SHAPExplainerConfig import SHAPExplainerConfig
from cellmil.explainability.shap import SHAPExplainer
from pathlib import Path
from cellmil.datamodels.model import ModelStorage

MODEL_CHECKPOINT = "../../results/OS24+ALL+CLAM+*+*"
DATASET_FOLDER_PATH = "../../dataset"
DP_METADATA = "../../data/metadata_INT.xlsx"

config = SHAPExplainerConfig(
    output_path=Path(f"./INT"),
)

explainer = SHAPExplainer(config)

results = explainer.generate_explanation(
    model_storage=ModelStorage.from_directory(MODEL_CHECKPOINT),
    dataset_folder=Path(DATASET_FOLDER_PATH),
    data=pd.read_excel(DP_METADATA),
)