from cellmil.interfaces.FeatureExtractorConfig import ExtractorType
from .dataset import split_dataset
from .losses import FocalLoss
import pandas as pd
import lightning as Pl
from typing import Callable
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from cellmil.utils.train.dataset import complementary_frequencies

__all__ = [ "split_dataset", "FocalLoss"]


def get_extractors_from_name(name: str):
    if name == "ALL":
        extractors = [
            ExtractorType.morphometrics,
            ExtractorType.pyradiomics_hed,
            ExtractorType.connectivity,
            ExtractorType.geometric,
        ]
    elif name == "MORPHO":
        extractors = ExtractorType.morphometrics
    elif name == "TOPO":
        extractors = [
            ExtractorType.connectivity,
            ExtractorType.geometric,
        ]
    elif name == "PYRAD":
        extractors = ExtractorType.pyradiomics_hed
    elif name == "RESNET":
        extractors = ExtractorType.resnet50
    elif name == "GIGAPATH":
        extractors = ExtractorType.gigapath
    else:
        raise ValueError(f"Unknown extractor configuration: {name}")
    return extractors

def preprocess_df(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "ADENOvsSQUA":
        df = df[df['HISTOLOGY'].isin(['adenocarcinoma', 'squamous'])] # type: ignore
        df[task] = (df['HISTOLOGY'] == 'adenocarcinoma').astype(int)
    elif task == "PDL1":
        df = df[df["PDL1_HIGH_LOW"].isin(["high", "low"])]  # type: ignore
        df[task] = (df["PDL1_HIGH_LOW"] == "high").astype(int)
    elif task in ["OS6", "DCR", "OS24", "ORR", "CBR"]:
        df = df[df[task].isin([0, 1])] # type: ignore
        df = df.dropna(subset=[task]) # type: ignore
        df[task] = df[task].astype(int) # type: ignore
    elif task in ["OS", "PFS"]:
        df['IO_START'] = pd.to_datetime(df['IO_START']) # type: ignore
        if task == "OS":
            df['Z03_DATE'] = pd.to_datetime(df['Z03_DATE']) # type: ignore
            df['duration'] = (df['Z03_DATE'] - df['IO_START']).dt.days / 30.44 # type: ignore
            df['event'] = df['DEATH_EVENT_OC'].astype(int)
        elif task == "PFS":
            df['PROGRESSION_DATE'] = pd.to_datetime(df['PROGRESSION_DATE']) # type: ignore
            df['duration'] = (df['PROGRESSION_DATE'] - df['IO_START']).dt.days / 30.44 # type: ignore
            df['event'] = df['PROGRESSION_EVENT_OC'].astype(int)
        else:
            raise ValueError(f"Unknown survival task: {task}")
        
        df = df[df['duration'] > 0]
        
    else:
        raise ValueError(f"Unknown task: {task}")
    
    if task not in ["OS", "PFS"]: 
        df = df.dropna(subset=[task]) # type: ignore
    return df

def get_lit_model_creator(model: str, task: str, n_bins: int, feature: str, df: pd.DataFrame, regularization: bool) -> Callable[[int, bool], Pl.LightningModule]:
    is_survival = task in ["OS", "PFS"]
    
    if model == "ABMIL":
        def lit_model_creator(input_dim: int, use_lr_scheduler = True) -> Pl.LightningModule:
            from cellmil.models.mil.attentiondeepmil import AttentionDeepMIL, LitAttentionDeepMIL, LitSurvAttentionDeepMIL
            
            model = AttentionDeepMIL(
                embed_dim=input_dim,
                size_arg=[256, 128] if feature != "RESNET" and feature != "GIGAPATH" else [500, 128],
                n_classes=2 if not is_survival else n_bins,
                attention_branches=8 if feature != "RESNET" and feature != "GIGAPATH" else 1,
                temperature=1.5 if feature != "RESNET" and feature != "GIGAPATH" else 1.0
            )

            optimizer = AdamW(
                model.parameters(), 
                lr=1e-4,
                weight_decay=1e-1 if regularization else 0.0
            )

            print("\nCreating trainer...")
            if is_survival:
                lit_model = LitSurvAttentionDeepMIL(
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.8) if use_lr_scheduler else None,
                    use_aem= True if regularization else False,
                    subsampling=0.8 if regularization else 1.0
                )
            else:
                lit_model = LitAttentionDeepMIL(
                    model=model,
                    optimizer=optimizer,
                    loss=FocalLoss(
                        alpha=complementary_frequencies(df, task)[1], 
                        gamma=2.0
                    ),
                    lr_scheduler=ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.8) if use_lr_scheduler else None,
                    use_aem= True if regularization else False,
                    subsampling=0.8 if regularization else 1.0
                )

            return lit_model
    
    elif model == "HEAD4TYPE":
        if feature in ["RESNET", "GIGAPATH"]:
            raise ValueError("HEAD4TYPE model is not compatible with RESNET or GIGAPATH features.")
        
        def lit_model_creator(input_dim: int, use_lr_scheduler = True) -> Pl.LightningModule:
            from cellmil.models.mil.head4type import Head4Type, LitHead4Type, LitSurvHead4Type
            
            model = Head4Type(
                embed_dim=input_dim, 
                size_arg=[256, 128], 
                n_classes=2 if not is_survival else n_bins, 
                temperature=1.5
            )

            optimizer = AdamW(
                model.parameters(), 
                lr=1e-4,
                weight_decay=1e-1 if regularization else 0.0
            )

            print("\nCreating trainer...")
            if is_survival:
                lit_model = LitSurvHead4Type(
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=ReduceLROnPlateau(
                        optimizer, 
                        mode="min", 
                        patience=5, 
                        factor=0.8
                    ) if use_lr_scheduler else None,
                    use_aem= True if regularization else False,
                    subsampling=0.8 if regularization else 1.0
                )
            else:
                lit_model = LitHead4Type(
                    model=model,
                    optimizer=optimizer,
                    loss=FocalLoss(
                        alpha=complementary_frequencies(df, task)[1], gamma=2.0
                    ),
                    lr_scheduler=ReduceLROnPlateau(
                        optimizer, 
                        mode="min", 
                        patience=5, 
                        factor=0.8
                    ) if use_lr_scheduler else None,
                    use_aem= True if regularization else False,
                    subsampling=0.8 if regularization else 1.0
                )

            return lit_model
        
        
    elif model == "CLAM":
        def lit_model_creator(input_dim: int, use_lr_scheduler = True) -> Pl.LightningModule:
            from cellmil.models.mil.clam import CLAM_SB, LitCLAM, LitSurvCLAM
            
            model = CLAM_SB(
                embed_dim=input_dim,
                size_arg="small", 
                n_classes=2 if not is_survival else n_bins,
                k_sample=8
            )

            optimizer = AdamW(
                model.parameters(), 
                lr=1e-4,
                weight_decay=1e-1 if regularization else 0.0
            )
            
            if is_survival:
                lit_model = LitSurvCLAM(
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        patience=5,
                        factor=0.8
                    ) if use_lr_scheduler else None,
                    use_aem= True if regularization else False,
                    subsampling=0.8 if regularization else 1.0
                )
            else:
                lit_model = LitCLAM(
                    model=model,
                    optimizer=optimizer,
                    loss_slide=FocalLoss(
                        alpha=complementary_frequencies(df, task)[1], 
                        gamma=2.0
                    ),
                    lr_scheduler=ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        patience=5,
                        factor=0.8
                    ) if use_lr_scheduler else None,
                    use_aem= True if regularization else False,
                    subsampling=0.8 if regularization else 1.0
                )

            return lit_model
        
    else:
        raise ValueError(f"Unknown model: {model}")
    
    return lit_model_creator
    
    
    