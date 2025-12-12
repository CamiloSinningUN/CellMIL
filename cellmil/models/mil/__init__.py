from .clam import LitCLAM
from .standard import LitStandard
# from .transmil import TransMIL
# from .histobistro import HistoBistro
from .graphmil import LitGraphMIL
from .attentiondeepmil import LitAttentionDeepMIL

__all__ = [
    "LitStandard",
    "LitGraphMIL",
    "LitCLAM",
    "LitAttentionDeepMIL"
]
