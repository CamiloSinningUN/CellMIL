from enum import Enum

class MILType(str, Enum):
    clam = "clam"
    standard = "standard"
    transmil = "transmil"
    histobistro = "histobistro"
    graphmil = "graphmil"
    ours = "ours"
    attention = "attention"
    head4type = "head4type"
    multifocus = "multifocus"

    @classmethod
    def values(cls):
        return [member.value for member in cls]

    def __str__(self):
        return self.value

class RouteType(str, Enum):
    mil = ["clam", "standard", "transmil", "histobistro", "attention", "ours", "head4type", "multifocus"]
    gnn_mil = ["graphmil"]
    
    def __contains__(self, item: str) -> bool:
        """Check if the item is in the feature extraction type."""
        return item in self.value