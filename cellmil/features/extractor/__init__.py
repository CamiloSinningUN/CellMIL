from cellmil.interfaces.FeatureExtractorConfig import FeatureExtractionType, ExtractorType
from .embedding import EmbeddingExtractor
from .morphological import MorphologicalExtractor
from .topological import TopologicalExtractor

class Extractor:
    @staticmethod
    def create(extractor_type: ExtractorType):
        if extractor_type in FeatureExtractionType.Morphological:
            return MorphologicalExtractor(extractor_name=extractor_type)
        elif extractor_type in FeatureExtractionType.Topological:
            return TopologicalExtractor(extractor_name=extractor_type)
        elif extractor_type in FeatureExtractionType.Embedding:
            return EmbeddingExtractor(extractor_name=extractor_type)
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")