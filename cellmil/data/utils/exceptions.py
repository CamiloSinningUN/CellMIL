# -*- coding: utf-8 -*-
# Exceptions
#
# References:
# CellViT: Vision Transformers for precise cell segmentation and classification
# Fabian Hörst et al., Medical Image Analysis, 2024
# DOI: https://doi.org/10.1016/j.media.2024.103143


class WrongParameterException(Exception):
    """
    Exceptions that occur when the given parameters are not supported.
    """

    pass


class OverwriteException(WrongParameterException):
    """
    Exceptions that occur when the data may be overwritten but there is a missing parameter.
    """

    pass


class UnalignedDataException(Exception):
    """
    Exceptions that occur when the given data is not aligned.
    """

    pass
