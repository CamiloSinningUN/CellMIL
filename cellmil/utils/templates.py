# -*- coding: utf-8 -*-
# Templates for GeoJSON objects
#
# References:
# CellViT: Vision Transformers for precise cell segmentation and classification
# Fabian Hörst et al., Medical Image Analysis, 2024
# DOI: https://doi.org/10.1016/j.media.2024.103143

from typing import Any

def get_template_segmentation() -> dict[str, Any]:
    """Return a template for a MultiPolygon geojson object

    Returns:
        dict: Template
    """
    template_multipolygon: dict[str, Any] = {
        "type": "Feature",
        "id": "TODO",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [],
            ],
        },
        "properties": {
            "objectType": "annotation",
            "classification": {"name": "TODO", "color": []},
        },
    }
    return template_multipolygon

def get_template_point() -> dict[str, Any]:
    """Return a template for a Point geojson object

    Returns:
        dict: Template
    """
    template_point: dict[str, Any] = {
        "type": "Feature",
        "id": "TODO",
        "geometry": {
            "type": "MultiPoint",
            "coordinates": [
                [],
            ],
        },
        "properties": {
            "objectType": "annotation",
            "classification": {"name": "TODO", "color": []},
        },
    }
    return template_point