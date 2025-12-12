from typing import List, Any
import pandas as pd
import uuid
from cellmil.utils.templates import get_template_segmentation, get_template_point
from cellmil.interfaces.CellSegmenterConfig import TYPE_NUCLEI_DICT

def convert_geojson(
        cell_list: list[dict[str, Any]], polygons: bool, model: str
    ) -> List[dict[str, Any]]:
        """Convert a list of cells to a geojson object

        Either a segmentation object (polygon) or detection points are converted

        Args:
            cell_list (list[dict]): Cell list with dict entry for each cell.
                Required keys for detection:
                    * type
                    * centroid
                Required keys for segmentation:
                    * type
                    * contour
            polygons (bool, optional): If polygon segmentations (True) or detection points (False). Defaults to False.

        Returns:
            List[dict]: Geojson like list
        """
        if model in ["cellvit", "hovernet"]:
            type_nuclei_dict = TYPE_NUCLEI_DICT
            
            color_dict = {
                1: [255, 0, 0],
                2: [34, 221, 77],
                3: [35, 92, 236],
                4: [254, 255, 0],
                5: [255, 159, 68],
            }
        elif model == "cellpose_sam":
            type_nuclei_dict = {
                1: "cell",
            }
            color_dict = {
                1: [255, 0, 0],
            }
        else:
            raise ValueError(f"Model {model} is not supported for geojson conversion.")
        
        # Handle empty cell list
        if not cell_list:
            return []
        
        if polygons:
            cell_segmentation_df: Any = pd.DataFrame(cell_list)
            detected_types = sorted(cell_segmentation_df.type.unique())
            geojson_placeholder: list[dict[str, Any]] = []
            for cell_type in detected_types:
                cells = cell_segmentation_df[cell_segmentation_df["type"] == cell_type]
                # Skip empty types to avoid QuPath reading issues
                if cells.empty:  # type: ignore
                    continue
                contours: list[Any] = cells["contour"].to_list() # type: ignore
                final_c: list[list[list[list[int]]]] = []
                for c in contours:
                    c.append(c[0])
                    final_c.append([c])

                cell_geojson_object = get_template_segmentation()
                cell_geojson_object["id"] = str(uuid.uuid4())
                # Use Polygon for single cell, MultiPolygon for multiple cells
                if len(final_c) == 1:
                    cell_geojson_object["geometry"]["type"] = "Polygon"
                    cell_geojson_object["geometry"]["coordinates"] = final_c[0]
                else:
                    cell_geojson_object["geometry"]["coordinates"] = final_c
                cell_geojson_object["properties"]["classification"][
                    "name"
                ] = type_nuclei_dict[cell_type]
                cell_geojson_object["properties"]["classification"][
                    "color"
                ] = color_dict[cell_type]
                geojson_placeholder.append(cell_geojson_object)
        else:
            cell_detection_df = pd.DataFrame(cell_list)
            detected_types: Any = sorted(cell_detection_df.type.unique()) # type: ignore
            geojson_placeholder = []
            for cell_type in detected_types:
                cells: pd.DataFrame = cell_detection_df[cell_detection_df["type"] == cell_type] # type: ignore
                # Skip empty types to avoid QuPath reading issues
                if cells.empty:  # type: ignore
                    continue
                centroids: List[Any] = cells["centroid"].to_list() # type: ignore
                cell_geojson_object = get_template_point()
                cell_geojson_object["id"] = str(uuid.uuid4())
                # Use Point for single point, MultiPoint for multiple points
                if len(centroids) == 1:  # type: ignore
                    cell_geojson_object["geometry"]["type"] = "Point"
                    cell_geojson_object["geometry"]["coordinates"] = centroids[0]
                else:
                    cell_geojson_object["geometry"]["coordinates"] = centroids
                cell_geojson_object["properties"]["classification"][
                    "name"
                ] = type_nuclei_dict[cell_type]
                cell_geojson_object["properties"]["classification"][
                    "color"
                ] = color_dict[cell_type]
                geojson_placeholder.append(cell_geojson_object)
        return geojson_placeholder