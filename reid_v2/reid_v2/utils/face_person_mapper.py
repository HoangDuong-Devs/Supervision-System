# face_person_mapper.py

"""
Face to Person Mapping
"""

import logging
from typing import Any, Dict, List

from .geometry import bbox_to_coords_tuple

logger = logging.getLogger(__name__)


def map_faces_to_persons(face_data: List[Dict[str, Any]], person_data: List[Dict[str, Any]]) -> Dict[str, tuple]:
    """
    Map faces to persons based on spatial proximity
    Returns dict of {person_id: (face_id, confidence)}
    """
    face_person_map: Dict[str, tuple] = {}

    def _parse_bbox(bbox) -> tuple:
        """Parse bbox to (x1, y1, x2, y2) format (float)"""
        # Dùng centralized bbox_utils, nhưng return (float) tuple thay vì (int)
        coords = bbox_to_coords_tuple(bbox)
        if coords is None:
            return None
        return tuple(float(x) for x in coords)

    def _is_face_inside_person(face_bbox, person_bbox):
        """Return True when face bbox is fully inside person bbox AND face center is in upper half of person bbox."""
        try:
            face_coords = _parse_bbox(face_bbox)
            person_coords = _parse_bbox(person_bbox)
            
            if face_coords is None or person_coords is None:

                return False
                
            fx1, fy1, fx2, fy2 = face_coords
            px1, py1, px2, py2 = person_coords
            
            # Ensure all coordinates are numbers
            if not all(isinstance(coord, (int, float)) for coord in [fx1, fy1, fx2, fy2, px1, py1, px2, py2]):
                logger.warning("Non-numeric coordinates: face=%s person=%s", face_coords, person_coords)
                return False
                
        except Exception as exc:
            logger.warning("Error parsing bbox coordinates: %s (face_bbox=%s, person_bbox=%s)", exc, face_bbox, person_bbox)
            return False

        # Logic 1: face bbox fully inside person bbox
        face_in_person = (fx1 >= px1 and fy1 >= py1 and fx2 <= px2 and fy2 <= py2)

        # Logic 2: face center in upper half of person bbox
        face_cx = (fx1 + fx2) / 2.0
        face_cy = (fy1 + fy2) / 2.0
        person_top = py1
        person_bottom = py2
        person_mid_y = person_top + 0.5 * (person_bottom - person_top)
        center_in_upper_half = (px1 <= face_cx <= px2) and (person_top <= face_cy <= person_mid_y)

        result = face_in_person and center_in_upper_half
        return result

    # Build mappings: faces -> persons and persons -> faces using indices
    faces_to_persons = [[] for _ in range(len(face_data))]
    persons_to_faces = [[] for _ in range(len(person_data))]

    for p_idx, person in enumerate(person_data):
        person_bbox = person.get("bbox", [])
        for f_idx, face in enumerate(face_data):
            face_bbox = face.get("bbox", [])
            if _is_face_inside_person(face_bbox, person_bbox):
                faces_to_persons[f_idx].append(p_idx)
                persons_to_faces[p_idx].append(f_idx)

    # Only accept one-to-one mappings
    for p_idx, person in enumerate(person_data):
        if len(persons_to_faces[p_idx]) == 1:
            f_idx = persons_to_faces[p_idx][0]
            if len(faces_to_persons[f_idx]) == 1 and faces_to_persons[f_idx][0] == p_idx:
                person_id = person["id"]
                face_id = face_data[f_idx]["id"]
                face_person_map[person_id] = (face_id, 0.9)

    return face_person_map