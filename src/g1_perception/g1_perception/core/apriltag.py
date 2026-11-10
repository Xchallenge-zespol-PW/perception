"""AprilTag detection and pose from one image (OpenCV aruco, no ROS)."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from g1_perception.core.geometry import rotation_matrix_to_quat_xyzw
from g1_perception.core.types import Pose

FAMILIES = {
    '16h5': 'DICT_APRILTAG_16h5',
    '25h9': 'DICT_APRILTAG_25h9',
    '36h10': 'DICT_APRILTAG_36h10',
    '36h11': 'DICT_APRILTAG_36h11',
}


@dataclass(frozen=True)
class TagDetection:
    """
    One AprilTag seen in an image.

    ``corners`` are the 4 image corners (px) in aruco order: top-left,
    top-right, bottom-right, bottom-left of the printed tag. ``pose`` is the
    tag frame in the camera frame: origin at the tag centre, x to the right,
    y up and z out of the printed face (up for a tag lying on the floor).
    ``reproj_px`` is the mean corner reprojection error of that pose and
    ``ambiguity`` the ratio of the best to the second planar solution's
    error (near 1 = the tag could also be tilted the other way).
    """

    tag_id: int
    corners: NDArray[np.float64]
    pose: Pose | None
    reproj_px: float
    ambiguity: float


def parse_tag_ids(text: str) -> tuple[int, ...]:
    """Parse ``"0, 3,7"`` into ``(0, 3, 7)``; empty text = no filter."""
    return tuple(int(part) for part in str(text).split(',') if part.strip())


def tag_object_points(size: float) -> NDArray[np.float64]:
    """Return the tag corners in the tag frame, in aruco / IPPE order."""
    h = float(size) / 2.0
    return np.array(
        [[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]],
        dtype=np.float64,
    )


class AprilTagDetector:
    """
    Detect AprilTags with OpenCV aruco and solve each pose with IPPE.

    ``tag_size`` is the side of the black square (m), not of the white
    border. ``tag_ids`` keeps only those ids (empty = all). A pose is
    dropped (``pose`` None, detection kept) when its reprojection error is
    above ``max_reproj_px``.
    """

    def __init__(
        self,
        *,
        family: str = '36h11',
        tag_size: float = 0.1,
        tag_ids: Collection[int] = (),
        max_reproj_px: float = 2.0,
    ) -> None:
        if not hasattr(cv2, 'aruco'):
            raise RuntimeError(
                'cv2.aruco not available: needs OpenCV >= 4.7 or '
                'opencv-contrib-python'
            )
        key = str(family).strip().lower().removeprefix('tag')
        if key not in FAMILIES:
            raise ValueError(
                f'family must be one of {sorted(FAMILIES)}, got {family!r}'
            )
        if tag_size <= 0.0:
            raise ValueError(f'tag_size must be > 0, got {tag_size}')
        self.family = key
        self.tag_size = float(tag_size)
        self.tag_ids = frozenset(int(i) for i in tag_ids)
        self.max_reproj_px = float(max_reproj_px)
        self._object = tag_object_points(self.tag_size)

        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, FAMILIES[key])
        )
        params = (
            cv2.aruco.DetectorParameters()
            if hasattr(cv2.aruco, 'DetectorParameters')
            else cv2.aruco.DetectorParameters_create()  # OpenCV < 4.7
        )
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        if hasattr(cv2.aruco, 'ArucoDetector'):
            detector = cv2.aruco.ArucoDetector(dictionary, params)
            self._detect = detector.detectMarkers
        else:  # OpenCV < 4.7 contrib API
            def _legacy(gray: NDArray[np.uint8]):
                return cv2.aruco.detectMarkers(
                    gray, dictionary, parameters=params
                )
            self._detect = _legacy

    def detect(
        self,
        image: NDArray[np.uint8],
        K: NDArray[np.float64],
        D: NDArray[np.float64] | None = None,
    ) -> list[TagDetection]:
        """Return the tags in ``image`` (RGB or gray) sorted by id."""
        gray = image if image.ndim == 2 else cv2.cvtColor(
            image, cv2.COLOR_RGB2GRAY
        )
        corners, ids, _rejected = self._detect(gray)
        if ids is None:
            return []
        K = np.asarray(K, dtype=np.float64)
        D = np.zeros(5) if D is None else np.asarray(D, dtype=np.float64)

        out: list[TagDetection] = []
        for quad, tag_id in zip(corners, ids.ravel()):
            tag_id = int(tag_id)
            if self.tag_ids and tag_id not in self.tag_ids:
                continue
            quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
            pose, reproj, ambiguity = self._solve(quad, K, D)
            out.append(TagDetection(tag_id, quad, pose, reproj, ambiguity))
        return sorted(out, key=lambda t: t.tag_id)

    def _solve(
        self,
        quad: NDArray[np.float64],
        K: NDArray[np.float64],
        D: NDArray[np.float64],
    ) -> tuple[Pose | None, float, float]:
        """Return ``(pose or None, reproj_px, ambiguity)`` of one tag."""
        n, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            self._object, quad, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if n == 0:
            return None, float('inf'), 1.0
        errors = np.asarray(errors, dtype=np.float64).ravel()
        best = int(np.argmin(errors))
        reproj = float(errors[best])
        ambiguity = (
            reproj / float(errors[1 - best])
            if n > 1 and errors[1 - best] > 0.0 else 0.0
        )
        if reproj > self.max_reproj_px:
            return None, reproj, ambiguity
        R, _ = cv2.Rodrigues(rvecs[best])
        pose = Pose(
            position=np.asarray(tvecs[best], dtype=np.float64).ravel(),
            quat=rotation_matrix_to_quat_xyzw(R),
            score=1.0 / (1.0 + reproj),
            method='apriltag',
        )
        return pose, reproj, ambiguity
