"""The modern native line helper is callable directly from the class."""

import numpy as np

from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics.tesseract_common import Isometry3d


def test_line_point_static_helper():
    start = np.eye(4)
    end = np.eye(4)
    end[0, 3] = 1.0
    source = np.eye(4)
    source[0, 3] = 0.5
    result = ti.CartLineConstraint.getLinePoint(
        Isometry3d(source), Isometry3d(start), Isometry3d(end)
    )
    np.testing.assert_allclose(result.matrix, source)
