"""Application contracts for file, XML, and generated joint groups."""

from pathlib import Path

import pytest

from tesseract_robotics.tesseract_common import GeneralResourceLocator
from tesseract_robotics.tesseract_environment import Environment


@pytest.fixture(scope="module")
def fixture_files():
    locator = GeneralResourceLocator()
    prefix = "package://tesseract/support/urdf/abb_irb2400_on_positioner"
    urdf = locator.locateResource(prefix + ".urdf")
    srdf = locator.locateResource(prefix + ".srdf")
    assert urdf is not None and srdf is not None
    return locator, Path(urdf.getFilePath()), Path(srdf.getFilePath())


def test_init_files_and_xml(fixture_files):
    locator, urdf, srdf = fixture_files
    assert Environment().init(urdf, srdf, locator)
    assert Environment().init(urdf, locator)
    assert Environment().init(urdf.read_text(), srdf.read_text(), locator)
    assert Environment().init(urdf.read_text(), locator)
    assert not Environment().init(str(urdf), str(srdf), locator)
    with pytest.raises(TypeError):
        Environment().init(urdf, srdf.read_text(), locator)


def test_generated_joint_group_and_unknown_joint(fixture_files):
    locator, urdf, srdf = fixture_files
    env = Environment()
    assert env.init(urdf, srdf, locator)
    names = env.getActiveJointIds()
    group = env.getJointGroup("generated", names)
    assert group.getName() == "generated"
    assert group.getJointIds() == names
    with pytest.raises(ValueError, match="not_a_joint"):
        env.getJointGroup("invalid", ["not_a_joint"])
