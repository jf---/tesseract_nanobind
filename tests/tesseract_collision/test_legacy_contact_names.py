"""Contact link-name compatibility preserves the original string-pair contract."""

import pytest

from tesseract_robotics.tesseract_collision import (
    ContactRequest,
    ContactResult,
    ContactResultMap,
    ContactResultVector,
    ContactTestType,
)
from tesseract_robotics.tesseract_common import GeneralResourceLocator
from tesseract_robotics.tesseract_environment import Environment

URDF = """<robot name="contact_pair" xmlns:tesseract="http://www.tesseract.com" tesseract:make_convex="true">
  <link name="base"><collision><geometry><sphere radius="0.5"/></geometry></collision></link>
  <link name="arm"><collision><geometry><sphere radius="0.5"/></geometry></collision></link>
  <joint name="joint" type="revolute"><parent link="base"/><child link="arm"/>
    <origin xyz="0.25 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>"""
SRDF = (
    '<robot name="contact_pair"><contact_managers_plugin_config '
    'filename="package://tesseract/support/urdf/contact_manager_plugins.yaml"/></robot>'
)


def test_contact_name_pair_copy_assignment_and_length():
    contact = ContactResult()
    assert contact.link_names == ["", ""]
    detached = contact.link_names
    detached[0] = "not-written"
    assert contact.link_names == ["", ""]
    contact.link_names = ("first", "second")
    assert contact.link_names == ["first", "second"]
    with pytest.raises(TypeError):
        contact.link_names = ["only-one"]


def test_real_contact_returns_string_link_names(tmp_path):
    urdf, srdf = tmp_path / "cell.urdf", tmp_path / "cell.srdf"
    urdf.write_text(URDF)
    srdf.write_text(SRDF)
    env = Environment()
    assert env.init(urdf, srdf, GeneralResourceLocator())
    manager = env.getDiscreteContactManager()
    manager.setActiveCollisionObjects(env.getActiveLinkNames())
    manager.setCollisionObjectsTransform(env.getState().link_transforms)
    contacts = ContactResultMap()
    manager.contactTest(contacts, ContactRequest(ContactTestType.ALL))
    flat = ContactResultVector()
    contacts.flattenMoveResults(flat)
    pair = next(contact for contact in flat if contact.distance < 0)
    assert set(pair.link_names) == {"base", "arm"}
    assert all(isinstance(name, str) for name in pair.link_names)
