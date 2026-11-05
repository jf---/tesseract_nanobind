"""Name compatibility leaves modern ID-valued properties intact."""

from tesseract_robotics.tesseract_collision import ContactResult
from tesseract_robotics.tesseract_common import LinkId
from tesseract_robotics.tesseract_scene_graph import Joint


def test_joint_id_and_name_views_share_writable_state():
    joint = Joint("joint")
    joint.parent_link_name = "parent"
    assert isinstance(joint.parent_link_id, LinkId)
    assert str(joint.parent_link_id) == "parent"
    joint.child_link_id = LinkId("child")
    assert joint.child_link_name == "child"


def test_contact_id_and_name_views_share_writable_state():
    contact = ContactResult()
    contact.link_names = ["first", "second"]
    assert [str(link) for link in contact.link_ids] == ["first", "second"]
    contact.link_ids = [LinkId("left"), LinkId("right")]
    assert contact.link_names == ["left", "right"]
