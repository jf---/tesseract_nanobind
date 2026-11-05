"""The additive name API keeps old consumers' strings and joint ordering."""

from pathlib import Path

from tesseract_robotics.tesseract_common import GeneralResourceLocator
from tesseract_robotics.tesseract_environment import Environment
from tesseract_robotics.tesseract_scene_graph import Joint

FIXTURES = Path(__file__).parent / "data"


def test_environment_group_and_solver_names_match_scene():
    env = Environment()
    assert env.init(
        FIXTURES / "rail_arm.urdf",
        FIXTURES / "rail_arm.srdf",
        GeneralResourceLocator(),
    )
    active = ["rail", *(f"arm_joint_{index}" for index in range(1, 7))]
    links = {link.getName() for link in env.getSceneGraph().getLinks()}
    joints = {joint.getName() for joint in env.getSceneGraph().getJoints()}
    group = env.getJointGroup("rail_arm")
    solver = env.getStateSolver()
    assert env.getActiveJointNames() == active
    assert group.getJointNames() == active
    assert solver.getActiveJointNames() == active
    assert set(env.getJointNames()) == set(solver.getJointNames()) == joints
    assert (
        set(env.getLinkNames()) == set(solver.getLinkNames()) == set(group.getLinkNames()) == links
    )
    assert (
        set(env.getActiveLinkNames())
        == set(solver.getActiveLinkNames())
        == set(group.getActiveLinkNames())
    )
    assert (
        set(env.getStaticLinkNames())
        == set(solver.getStaticLinkNames())
        == set(group.getStaticLinkNames())
    )
    for obj, methods in (
        (
            env,
            (
                "getJointNames",
                "getActiveJointNames",
                "getLinkNames",
                "getActiveLinkNames",
                "getStaticLinkNames",
            ),
        ),
        (
            solver,
            (
                "getJointNames",
                "getActiveJointNames",
                "getLinkNames",
                "getActiveLinkNames",
                "getStaticLinkNames",
            ),
        ),
        (group, ("getJointNames", "getLinkNames", "getActiveLinkNames", "getStaticLinkNames")),
    ):
        for method in methods:
            assert all(isinstance(name, str) for name in getattr(obj, method)())


def test_joint_parent_and_child_name_properties_are_writable_strings():
    joint = Joint("joint")
    joint.parent_link_name = "parent"
    joint.child_link_name = "child"
    assert joint.parent_link_name == "parent"
    assert joint.child_link_name == "child"
    assert isinstance(joint.parent_link_name, str)
    assert isinstance(joint.child_link_name, str)
