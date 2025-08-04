import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution


def launch_setup(context, *args, **kwargs):

    depthai_prefix = get_package_share_directory("depthai_ros_driver")
    params_file_bassa = os.path.join(depthai_prefix, "config", "camera_bassa.yaml")
    params_file_alta = os.path.join(depthai_prefix, "config", "camera_alta.yaml")

    coor_bassa = LaunchConfiguration("coor_bassa").perform(context)
    coor_alta = LaunchConfiguration("coor_alta").perform(context)
    cams = ["oak_bassa", "oak_alta"]
    camera_model = ["OAK-1", "OAK-1"]

    coordinates = [coor_bassa, coor_alta]
    params_files = [params_file_bassa, params_file_alta]
    nodes = []

    for cam_name, cam_coordinate in zip(cams, coordinates):
        node = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(depthai_prefix, "launch", "camera.launch.py")),
            launch_arguments={
                "name": cam_name,
                "namespace": cam_name,
                "parent_frame": "world",
                "params_file": params_files[cams.index(cam_name)],
                "cam_pos_x": cam_coordinate.split()[0] + '"',
                "cam_pos_y": '"' + cam_coordinate.split()[1] + '"',
                "cam_pos_z": '"' + cam_coordinate.split()[2] + '"',
                "cam_yaw": '"' + cam_coordinate.split()[5],
                "camera_model": camera_model[cams.index(cam_name)],
            }.items(),
            #   "cam_roll": str(i),
            #   "cam_pitch": str(i),
            #   "cam_roll": str(i),
        )
        nodes.append(node)
    #     i = i + 0.1
    # spatial_rgbd = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(depthai_prefix, "launch", "rgbd_pcl.launch.py")
    #     ),
    #     launch_arguments={
    #         "name": "oak_d_pro",
    #         "parent_frame": "map",
    #         "params_file": params_file,
    #         "cam_pos_y": str(-0.1),
    #     }.items(),
    # )

    # obj_det = Node(
    #     package="depthai_ros_driver",
    #     executable="obj_pub.py",
    #     remappings=[
    #         ("/oak/nn/detections", "/oak_d_pro/nn/detections"),
    #         ("/oak/nn/detection_markers", "/oak_d_pro/nn/detection_markers"),
    #     ],
    # )

    # nodes.append(spatial_rgbd)
    # nodes.append(obj_det)

    cam_alta_optical_frame = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=[
            "0.5970032310550013",
            "-0.5236218571628566",
            "0.7462807227592244",
            "0.67666853",
            "0.68813629",
            "-0.18458932",
            "-0.18578195",
            "ur_right_base_link_inertia",
            "cam_alta_optical_frame",
        ],
    )

    cam_alta_base_frame = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=[
            "0.5970032310550013",
            "-0.5236218571628566",
            "0.7462807227592244",
            "-0.3471092298744374",
            "-0.35606147750526485",
            "0.6171101975914651",
            "0.6098445822078915",
            "ur_right_base_link_inertia",
            "cam_alta_base_frame",
        ],
    )

    cam_bassa_optical_frame = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=[
            "-0.2658697781222589",
            "-0.5185093732589804",
            "0.23820403037687854",
            "-0.50578571",
            "0.50414087",
            "-0.49734753",
            "0.49261368",
            "ur_right_base_link_inertia",
            "cam_bassa_optical_frame",
        ],
    )

    cam_bassa_base_frame = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=[
            "-0.2658697781222589",
            "-0.5185093732589804",
            "0.23820403037687854",
            "-0.00931403174565598",
            "0.004803616786405307",
            "-0.708159239730411",
            "0.7059749781932678",
            "ur_right_base_link_inertia",
            "cam_bassa_base_frame",
        ],
    )

    nodes.append(cam_alta_optical_frame)
    nodes.append(cam_alta_base_frame)
    nodes.append(cam_bassa_optical_frame)
    nodes.append(cam_bassa_base_frame)

    return nodes


def generate_launch_description():

    coor_bassa = DeclareLaunchArgument("coor_bassa", default_value=TextSubstitution(text="0 0 0 0 0 0"))
    coor_alta = DeclareLaunchArgument("coor_alta", default_value=TextSubstitution(text="0 0 0 0 0 0"))

    return LaunchDescription(
        [
            coor_bassa,
            coor_alta,
            OpaqueFunction(function=launch_setup),
        ]
    )
