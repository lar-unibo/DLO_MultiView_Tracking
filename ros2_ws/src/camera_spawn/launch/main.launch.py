#!/usr/bin/env python3

from launch import LaunchDescription, LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    coor_bassa = '"-0.047 0.293 0.17 0.0 0.0 -1.57"'
    coor_alta  = '"-0.047 0.293 0.623 0.0 0.0 -1.57"'
    
    camera_spawner = IncludeLaunchDescription(
                        PythonLaunchDescriptionSource([get_package_share_directory('depthai_ros_driver'), '/launch', '/example_multicam.launch.py']),
                        launch_arguments={'coor_bassa': coor_bassa,
                                          'coor_alta': coor_alta,
                                        }.items())
    
    work_on_cable = Node(
        package="simod_vision",
        executable="workoncable",
        output="screen",
    
        )

    ld = LaunchDescription()


    ld.add_action(camera_spawner)
    #ld.add_action(work_on_cable)
    #ld.add_action(cable_spawner)
    #ld.add_action(canaline_spawner)   #metto direttamente nel sa.world 
    
    return ld