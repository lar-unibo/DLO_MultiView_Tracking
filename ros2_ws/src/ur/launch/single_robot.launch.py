#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch import LaunchDescription
import launch_ros.parameter_descriptions
from launch.actions import OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():
    namespace = DeclareLaunchArgument('namespace')
    ur_type = DeclareLaunchArgument('ur_type', choices=["ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e", "ur20"])
    safety_limits = DeclareLaunchArgument('safety_limits', default_value="true")
    safety_pos_margin = DeclareLaunchArgument('safety_pos_margin', default_value="0.15")
    safety_k_position = DeclareLaunchArgument('safety_k_position', default_value="20")
    description_package = DeclareLaunchArgument('description_package', default_value="ur")
    description_file = DeclareLaunchArgument('description_file', default_value="ur_spawn.urdf.xacro")
    tf_prefix = DeclareLaunchArgument('tf_prefix', default_value='""')
    base_prefix = DeclareLaunchArgument('base_prefix', default_value='""')
    xyz = DeclareLaunchArgument('xyz', default_value=TextSubstitution(text="0 0 0"))
    rpy_base_ur = DeclareLaunchArgument('rpy_base_ur', default_value=TextSubstitution(text="0 0 0"))
    xyz_base = DeclareLaunchArgument('xyz_base', default_value=TextSubstitution(text="0 0 0"))
    parent_ur = DeclareLaunchArgument('parent_ur')
    parent_hand = DeclareLaunchArgument('parent_hand')
    rpy = DeclareLaunchArgument('rpy', default_value=TextSubstitution(text="0 0 0"))
    finger_tip_cor = DeclareLaunchArgument('finger_tip_cor', default_value=TextSubstitution(text="0 0 0"))
    robot_ip = DeclareLaunchArgument('robot_ip', default_value="192.168.0.102")
    use_tool_communication = DeclareLaunchArgument('use_tool_communication', default_value="true")
    tool_tcp_port = DeclareLaunchArgument('tool_tcp_port', default_value="54321")
    tool_device_name = DeclareLaunchArgument('tool_device_name', default_value="/tmp/ttyUR")
    spawn_gazebo_robot = DeclareLaunchArgument('spawn_gazebo_robot')
    initial_positions_file = DeclareLaunchArgument('initial_positions_file', default_value="")
    physical_parameters_file = DeclareLaunchArgument('physical_parameters_file', default_value="")
    controller_file = DeclareLaunchArgument('controller_file', default_value="")
    vel_controller = DeclareLaunchArgument('vel_controller', default_value="")
    spawn_gazebo_base = DeclareLaunchArgument('spawn_gazebo_base', default_value="")
    parent_base = DeclareLaunchArgument('parent_base', default_value="")
    script_sender_port = DeclareLaunchArgument('script_sender_port', default_value="")
    reverse_port = DeclareLaunchArgument('reverse_port', default_value="30003")
    script_command_port = DeclareLaunchArgument('script_command_port', default_value="30001")
    trajectory_port = DeclareLaunchArgument('trajectory_port', default_value="30002")


    return LaunchDescription(

        [namespace, ur_type, safety_limits, safety_pos_margin, safety_k_position, description_package, description_file, tf_prefix,base_prefix, xyz,rpy_base_ur,xyz_base, parent_ur, parent_hand, 
         rpy, finger_tip_cor, robot_ip, use_tool_communication, tool_tcp_port, tool_device_name, spawn_gazebo_robot, initial_positions_file, 
         physical_parameters_file, controller_file,vel_controller,spawn_gazebo_base, parent_base, script_sender_port, reverse_port, script_command_port, trajectory_port,
        OpaqueFunction(function=launch_setup) ]
        ) 
    

def launch_setup(context, *args, **kwargs):

    # Initialize Arguments Left
    namespace              = LaunchConfiguration("namespace").perform(context)
    ur_type                = LaunchConfiguration("ur_type").perform(context)
    tf_prefix              = LaunchConfiguration("tf_prefix").perform(context)
    base_prefix            = LaunchConfiguration("base_prefix").perform(context)
    xyz                    = LaunchConfiguration('xyz').perform(context)
    rpy_base_ur            = LaunchConfiguration('rpy_base_ur').perform(context)
    xyz_base               = LaunchConfiguration('xyz_base').perform(context)
    parent_hand            = LaunchConfiguration("parent_hand").perform(context)
    parent_ur              = LaunchConfiguration("parent_ur").perform(context)
    parent_base            = LaunchConfiguration("parent_base").perform(context)
    rpy                    = LaunchConfiguration('rpy').perform(context)
    finger_tip_cor         = LaunchConfiguration('finger_tip_cor').perform(context)
    robot_ip               = LaunchConfiguration("robot_ip").perform(context)
    initial_positions_file = LaunchConfiguration("initial_positions_file").perform(context)
    physical_parameters_file = LaunchConfiguration("physical_parameters_file").perform(context)
    controller_file        = LaunchConfiguration("controller_file").perform(context)
    vel_controller         = LaunchConfiguration("vel_controller").perform(context)

    spawn_gazebo_base         = LaunchConfiguration("spawn_gazebo_base").perform(context)
    spawn_gazebo_robot        = LaunchConfiguration("spawn_gazebo_robot").perform(context)

    safety_limits       = LaunchConfiguration("safety_limits").perform(context)
    safety_pos_margin   = LaunchConfiguration("safety_pos_margin").perform(context)
    safety_k_position   = LaunchConfiguration("safety_k_position").perform(context)

    # General arguments
    description_package = LaunchConfiguration("description_package").perform(context)
    description_file    = LaunchConfiguration("description_file").perform(context)
    script_sender_port  = LaunchConfiguration("script_sender_port").perform(context)
    reverse_port        = LaunchConfiguration("reverse_port").perform(context)
    script_command_port = LaunchConfiguration("script_command_port").perform(context)
    trajectory_port     = LaunchConfiguration("trajectory_port").perform(context)

    script_filename = PathJoinSubstitution(
        [FindPackageShare("ur_client_library"), "resources", "external_control.urscript"]
    )

    input_recipe_filename = PathJoinSubstitution(
        [FindPackageShare("ur_robot_driver"), "resources", "rtde_input_recipe.txt"]
    )
    output_recipe_filename = PathJoinSubstitution(
        [FindPackageShare("ur_robot_driver"), "resources", "rtde_output_recipe.txt"]
    )

    namespaced_node_name = ['/', namespace, '/controller_manager']
    namespaced_robot_description = ['/', namespace, '/robot_description']

    control_params_file = PathJoinSubstitution(
        [FindPackageShare("ur"), 'config', controller_file])

    

    #Robot description
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare(description_package), "xacro", description_file]),
            " ",
            "namespace:=",
            namespace,
            " ",
            "robot_ip:=",
            robot_ip,
            " ",
            "safety_limits:=",
            safety_limits,
            " ",
            "safety_pos_margin:=",
            safety_pos_margin,
            " ",
            "safety_k_position:=",
            safety_k_position,
            " ",
            "ur_type:=",
            ur_type,
            " ",
            "tf_prefix:=",
            tf_prefix,
            " ",
            "base_prefix:=",
            base_prefix,
            " ",
            "xyz:=",
            xyz,
            " ",
            "rpy_base_ur:=",
            rpy_base_ur,
            " ",
            "xyz_base:=",
            xyz_base,
            " ",
            "parent_ur:=",
            parent_ur,
            " ",
            "parent_hand:=",
            parent_hand,
            " ",
            "rpy:=",
            rpy,
            " ",
            "finger_tip_cor:=",
            finger_tip_cor,
             " ",
            "script_filename:=",
            script_filename,
            " ",
            "input_recipe_filename:=",
            input_recipe_filename,
            " ",
            "output_recipe_filename:=",
            output_recipe_filename,
            " ",
            "initial_positions_file:=",
            initial_positions_file,
            " ",
            "physical_parameters:=",
            physical_parameters_file,
            " ",
            "controller_name:=",
            controller_file,  
            " ",
            "spawn_gazebo_base:=",
            spawn_gazebo_base,  
            " ",
            "parent_base:=",
            parent_base,  
            " ",
            "script_sender_port:=",
            script_sender_port,
            " ",
            "reverse_port:=",
            reverse_port,
            " ",
            "script_command_port:=",
            script_command_port,
            " ",
            "trajectory_port:=",
            trajectory_port,
            " ",
            "sim_gazebo:=",
            spawn_gazebo_robot,
            " ",
        ]
    )

    robot_description_param = launch_ros.descriptions.ParameterValue(robot_description_content, value_type=str)

    # Publish robot state
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        namespace=namespace,
        # remappings=remappings,
        output="both",
        parameters=[{'use_sim_time': True},{'robot_description': robot_description_param}],
    )

    # Spawn from topic                                                      
    gazebo_spawn_robot_description = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name=namespace,
        output='screen',
        arguments=[
                    '-entity', namespace + '_robot',
                    '-topic', namespaced_robot_description,
                    '-timeout', '5',
                    '-unpause'
                    ],
        condition=IfCondition(LaunchConfiguration('spawn_gazebo_robot'))
    )


    velocity_controller = Node(
        package='controller_manager',
        executable='spawner',
        namespace=namespace,
        # parameters=[control_params_file],
        arguments=[vel_controller, '-c', namespaced_node_name],
        output='screen',
    )

    
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        #namespace=namespace,
        #parameters=[{'use_local_topics': False}],
        arguments=['joint_state_broadcaster', '-c', namespaced_node_name, '-n', namespace],
        output='screen',
        
    )

    real_joint_controllers = PathJoinSubstitution(
        [FindPackageShare("ur"), "config", controller_file]
    )

    ur_control_node = Node(
        package="ur_robot_driver",
        executable="ur_ros2_control_node",
        namespace=namespace,
        parameters=[
            {'robot_description': robot_description_param},
            launch_ros.parameter_descriptions.ParameterFile(real_joint_controllers, allow_substs=True),
        ],
        # output="screen",
        condition=UnlessCondition(LaunchConfiguration('spawn_gazebo_robot')),
    )

 
    return [
        robot_state_publisher_node,
        gazebo_spawn_robot_description,
        ur_control_node,
        #joint_state_publisher,
        # joint_state_publisher_,
        joint_state_broadcaster_spawner,
        velocity_controller,
        # robot_state_publisher_node_,
    ]

    