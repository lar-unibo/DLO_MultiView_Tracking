from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch import LaunchDescription




def generate_launch_description():
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "gripper_ip_left",
            default_value="192.168.0.102",
            description="IP address by which the robot can be reached.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "topic_name_left",
            default_value="ur_left/robotiq_hand_e",
            description="Start robot with fake hardware mirroring command to its states.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "gripper_ip_right",
            default_value="192.168.0.103",
            description="IP address by which the robot can be reached.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "topic_name_right",
            default_value="ur_right/robotiq_hand_e",
            description="Start robot with fake hardware mirroring command to its states.",
        )
    )

    gripper_ip_left = LaunchConfiguration('gripper_ip_left')
    topic_name_left = LaunchConfiguration('topic_name_left')
    gripper_ip_right = LaunchConfiguration('gripper_ip_right')
    topic_name_right= LaunchConfiguration('topic_name_right')

    gripper_controller_right = Node(
            package='robotiq_control',
            executable='service',
            name='robotiq_control_right',
            output='screen',
            arguments =[
               '-gripper_ip', gripper_ip_right,
               '-topic_name', topic_name_right,
            ],
   )
    
    gripper_controller_left = Node(
            package='robotiq_control',
            executable='service',
            name='robotiq_control_left',
            output='screen',
            arguments =[
               '-gripper_ip', gripper_ip_left,
               '-topic_name', topic_name_left,
            ],
   )

    # Creation of the LaunchDescription
    ld = LaunchDescription()

    ld.add_action(declared_arguments[0])
    ld.add_action(declared_arguments[1])
    ld.add_action(declared_arguments[2])
    ld.add_action(declared_arguments[3])

    # Add cmd to the launch description
    ld.add_action(gripper_controller_right)   
    ld.add_action(gripper_controller_left)   

    return ld



 
