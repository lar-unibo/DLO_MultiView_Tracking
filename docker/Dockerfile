FROM ros:humble-perception

ENV WS_DIR="/ros2_ws"
WORKDIR ${WS_DIR}

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y \
    build-essential \
    cmake \
    git-all \
    software-properties-common \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && apt-get install -y \
    ros-${ROS_DISTRO}-librealsense2* \
    ros-${ROS_DISTRO}-realsense2-* \
    ros-${ROS_DISTRO}-pcl-ros \
    ros-${ROS_DISTRO}-vision-msgs \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-rqt-image-view \
    ros-${ROS_DISTRO}-ur-client-library \
    ros-${ROS_DISTRO}-joint-trajectory-controller \
    ros-${ROS_DISTRO}-gazebo-ros2-control \
    ros-${ROS_DISTRO}-ros2-controllers \
    python3-opencv \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && apt-get install -y \
    ros-${ROS_DISTRO}-rviz2 \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
   && apt-get install terminator wget -y

RUN wget -qO- https://docs.luxonis.com/install_dependencies.sh | bash

RUN sudo apt install -y ros-${ROS_DISTRO}-depthai-ros
RUN . /opt/ros/${ROS_DISTRO}/setup.bash

RUN pip install \  
      pyelastica \ 
      colorama \
      arrow \
      torch==2.5 \
      depthai \
      pymeshfix  \
      trimesh \
      mesh_to_sdf \

RUN pip install \  
      torch_geometric==2.6.1 \
      torch-cluster==1.6.3

   
ARG DEBIAN_FRONTEND=dialog
