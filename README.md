## 1. Building

Build the Docker container with

```shell
docker compose -f docker-compose-gui.yml build
```

## 2. Starting Container

Allow the container to display contents on your host machine by typing

```bash
xhost +local:root
```

IMPORTANT: Modify the path of the current directory in the docker-compose-gui.yml
```shell
- type: bind
source: /home/lar/dev24/simod_vision
```

```shell

Inside the docker folder, start the container
```shell
docker compose -f docker-compose-gui.yml up
```

Terminator GUI will open up with the Docker container running inside it and the ros2 environment already sourced.


## 3. Prepare Setup

First time, you need to install the dependencies ```dlo_python``` and ```pipy``` by running the following command:

```shell
cd /docker_camere/ros2_ws/src/pipy && pip install -e .
cd /docker_camere/ros2_ws/src/dlo_python && pip install -e .
```

Now everything is ready!!!


## 4. How to Run

The system requires two calibrated cameras. In this environment we use two OAK-1 cameras (camera_alta.yaml and camera_bassa.yaml inside depthai_ros_driver/config). To start the camera system:

```shell
ros2 launch camera_spawn main.launch
```

The system requires a dual-arm robot setup. In this environment we use two UR5 robots.


```shell
ros2 launch ur main.launch
```

The package ```simod_vision``` contains the actual implementation of the paper approach. 

The ```feedback_camera``` node manages the extract of the current DLO state from each camera view based on the predicted state from the DLO model. 

```shell
ros2 run simod_vision feedback_camera.py
```

The ```init_dlo_shape``` node is exploited just for the very first frame in order to provide an initialization of the DLO state via the PyElastic simulator.
```shell
ros2 run simod_vision init_dlo_shape.py
```

The ```The ```init_dlo_shape``` node is the core of the approach. It gather the data from the feedback camera nodes and compute the output 3D state of the DLO.

```shell
ros2 run simod_vision work_on_cable.py
```




--------------------------------------------------------------
## Nvidia Support

Install
```shell
sudo apt install nvidia-container-toolkit
```
Reconfigure
```shell
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
## Docker container same machine
source: https://stackoverflow.com/questions/65900201/troubles-communicating-with-ros2-node-in-docker-container

Using --net=host implies both DDS participants believe they are in the same machine and they try to communicate using SharedMemory instead of UDP.
The solution is to enable SharedMemory between host and container. For this you should share /dev/shm, for example:
```shell
docker run -ti --net host -v /dev/shm:/dev/shm <DOCKER_IMAGE>
```
Also, both applications should be run with the same UID. Docker container's user is usually root (UID=0). Then the host application should be run as root.



# Contact

Email: alessio [dot] caporali [at] unibo.it


# Citation
If you find this work useful, please consider citing:

```
@ARTICLE{caporali2025robotic,
  author={Caporali, Alessio and Palli, Gianluca},
  journal={IEEE/ASME Transactions on Mechatronics}, 
  title={Robotic Manipulation of Deformable Linear Objects via Multiview Model-Based Visual Tracking}, 
  year={2025},
  doi={10.1109/TMECH.2025.3562295}
  }
```

