import rclpy
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from rclpy.executors import MultiThreadedExecutor
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

import os
import cv2
import argparse
from colorama import Fore
import numpy as np
import threading
from threading import Condition

from simod_vision_msgs.msg import Data
from simod_vision.bbox import BboxTcp
from simod_vision.points import Points2DFromModel
from simod_vision.points import Points3DTriangulation


from pipy.tf import Frame, Vector, Rotation


class StreamingMovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = None

    def process(self, value):
        value = np.array(value)

        if self.sum is None:
            self.sum = np.zeros_like(value)

        self.values.append(value)
        self.sum += value

        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)

        return self.sum / len(self.values)


class WorkOnCable(Node):
    def __init__(self, ur_type_right, ur_type_left, rate, exp_name, save_data):

        super().__init__("work_on_camere")

        # self.checkpoint_path_dlo_model = "/docker_camere/data/Modello_cavo.pth"
        self.checkpoint_path_dlo_model = "/docker_camere/data/lyric-violet-70_model_best_E6_max_angle.pth"

        self.ur_type_right = ur_type_right
        self.ur_type_left = ur_type_left
        self.tcp_link_right = None
        self.tcp_link_left = None
        self.frame_alta2bassa = None
        self.frame_bassa2alta = None
        self.init_shape_model = None
        self.compute_bbox = None
        self.compute_points2d_model = None
        self.num_points = 51
        self.work_completed = False
        self.r = self.create_rate(rate)

        #####################
        self.save_data = save_data
        self.loop_counter = 0
        self.output_data_path = f"/docker_camere/data/{exp_name}"
        self.streaming_filter = StreamingMovingAverage(window_size=5)
        #####################

        self.data_bassa = None
        self.data_alta = None

        self.bridge = CvBridge()
        self.condition_tf = Condition()
        self.condition_init = Condition()
        self.condition_tf_flag = False
        self.condition_init_flag = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.001, self.listener_tf)

        # Publishers
        self.pub_points3d = self.create_publisher(Float32MultiArray, "points3d", 1)
        self.pub_world_points3d = self.create_publisher(Float32MultiArray, "points3d_world", 1)
        self.publisher_points_rviz = self.create_publisher(Marker, "visualization_marker", 1)

        # Publishers simod_vision_msgs
        self.pub_data_input_bassa = self.create_publisher(Data, "data_input_bassa", 1)
        self.pub_data_input_alta = self.create_publisher(Data, "data_input_alta", 1)

        # Subscribers simod_vision_msgs
        self.create_subscription(Data, "data_output_bassa", self.callback_data_output_bassa, 1)
        self.create_subscription(Data, "data_output_alta", self.callback_data_output_alta, 1)

        print("Waiting for init shape...")
        # get init shape
        _, msg = wait_for_message(Float32MultiArray, self, "init_dlo_shape")
        self.init_shape_model = np.array(msg.data).reshape(-1, 3)
        print("Init shape received!")

        # Camera info
        _, msg = wait_for_message(CameraInfo, self, "oak_bassa/camera/rgb/camera_info")
        self.K_bassa = np.array(msg.k).reshape(3, 3)
        self.D_bassa = np.array(list(msg.d))

        _, msg = wait_for_message(CameraInfo, self, "oak_alta/camera/rgb/camera_info")
        self.K_alta = np.array(msg.k).reshape(3, 3)
        self.D_alta = np.array(list(msg.d))

        ########################################################################################
        thread_init_classes = threading.Thread(target=self.init_classes, daemon=True)
        thread_init_classes.start()

        print(Fore.GREEN + "Node created")

    def get_frame_from_tf(self, frame1, frame2):
        """
        Get the transformation frame2 -> frame1
        """
        tf = self.tf_buffer.lookup_transform(frame1, frame2, rclpy.time.Time(), rclpy.time.Duration(seconds=2.0))
        return Frame(
            Rotation.quaternion(
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z,
                tf.transform.rotation.w,
            ),
            Vector(
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ),
        )

    def tf_method(self, source_frame, target_frame):
        try:
            t = self.tf_buffer.lookup_transform(source_frame, target_frame, rclpy.time.Time())
            return np.array(
                [
                    t.transform.translation.x,
                    t.transform.translation.y,
                    t.transform.translation.z,
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                    t.transform.rotation.w,
                ]
            )
        except TransformException as ex:
            return None

    def listener_tf(self):
        # rispetto alla camera bassa
        with self.condition_tf:
            self.tcp_optical_right = self.tf_method(
                "cam_bassa_optical_frame", "tf_virtuale_{}".format(self.ur_type_right)
            )
            self.tcp_optical_left = self.tf_method(
                "cam_bassa_optical_frame", "tf_virtuale_{}".format(self.ur_type_left)
            )

            self.tcp_link_right = self.tf_method("cam_bassa_base_frame", "tf_virtuale_{}".format(self.ur_type_right))
            self.tcp_link_left = self.tf_method("cam_bassa_base_frame", "tf_virtuale_{}".format(self.ur_type_left))

            if self.tcp_link_right is not None and self.tcp_link_left is not None:
                self.condition_init_flag = True

            self.condition_tf.notify_all()

    def init_classes(self):
        with self.condition_init:
            with self.condition_tf:
                self.condition_tf.wait_for(lambda: self.condition_init_flag)

            self.frame_world2bassa_optical = self.get_frame_from_tf("world", "cam_bassa_optical_frame")
            self.frame_world2alta_optical = self.get_frame_from_tf("world", "cam_alta_optical_frame")

            self.frame_bassa2world = self.get_frame_from_tf("cam_bassa_base_frame", "world")
            self.T_bassa2world = self.frame_bassa2world.to_numpy()
            self.T_world2bassa = np.linalg.inv(self.T_bassa2world)

            self.frame_alta2bassa = self.get_frame_from_tf("cam_bassa_base_frame", "cam_alta_base_frame")

            self.frame_bassa2alta = self.frame_alta2bassa.inverse()

            rot = Rotation()
            rot.do_rotX(np.pi / 2)
            self.frame_cam_optical2base = Frame(rot, Vector(0, 0, 0))
            self.frame_cam_base2optical = self.frame_cam_optical2base.inverse()

            self.compute_bbox = BboxTcp(
                self.frame_cam_base2optical,
                self.frame_bassa2alta,
                self.K_bassa,
                self.D_bassa,
                self.K_alta,
                self.D_alta,
            )

            self.compute_points2d_model = Points2DFromModel(
                self.checkpoint_path_dlo_model,
                self.num_points,
                self.frame_bassa2alta,
                self.frame_cam_base2optical,
                self.K_bassa,
                self.D_bassa,
                self.K_alta,
                self.D_alta,
            )

            self.triangulation = Points3DTriangulation(
                self.K_bassa,
                self.D_bassa,
                self.K_alta,
                self.D_alta,
                self.frame_world2bassa_optical,
                self.frame_world2alta_optical,
                self.frame_bassa2world,
            )

            self.compute_points2d_model.update_prev_prediction(
                self.init_shape_model, self.tcp_link_right, self.tcp_link_left
            )
            self.compute_points2d_model.set_curr_frames(self.tcp_link_right, self.tcp_link_left)

            self.condition_init_flag = True
            self.condition_init.notify_all()

            print("DONE INIT!")

    def callback_data_output_bassa(self, msg):
        if msg.ready == True:
            # print("callback_data_output_bassa")
            self.data_bassa = {
                "vision": np.array(msg.p2d_vision.data).reshape(-1, 2),
                "model": np.array(msg.p2d_model.data).reshape(-1, 2),
                "curr_frame_1": np.array(msg.curr_frame_1.data),
                "curr_frame_2": np.array(msg.curr_frame_2.data),
                "mask": msg.mask,
                "img": msg.img,
                "bbox": msg.bbox,
            }

    def callback_data_output_alta(self, msg):
        if msg.ready == True:
            # print("callback_data_output_alta")
            self.data_alta = {
                "vision": np.array(msg.p2d_vision.data).reshape(-1, 2),
                "model": np.array(msg.p2d_model.data).reshape(-1, 2),
                "curr_frame_1": np.array(msg.curr_frame_1.data),
                "curr_frame_2": np.array(msg.curr_frame_2.data),
                "mask": msg.mask,
                "img": msg.img,
                "bbox": msg.bbox,
            }

    def publish(self):
        while rclpy.ok():

            with self.condition_init:
                self.condition_init.wait_for(lambda: self.condition_init_flag)

            # print("*****************")
            # bbox for segmentation
            bbox_bassa, bbox_alta = self.compute_bbox.compute_bbox(
                self.tcp_link_right, self.tcp_link_left, dlo_length=0.5
            )

            # set current frames
            self.compute_points2d_model.set_curr_frames(self.tcp_link_right, self.tcp_link_left)

            # points2d_model
            msg_points_bassa, msg_points_alta, msg_tcps_bassa, msg_tcps_alta = (
                self.compute_points2d_model.compute_points2d()
            )

            msg_curr_frame_1 = Float32MultiArray(data=self.tcp_link_right)
            msg_curr_frame_2 = Float32MultiArray(data=self.tcp_link_left)

            msg_bassa_data = Data()
            msg_bassa_data.p2d_model = msg_points_bassa
            msg_bassa_data.bbox = bbox_bassa
            msg_bassa_data.tcps_proj = msg_tcps_bassa
            msg_bassa_data.curr_frame_1 = msg_curr_frame_1
            msg_bassa_data.curr_frame_2 = msg_curr_frame_2

            msg_alta_data = Data()
            msg_alta_data.p2d_model = msg_points_alta
            msg_alta_data.bbox = bbox_alta
            msg_alta_data.tcps_proj = msg_tcps_alta
            msg_alta_data.curr_frame_1 = msg_curr_frame_1
            msg_alta_data.curr_frame_2 = msg_curr_frame_2

            self.pub_data_input_bassa.publish(msg_bassa_data)
            self.pub_data_input_alta.publish(msg_alta_data)
            # print("data input published")

            self.r.sleep()

    def points_triangulation(self):
        while rclpy.ok():

            with self.condition_init:
                self.condition_init.wait_for(lambda: self.condition_init_flag)

            if self.data_bassa is None or self.data_alta is None:
                continue

            points2d_vision_bassa = self.data_bassa["vision"]
            points2d_vision_alta = self.data_alta["vision"]
            curr_frame_1 = self.data_bassa["curr_frame_1"]
            curr_frame_2 = self.data_bassa["curr_frame_2"]

            # triangulation in world frame
            new_points = self.triangulation.triangulate_dlo_shape(points2d_vision_bassa, points2d_vision_alta)

            # check + add tip points
            new_points_f = self.triangulation.check_and_add_tip_points_smooth(new_points, curr_frame_1, curr_frame_2)

            # spline
            new_points_out = self.triangulation.smooth(new_points_f, num_points=self.num_points, s=1e-5)
            new_points_out = self.streaming_filter.process(new_points_out)

            ##################
            # PUB
            new_points_world = new_points_out @ self.T_world2bassa[:3, :3].T + self.T_world2bassa[:3, 3]
            self.pub_points3d.publish(Float32MultiArray(data=new_points_out.flatten()))
            self.pub_world_points3d.publish(Float32MultiArray(data=new_points_world.flatten()))
            self.publish_marker_points_rviz(new_points_world)

            # update
            self.compute_points2d_model.update_prev_prediction(new_points_out, curr_frame_1, curr_frame_2)

            self.loop_counter += 1
            ############################################
            if self.save_data:

                points2d_model_bassa = self.data_bassa["model"]
                points2d_model_alta = self.data_alta["model"]
                mask_bassa = self.bridge.imgmsg_to_cv2(self.data_bassa["mask"], desired_encoding="passthrough")
                mask_alta = self.bridge.imgmsg_to_cv2(self.data_alta["mask"], desired_encoding="passthrough")
                img_alta = self.bridge.imgmsg_to_cv2(self.data_alta["img"], desired_encoding="bgr8")
                img_bassa = self.bridge.imgmsg_to_cv2(self.data_bassa["img"], desired_encoding="bgr8")
                bbox_bassa = self.data_bassa["bbox"].data
                bbox_alta = self.data_alta["bbox"].data

                # save data
                folder_name = str(self.loop_counter).zfill(5)
                folder_path = os.path.join(self.output_data_path, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                print("Saving data to: ", folder_name)

                np.savetxt(os.path.join(folder_path, "pose_left.txt"), curr_frame_1)
                np.savetxt(os.path.join(folder_path, "pose_right.txt"), curr_frame_2)
                np.savetxt(os.path.join(folder_path, "points2d_vision_bassa.txt"), points2d_vision_bassa)
                np.savetxt(os.path.join(folder_path, "points2d_vision_alta.txt"), points2d_vision_alta)
                np.savetxt(os.path.join(folder_path, "points2d_model_bassa.txt"), points2d_model_bassa)
                np.savetxt(os.path.join(folder_path, "points2d_model_alta.txt"), points2d_model_alta)
                np.savetxt(os.path.join(folder_path, "new_points.txt"), new_points)
                np.savetxt(os.path.join(folder_path, "new_points_f.txt"), new_points_f)
                np.savetxt(os.path.join(folder_path, "new_points_out.txt"), new_points_out)
                np.savetxt(os.path.join(folder_path, "bbox_bassa.txt"), bbox_bassa)
                np.savetxt(os.path.join(folder_path, "bbox_alta.txt"), bbox_alta)
                cv2.imwrite(os.path.join(folder_path, "mask_bassa.png"), mask_bassa)
                cv2.imwrite(os.path.join(folder_path, "mask_alta.png"), mask_alta)
                cv2.imwrite(os.path.join(folder_path, "img_bassa.jpg"), img_bassa)
                cv2.imwrite(os.path.join(folder_path, "img_alta.jpg"), img_alta)

    def publish_marker_points_rviz(self, points3d, color=(1.0, 0.0, 0.0), id=0):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "points"
        marker.id = id
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Set marker properties
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in points3d]

        self.publisher_points_rviz.publish(marker)
        # self.get_logger().info("Published points to RViz2.")


def main(args=None):
    rclpy.init(args=args)

    args_without_ros = rclpy.utilities.remove_ros_args(args)

    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_name", default="pippo", type=str, help="Experiment name")
    parser.add_argument("-save", action="store_true", help="Save data")
    args = parser.parse_args(args_without_ros[1:])
    exp_name = args.exp_name
    save_data = args.save

    ur_type_left = "left"
    ur_type_right = "right"

    rate = 12

    node = WorkOnCable(
        ur_type_right=ur_type_right, ur_type_left=ur_type_left, rate=rate, exp_name=exp_name, save_data=save_data
    )

    # Crea un MultiThreadedExecutor con un numero di thread adeguato
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    thread_pub_points = threading.Thread(target=node.publish, daemon=True)
    thread_pub_points.start()

    thread_pub_3d = threading.Thread(target=node.points_triangulation, daemon=True)
    thread_pub_3d.start()

    try:
        executor.spin()  # Main thread sleep
    except KeyboardInterrupt:
        pass
    finally:
        thread_pub_points.join()
        thread_pub_3d.join()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
