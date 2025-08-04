import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.executors import MultiThreadedExecutor
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from colorama import Fore
import cv2
from sensor_msgs.msg import CameraInfo
from pipy.tf import Frame, Vector, Rotation
import threading
from rclpy.wait_for_message import wait_for_message
from threading import Condition
from test_camera.bbox import BboxTcp
from test_camera.points import Points2DFromModel
from dlo_python.cosserat_model.shape_initializer import DloInitShapeFromModel


from simod_vision_msgs.msg import Data
from std_msgs.msg import Float32MultiArray


class WorkOnCable(Node):
    def __init__(self, ur_type_right, ur_type_left):

        super().__init__("work_on_camere")

        self.ur_type_right = ur_type_right
        self.ur_type_left = ur_type_left
        self.tcp_link_right = None
        self.tcp_link_left = None
        self.frame_alta2bassa = None
        self.bboxer = None
        self.init_shape_model = None

        self.data_bassa = None
        self.data_alta = None
        self.mask_bassa = None
        self.mask_alta = None

        self.condition = Condition()
        self.condition_p2d_bassa = Condition()
        self.condition_p2d_alta = Condition()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.001, self.listener_tf)

        self.dlo_initialzier = DloInitShapeFromModel()

        # Publishers
        self.pub_data_input_bassa = self.create_publisher(Data, "data_input_bassa", 1)
        self.pub_data_input_alta = self.create_publisher(Data, "data_input_alta", 1)

        # Subscribers
        self.create_subscription(Data, "data_output_bassa", self.callback_data_output_bassa, 1)
        self.create_subscription(Data, "data_output_alta", self.callback_data_output_alta, 1)

        # Camera info
        _, msg = wait_for_message(CameraInfo, self, "oak_bassa/camera/rgb/camera_info")
        self.K_bassa = np.array(msg.k).reshape(3, 3)
        self.D_bassa = np.array(list(msg.d))

        _, msg = wait_for_message(CameraInfo, self, "oak_alta/camera/rgb/camera_info")
        self.K_alta = np.array(msg.k).reshape(3, 3)
        self.D_alta = np.array(list(msg.d))

        rot = Rotation()
        rot.do_rotX(np.pi / 2)
        self.frame_cam_optical2base = Frame(rot, Vector(0, 0, 0))

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

    def listener_tf(self):
        # rispetto alla camera bassa
        with self.condition:

            self.tcp_link_right = self.tf_method("cam_bassa_base_frame", "tf_virtuale_{}".format(self.ur_type_right))
            self.tcp_link_left = self.tf_method("cam_bassa_base_frame", "tf_virtuale_{}".format(self.ur_type_left))

            self.frame_alta2bassa = self.get_frame_from_tf("cam_bassa_base_frame", "cam_alta_base_frame")

            self.frame_bassa2alta = self.frame_alta2bassa.inverse()
            self.frame_cam_base2optical = self.frame_cam_optical2base.inverse()

            if (
                self.bboxer is None
                and self.init_shape_model is None
                and self.tcp_link_right is not None
                and self.tcp_link_left is not None
            ):
                print("init bboxer and shape model")
                self.bboxer = BboxTcp(
                    self.frame_cam_base2optical,
                    self.frame_bassa2alta,
                    self.K_bassa,
                    self.D_bassa,
                    self.K_alta,
                    self.D_alta,
                )

                self.compute_points2d_model = Points2DFromModel(
                    51,
                    self.frame_bassa2alta,
                    self.frame_cam_base2optical,
                    self.K_bassa,
                    self.D_bassa,
                    self.K_alta,
                    self.D_alta,
                )

                self.init_shape_model = self.dlo_initialzier.run(
                    target_1=self.tcp_link_right,
                    target_2=self.tcp_link_left,
                    debug=False,
                )
                self.compute_points2d_model.update_prev_prediction(
                    self.init_shape_model, self.tcp_link_right, self.tcp_link_left
                )
                self.compute_points2d_model.set_curr_frames(self.tcp_link_right, self.tcp_link_left)

            self.condition.notify_all()

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

    def callback_data_output_bassa(self, msg):
        if msg.ready == True:
            print("callback_data_output_bassa")
            self.data_bassa = {
                "vision": np.array(msg.p2d_vision.data).reshape(-1, 2),
                "model": np.array(msg.p2d_model.data).reshape(-1, 2),
                "curr_frame_1": np.array(msg.curr_frame_1.data),
                "curr_frame_2": np.array(msg.curr_frame_2.data),
                "mask": msg.mask,
                "bbox": msg.bbox,
            }

    def callback_data_output_alta(self, msg):
        if msg.ready == True:
            print("callback_data_output_alta")
            self.data_alta = {
                "vision": np.array(msg.p2d_vision.data).reshape(-1, 2),
                "model": np.array(msg.p2d_model.data).reshape(-1, 2),
                "curr_frame_1": np.array(msg.curr_frame_1.data),
                "curr_frame_2": np.array(msg.curr_frame_2.data),
                "mask": msg.mask,
                "bbox": msg.bbox,
            }

    def main_loop(self):
        self.create_timer(0.001, self.publish)

    def publish(self):

        if self.tcp_link_left is None or self.tcp_link_right is None:
            print("TCP not available")
            return

        print("*****************")
        # bbox for segmentation
        bbox_bassa, bbox_alta = self.bboxer.compute_bbox(self.tcp_link_right, self.tcp_link_left, dlo_length=0.5)

        # set current frames
        self.compute_points2d_model.set_curr_frames(self.tcp_link_right, self.tcp_link_left)

        # points2d_model
        msg_points_bassa, msg_points_alta = self.compute_points2d_model.compute_points2d()

        msg_bassa_data = Data()
        msg_bassa_data.p2d_model = msg_points_bassa
        msg_bassa_data.bbox = bbox_bassa
        msg_bassa_data.curr_frame_1 = Float32MultiArray(data=self.tcp_link_right)
        msg_bassa_data.curr_frame_2 = Float32MultiArray(data=self.tcp_link_left)

        msg_alta_data = Data()
        msg_alta_data.p2d_model = msg_points_alta
        msg_alta_data.bbox = bbox_alta
        msg_alta_data.curr_frame_1 = Float32MultiArray(data=self.tcp_link_right)
        msg_alta_data.curr_frame_2 = Float32MultiArray(data=self.tcp_link_left)

        self.pub_data_input_bassa.publish(msg_bassa_data)
        self.pub_data_input_alta.publish(msg_alta_data)
        print("data input published")


def main(args=None):
    rclpy.init(args=args)

    ur_type_left = "left"
    ur_type_right = "right"

    node = WorkOnCable(ur_type_right=ur_type_right, ur_type_left=ur_type_left)

    # Crea un MultiThreadedExecutor con un numero di thread adeguato
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    thread_left = threading.Thread(target=node.main_loop, daemon=True)
    thread_left.start()

    try:
        executor.spin()  # Main thread sleep

    except KeyboardInterrupt:
        pass
    finally:
        thread_left.join()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
