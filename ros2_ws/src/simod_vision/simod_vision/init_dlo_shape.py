import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

from colorama import Fore
import numpy as np

import threading
from threading import Condition
from std_msgs.msg import Float32MultiArray
from dlo_python.cosserat_model.shape_initializer import DloInitShapeFromModel


class InitDloShapeNode(Node):
    def __init__(self, ur_type_right, ur_type_left, rate):

        super().__init__("init_dlo_shape_node")

        self.ur_type_right = ur_type_right
        self.ur_type_left = ur_type_left
        self.tcp_link_right = None
        self.tcp_link_left = None

        self.r = self.create_rate(rate)

        self.condition_tf = Condition()
        self.condition_tf_flag = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.001, self.listener_tf)

        self.dlo_initialzier = DloInitShapeFromModel()

        self.pub_init_shape = self.create_publisher(Float32MultiArray, "init_dlo_shape", 1)

        print(Fore.GREEN + "Node created")

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
                self.condition_tf_flag = True

            self.condition_tf.notify_all()

    def publish(self):

        with self.condition_tf:
            self.condition_tf.wait_for(lambda: self.condition_tf_flag)

        self.init_shape_model = self.dlo_initialzier.run(
            target_1=self.tcp_link_right,
            target_2=self.tcp_link_left,
            debug=False,
        )

        while rclpy.ok():
            self.pub_init_shape.publish(Float32MultiArray(data=self.init_shape_model.flatten()))
            self.r.sleep()


def main(args=None):
    rclpy.init(args=args)

    ur_type_left = "left"
    ur_type_right = "right"
    rate = 1

    node = InitDloShapeNode(ur_type_right=ur_type_right, ur_type_left=ur_type_left, rate=rate)

    # Crea un MultiThreadedExecutor con un numero di thread adeguato
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    thread_pub = threading.Thread(target=node.publish, daemon=True)
    thread_pub.start()

    try:
        executor.spin()  # Main thread sleep
    except KeyboardInterrupt:
        pass
    finally:
        thread_pub.join()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
