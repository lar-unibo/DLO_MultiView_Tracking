import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.wait_for_message import wait_for_message

import cv2
import numpy as np
import networkx as nx

from simod_vision.dlo_seg import DloSeg
from dlo_python.dlo_fusion.graph_gen import GraphGeneration, points_association, correction

from simod_vision_msgs.msg import Data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


class FeedBackCamera(Node):
    def __init__(self, checkpoint_seg_path, camera_name):

        super().__init__("feedback_camera_{}".format(camera_name))
        self.camera_name = camera_name
        self.topic_camera = "/oak_{}/camera/rgb/image_raw".format(camera_name)

        self.dlo_seg = DloSeg(checkpoint_seg_path, show=False)
        self.graph_gen = GraphGeneration(n_knn=8, th_edges_similarity=0.25, th_mask=127, wsize=15, sampling_ratio=0.1)

        self.bridge = CvBridge()
        self.cbg = ReentrantCallbackGroup()
        self.create_subscription(Image, self.topic_camera, self.callback_img, 1, callback_group=self.cbg)
        self.create_subscription(Data, "data_input_{}".format(camera_name), self.callback, 1)
        self.pub_points = self.create_publisher(Data, "data_output_{}".format(camera_name), 1)

        # only for debug
        self.pub_mask = self.create_publisher(Image, "mask_points2d_{}".format(camera_name), 1)

        # check if the camera is working
        _, msg = wait_for_message(Image, self, self.topic_camera)
        self.img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        print("Node {} created".format(camera_name))

    def callback_img(self, msg):
        self.img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def callback(self, msg):

        points2d = np.array(msg.p2d_model.data).reshape(-1, 2)
        tcps_proj = np.array(msg.tcps_proj.data).reshape(-1, 2)

        # semantic segmentation
        mask, _ = self.dlo_seg.single_crop_predict(self.img, msg.bbox.data)

        # extract DLO shape from vision
        nodes, edges = self.graph_gen.exec(mask)
        nodes = nodes[:, [1, 0]]

        ## add tcps to the graph and invert x and y
        nodes = np.concatenate((nodes, tcps_proj))

        # create the graph
        graph = nx.Graph()
        graph.add_nodes_from([(it, {"pos": np.array(x)}) for it, x in enumerate(nodes)])
        graph.add_edges_from(edges)

        # compute corrected points starting from the model DLO points
        new_points = points_association(points2d, nodes, graph)

        # correction by interpolation
        valid_points, _ = correction(points2d, new_points, mask)

        # Publish the corrected points
        msg_out = Data()
        msg_out.bbox = msg.bbox
        msg_out.p2d_model = msg.p2d_model
        msg_out.p2d_vision = Float32MultiArray(data=valid_points.flatten())
        msg_out.curr_frame_1 = msg.curr_frame_1
        msg_out.curr_frame_2 = msg.curr_frame_2
        msg_out.mask = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        msg_out.img = self.bridge.cv2_to_imgmsg(self.img, encoding="bgr8")
        msg_out.ready = True
        self.pub_points.publish(msg_out)
        print("Points published")
        ##########
        # Debug
        canvas = cv2.addWeighted(self.img, 0.5, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
        for p in valid_points:
            try:
                cv2.circle(canvas, (int(p[0]), int(p[1])), 5, (255, 0, 0), -1)
            except:
                pass
        self.pub_mask.publish(self.bridge.cv2_to_imgmsg(canvas, encoding="bgr8"))


def main(args=None):
    rclpy.init(args=args)

    camera_bassa = "bassa"
    camera_alta = "alta"

    name = "Modello_segmentazione.pth"
    checkpoint_seg_path = "/docker_camere/data/" + name

    node_bassa = FeedBackCamera(checkpoint_seg_path, camera_bassa)
    node_alta = FeedBackCamera(checkpoint_seg_path, camera_alta)

    # Crea un MultiThreadedExecutor con un numero di thread adeguato
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node_bassa)
    executor.add_node(node_alta)

    try:
        executor.spin()  # Main thread sleep

    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
