import sys, os, glob
from pathlib import Path

# sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
sys.path.append(os.path.dirname(__file__))

from dataclasses import dataclass
from run_omnivla_modified import Inference, InferenceConfig, define_model
import numpy as np
from scipy.spatial.transform import Rotation as R
import json

import cv2
import matplotlib.pyplot as plt
import rosbag
from cv_bridge import CvBridge
from ros_utils import load_calibration, make_offset_paths, create_yaws_from_path,\
      make_corridor_polygon, make_corridor_polygon_from_cam_lines, project_clip, \
        clean_2d, draw_polyline, draw_corridor
from PIL import Image

PATH_COLOR = (0, 0, 255) # Red
@dataclass
class FrameItem:
    idx: int
    stamp: object   # rospy.Time
    img_pil: Image.Image
    img_bgr: np.ndarray
    position: np.ndarray 
    velocity: float
    omega: float
    rotation: np.ndarray
    yaw: float

class InferenceROSBag():
    def __init__(self, bag_path: str, calib_path: str, topics_path: str, timestamp_path: str,vla: Inference):
        self.bag_path = bag_path
        self.bag_name = Path(bag_path).name
        self.needs_correction = False
        stem = Path(self.bag_name).stem

        fx, fy, cx, cy = 640.0, 637.0, 640.0, 360.0
        self.looakhead = 2.4 #s
        self.frames : list[FrameItem] = []
        self.bridge = CvBridge()

        self.expert_action_annotation_dir = os.path.join(timestamp_path, self.bag_name.replace(".bag", ".json"))

        with open(topics_path, 'r') as f:
            topics = json.load(f)

        try:
            with open(self.expert_action_annotation_dir, 'r') as f:
                self.action_annotations = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load expert action annotations from {self.expert_action_annotation_dir}: {e}")
            raise e
        
        self.timestamps = self._get_timestamps_from_expert_annotations()

        if "Jackal" in self.bag_name:
            self.K, self.dist, self.T_base_from_cam = load_calibration(calib_path, fx, fy, cx, cy, mode="jackal")
            self.T_cam_from_base = np.linalg.inv(self.T_base_from_cam)
            mode = "jackal"
        elif "Spot" in self.bag_name:
            self.K, self.dist, self.T_base_from_cam = load_calibration(calib_path, fx, fy, cx, cy, mode="spot")
            self.T_cam_from_base = np.linalg.inv(self.T_base_from_cam)
            mode = "spot"
            self.needs_correction = True
        else:
            raise Exception
        
        self.vla = vla
        self.image_topic = topics.get(mode).get("camera")
        self.odom_topic = topics.get(mode).get("odom")
        self.robot_width = topics.get(mode).get("width")

        self.goal_lookahead = None
        self.waypoints = None
        self.left_boundary = None
        self.right_boundary = None
        self.goal_point_base = None

        self.current_img_bgr = None
        self.current_img_show = None
        self.current_goal_img = None
        self.fig = None
        self.ax_overlay = None
        self.ax_bev = None
        self.visualize = False

    def _setup_matplotlib(self):
        if self.fig is not None:
            return
        plt.ion()
        self.fig = plt.figure(figsize=(18, 10), dpi=80)
        gs = self.fig.add_gridspec(2, 2)
        self.ax_cur = self.fig.add_subplot(gs[0, 0])
        self.ax_goal = self.fig.add_subplot(gs[1, 0])
        self.ax_bev = self.fig.add_subplot(gs[:, 1])
        self.ax_cur.set_title("Egocentric current image", fontsize=16)
        self.ax_goal.set_title("Egocentric goal image", fontsize=16)
        self.ax_bev.set_title("Normalized generated 2D trajectories from OmniVLA", fontsize=18)
        self.ax_cur.axis("off")
        self.ax_goal.axis("off")

        # >>> add this line so _update_matplotlib_view() can use ax_overlay
        self.ax_overlay = self.ax_cur

    def _get_timestamps_from_expert_annotations(self):
        timestamps = []
        for key in self.action_annotations.get("annotations_by_stamp", {}).keys():
            timestamps.append(int(key))
        return timestamps
    
    def _get_goal_metadata(self, frame_idx: int):
        self.goal_lookahead = np.random.uniform(5.0 , 15.0) # create a uniform distribution between 5 and 15 meters
        dist = 0.0
        goal_idx = frame_idx
        while dist < self.goal_lookahead:
            if goal_idx + 1 >= len(self.frames):
                break
            p1 = self.frames[goal_idx].position
            p2 = self.frames[goal_idx + 1].position
            dist += np.linalg.norm(p2 - p1)
            goal_idx += 1

        return (
            self.frames[goal_idx].position,
            self.frames[goal_idx].yaw,
            self.frames[goal_idx].img_pil,
        )

    def _goal_in_base_frame(self, goal_pos: np.ndarray, frame: FrameItem) -> np.ndarray:
        """Return the goal position expressed in the frame's base coordinates."""
        rel = goal_pos - frame.position
        rel_base = frame.rotation.T @ rel  # rotation maps base->world
        rel_base[2] = 0.0
        return rel_base
                    
    def process_odom(self, msg):
        
        quaternion = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        rotation_matrix = R.from_quat(quaternion).as_matrix()

        if self.needs_correction:
            current_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
            velocity_robot_frame = np.linalg.inv(rotation_matrix) @ current_vel

            v = velocity_robot_frame[0]
        else:
            v = msg.twist.twist.linear.x
        
        yaw = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        w = msg.twist.twist.angular.z

        return pos,v,w, rotation_matrix, yaw

    def process_image(self, msg):
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        return cv_img

    def create_path_meta(self):

        yaws = create_yaws_from_path(self.waypoints)
        left_b, right_b, poly_b = make_corridor_polygon(self.waypoints, yaws, self.robot_width)
        
        return left_b, right_b, poly_b
    
    def process_waypoints(self):
        self.waypoints = self.waypoints.squeeze()
        x = self.waypoints[:, 0]
        y = self.waypoints[:, 1]
        z = np.zeros_like(x)
        waypoints_b = np.stack([x, y, z], axis=1)
        # print(waypoints_b.shape)
        waypoints_b = np.stack([[0, 0, 0], *waypoints_b], axis=0)  # add robot position at start
        self.waypoints = waypoints_b

    def draw(self):

        left_b, right_b, poly_b = self.create_path_meta()
        if (self.current_img_bgr is None or self.waypoints is None
                or getattr(self.waypoints, "size", 0) == 0):
            return
        img = self.current_img_bgr.copy()
        img_h, img_w = img.shape[:2]

        # print(left_b.shape, right_b.shape, self.waypoints.shape)
        points_2d = clean_2d(project_clip(self.waypoints, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True), img_w, img_h)
        left_2d   = clean_2d(project_clip(left_b,  self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True), img_w, img_h)
        right_2d  = clean_2d(project_clip(right_b, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True), img_w, img_h)

        poly_2d   = make_corridor_polygon_from_cam_lines(left_2d, right_2d)
        draw_polyline(img, points_2d, 2, color=PATH_COLOR)
        draw_corridor(img, poly_2d, left_2d, right_2d, fill_alpha=0.15, fill_color=PATH_COLOR, edge_color=PATH_COLOR, edge_thickness=2)

        # print(self.goal_point_base)
        if self.goal_point_base is not None:
            center = np.array([self.goal_point_base])
            center = clean_2d(project_clip(center, self.T_cam_from_base, self.K, self.dist, img_h, img_w, smooth_first=True), img_w, img_h)[:2][0]
            center = center.astype(int)
            cv2.circle(img, center, 10, (0, 0, 255), -1)
            cv2.circle(img, center, 16, (255, 255, 255), 1)

        self.current_img_show = img
        self._update_matplotlib_view(img, left_b, right_b, poly_b)

    def _update_matplotlib_view(self, overlay_img, left_b, right_b, poly_b):
        if overlay_img is None or self.waypoints is None:
            return

        self._setup_matplotlib()
        self.ax_overlay.clear()
        self.ax_bev.clear()

        self.ax_overlay.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
        self.ax_overlay.axis("off")
        self.ax_overlay.set_title("Egocentric overlay", fontsize=16)

        x_seq = self.waypoints[:, 0]
        y_seq_inv = -self.waypoints[:, 1]
        self.ax_bev.plot(
            np.insert(y_seq_inv, 0, 0.0),
            np.insert(x_seq, 0, 0.0),
            linewidth=4.0,
            markersize=10,
            marker="o",
            color="blue",
            label="VLA path",
        )

        if left_b is not None and len(left_b) > 0:
            self.ax_bev.plot(-left_b[:, 1], left_b[:, 0], color="red", linewidth=2, label="Left boundary")
        if right_b is not None and len(right_b) > 0:
            self.ax_bev.plot(-right_b[:, 1], right_b[:, 0], color="red", linewidth=2, label="Right boundary")
        if poly_b is not None and len(poly_b) > 0:
            self.ax_bev.fill(-poly_b[:, 1], poly_b[:, 0], color="cyan", alpha=0.15, edgecolor="cyan", linewidth=1, label="Corridor")

        if self.goal_point_base is not None:
            goal_x = self.goal_point_base[0]
            goal_y = -self.goal_point_base[1]
            self.ax_bev.plot(goal_y, goal_x, marker="*", color="magenta", markersize=15, label="Goal")

        self.ax_bev.set_xlim(-3.0, 3.0)
        self.ax_bev.set_ylim(-0.1, 10.0)
        self.ax_bev.set_xlabel("Left (+) / Right (-) [m]")
        self.ax_bev.set_ylabel("Forward [m]")
        self.ax_bev.legend(loc="upper right")
        self.ax_bev.grid(True, linestyle="--", alpha=0.3)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def run_rosbag(self):
        print(f"Running inference on rosbag: {self.bag_path}")

        count = 0
        skip_count = 0
        timestamp_counter = 0
        with rosbag.Bag(self.bag_path, "r") as bag:

            for topic, msg, t in bag.read_messages(topics=[self.odom_topic, self.image_topic]):
                if topic == self.odom_topic:
                    pos, v, w, rot, yaw = self.process_odom(msg)
                    pos_defined = True
                elif topic == self.image_topic:
                    cv_img = self.process_image(msg)
                    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

                    if pos_defined and str(t) == str(self.timestamps[timestamp_counter]):
                        self.frames.append(
                            FrameItem(
                                idx=count,
                                stamp=t,
                                img_pil=pil_img,
                                img_bgr=cv_img.copy(),
                                position=pos,
                                velocity=v,
                                omega=w,
                                rotation=rot,
                                yaw=yaw,
                            )
                        )
                        timestamp_counter += 1
                        count+=1       
                if timestamp_counter >= len(self.timestamps):
                    break 
        
        print(f"[INFO] Loaded {len(self.frames)} frames from bag after skipping {skip_count} frames.")
        if not self.frames:
            print("[WARN] No frames after undersampling.")
            return
        
        idx = 0
        # try:
        while 0 <= idx < len(self.frames):
            fr = self.frames[idx]
            self.frame_idx = fr.idx
            self.frame_stamp = fr.stamp
            self.current_img_bgr = fr.img_bgr
            
            goal_pos, goal_yaw, goal_img = self._get_goal_metadata(idx)
            self.goal_point_base = self._goal_in_base_frame(goal_pos, fr)
            self.current_goal_img = goal_img

            self.vla.update_current_state(fr.img_pil, fr.position, fr.yaw)
            self.vla.update_goal(goal_image_PIL=goal_img, 
                                    goal_utm=goal_pos,
                                    goal_compass=goal_yaw, 
                                    lan_inst_prompt=None)
            self.vla.run()
            self.waypoints = self.vla.waypoints * self.vla.metric_waypoint_spacing
            self.process_waypoints()
            # print(self.waypoints)
            if self.visualize:
                self.draw()
            # print(f"Wayppoints for frame {idx} computed.: {self.vla.waypoints}")

            idx += 1
        # finally:
        #     self.cleanup()
        print("Inference completed.")

if __name__ == "__main__":

    omnivla = Inference(save_dir="./inference",
                        ego_frame_mode=True,
                        save_images=False, 
                        radians=True
                        )

    bag_dir = "/media/beast-gamma/Media/Datasets/SCAND/annt"   # Point to path with rosbags being annotated for the day
    timestamp_path = "/media/beast-gamma/Media/Datasets/SCAND/ActionAnnotations"
    annotations_root = "./Annotations"
    calib_path = "/home/beast-gamma/Documents/GAMMA/Projects/VisualFrontiers-Annotation/SCAND/tf.json"
    skip_json_path = "/home/beast-gamma/Documents/GAMMA/Projects/VisualFrontiers-Annotation/SCAND/bags_to_skip.json"
    topic_json_path = "/home/beast-gamma/Documents/GAMMA/Projects/VisualFrontiers-Annotation/SCAND/topics_for_project.json"

    fx, fy, cx, cy = 640.0, 637.0, 640.0, 360.0                   #  SCAND Kinect intrinsics ### DO NOT CHANGE
    undersampling_factor = 6
    num_keypoints = 5
    max_deviation = 1.5

    bag_files = sorted(glob.glob(os.path.join(bag_dir, "*.bag")))
    print(len(bag_files), "bags found in", bag_dir)
    with open(skip_json_path, 'r') as f:
        bags_to_skip = json.load(f)

    if not bag_files:
        print(f"[ERROR] No .bag files found in {bag_dir}")

    for bp in bag_files:
        if bags_to_skip.get(os.path.basename(bp), False):
            print(f"[INFO] Skipping {bp}")
            continue

        print(f"[INFO] Processing {bp}")
        inference = InferenceROSBag(bag_path=bp, calib_path=calib_path, topics_path=topic_json_path, 
                                    timestamp_path=timestamp_path, vla=omnivla)
        inference.run_rosbag()
