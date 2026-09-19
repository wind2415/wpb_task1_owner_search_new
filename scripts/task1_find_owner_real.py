#!/usr/bin/env python3
# coding: utf-8
import base64
import importlib.util
import math
import audioop
import json
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
import xml.etree.ElementTree as ET

import actionlib
import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
try:
    import tf
except ImportError:
    tf = None
from actionlib_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetPlan
from perception_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image, JointState, LaserScan, PointCloud2
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty


DEFAULT_QWEN_MODEL = "qwen3.5:0.8b"


def clamp(value, low, high):
    return max(low, min(high, value))


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def signed_angle_diff(target, current):
    return math.atan2(math.sin(target - current), math.cos(target - current))


def load_qwen_action_core(configured_path=""):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if configured_path:
        candidates.append(os.path.expanduser(str(configured_path)))
    candidates.extend(
        [
            os.path.join(
                script_dir,
                "..",
                "..",
                "offline_voice_bridge",
                "scripts",
                "qwen_action_recognition_node.py",
            ),
            "/home/ubuntu20/catkin_ws/src/offline_voice_bridge/scripts/qwen_action_recognition_node.py",
        ]
    )
    core_path = next((os.path.abspath(path) for path in candidates if os.path.exists(path)), "")
    if not core_path:
        raise RuntimeError("verified Qwen action core not found")

    module_name = "offline_voice_bridge_qwen_action_core"
    spec = importlib.util.spec_from_file_location(module_name, core_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Qwen action core: %s" % core_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, core_path


class MissingHardwareError(RuntimeError):
    """Raised when required robot sensor streams are not producing messages."""


class RealOwnerSearchBeforeAction:
    """Real WPB owner-search task with owner centering and action recognition."""

    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.latest_image = None
        self.latest_image_time = None
        self.latest_detections = []
        self.latest_detections_time = None
        self.latest_scan = None
        self.latest_scan_time = None
        self.latest_pointcloud = None
        self.latest_pointcloud_time = None
        self.latest_yaw = None
        self.latest_odom_xy = None
        self.latest_odom_linear_speed = None
        self.latest_odom_time = None
        self.owner_track_center = None
        self.last_pointcloud_reason = ""
        self.last_approach_failure_reason = ""

        self.waypoint_name = str(rospy.get_param("~waypoint_name", "living_room")).strip()
        self.waypoint_file = os.path.expanduser(rospy.get_param("~waypoint_file", "~/waypoints.xml"))
        self.owner_image_path = os.path.expanduser(rospy.get_param("~owner_image_path", ""))
        self.task_waypoint_names = self.parse_waypoint_name_list(
            rospy.get_param("~task_waypoint_names", [self.waypoint_name])
        ) or [self.waypoint_name]
        self.exit_waypoint_name = str(rospy.get_param("~exit_waypoint_name", "exit")).strip()
        self.exit_waypoint_aliases = self.parse_waypoint_name_list(rospy.get_param("~exit_waypoint_aliases", ["1"]))
        self.return_to_exit_when_complete = bool(rospy.get_param("~return_to_exit_when_complete", True))
        self.rename_exit_waypoint_alias = bool(rospy.get_param("~rename_exit_waypoint_alias", True))
        self.waypoint_speech_names = {
            "living_room": "客厅",
            "kitchen": "厨房",
            "bedroom": "卧室",
            "canteen": "餐厅",
            "exit": "出口",
        }
        self.waypoint_speech_names.update(
            self.parse_waypoint_speech_names(rospy.get_param("~waypoint_speech_names", {}))
        )

        self.image_topic = rospy.get_param("~image_topic", "/kinect2/qhd/image_color_rect")
        self.detections_topic = rospy.get_param("~detections_topic", "/perception/person_detections_2d")
        self.points_topic = rospy.get_param("~points_topic", "/kinect2/qhd/points")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.say_topic = rospy.get_param("~say_topic", "/voice/say")
        self.asr_topic = rospy.get_param("~asr_topic", "/voice/asr_text")
        self.electrical_switch_state_topic = rospy.get_param(
            "~electrical_switch_state_topic", "/electrical_switch/state"
        )
        self.pause_yolo_topic = rospy.get_param("~pause_yolo_topic", "/yoloworld/pause")

        self.require_kinect = bool(rospy.get_param("~require_kinect", True))
        self.require_lidar = bool(rospy.get_param("~require_lidar", True))
        self.require_odom = bool(rospy.get_param("~require_odom", True))
        self.hardware_check_timeout = float(rospy.get_param("~hardware_check_timeout", 90.0))

        self.clear_costmaps_before_navigation = bool(rospy.get_param("~clear_costmaps_before_navigation", True))
        self.retry_navigation_after_clear = bool(rospy.get_param("~retry_navigation_after_clear", True))
        self.clear_costmaps_service = rospy.get_param("~clear_costmaps_service", "/move_base/clear_costmaps")
        self.clear_costmaps_timeout = float(rospy.get_param("~clear_costmaps_timeout", 5.0))

        self.face_verify_enabled = bool(rospy.get_param("~face_verify_enabled", True))
        self.face_verify_required = bool(rospy.get_param("~face_verify_required", True))
        self.allow_unverified_owner = bool(rospy.get_param("~allow_unverified_owner", False))
        self.owner_enrollment_enabled = bool(rospy.get_param("~owner_enrollment_enabled", False))
        self.owner_enrollment_name_timeout = max(
            1.0, float(rospy.get_param("~owner_enrollment_name_timeout", 15.0))
        )
        self.owner_enrollment_confirm_timeout = max(
            1.0, float(rospy.get_param("~owner_enrollment_confirm_timeout", 8.0))
        )
        self.owner_enrollment_capture_seconds = max(
            1.0, float(rospy.get_param("~owner_enrollment_capture_seconds", 5.0))
        )
        self.owner_enrollment_sample_interval = max(
            0.1, float(rospy.get_param("~owner_enrollment_sample_interval", 0.35))
        )
        self.owner_enrollment_min_samples = max(
            1, int(rospy.get_param("~owner_enrollment_min_samples", 6))
        )
        self.owner_enrollment_min_face_size = max(
            20, int(rospy.get_param("~owner_enrollment_min_face_size", 70))
        )
        self.owner_enrollment_retries = max(
            1, int(rospy.get_param("~owner_enrollment_retries", 3))
        )
        self.owner_enrollment_asr_settle_seconds = max(
            0.0, float(rospy.get_param("~owner_enrollment_asr_settle_seconds", 0.8))
        )
        self.face_auto_download = bool(rospy.get_param("~face_auto_download", False))
        self.face_model_name = rospy.get_param("~face_model_name", "buffalo_sc")
        self.face_model_root = os.path.expanduser(rospy.get_param("~face_model_root", "~/.insightface"))
        self.face_ctx_id = int(rospy.get_param("~face_ctx_id", -1))
        self.face_det_size = int(rospy.get_param("~face_det_size", 480))
        self.face_det_thresh = float(rospy.get_param("~face_det_thresh", 0.35))
        self.face_accept_threshold = float(rospy.get_param("~face_accept_threshold", 0.45))
        self.face_reject_threshold = float(rospy.get_param("~face_reject_threshold", 0.25))
        self.face_fast_reject = bool(rospy.get_param("~face_fast_reject", True))
        self.face_reference_try_rotations = bool(rospy.get_param("~face_reference_try_rotations", False))
        self.face_crop_padding = float(rospy.get_param("~face_crop_padding", 0.28))
        self.face_crop_top_ratio = float(rospy.get_param("~face_crop_top_ratio", 0.68))
        self.face_crop_lying_side_ratio = float(rospy.get_param("~face_crop_lying_side_ratio", 0.50))
        self.face_crop_lying_extra_side_ratios = self.parse_float_list(
            rospy.get_param("~face_crop_lying_extra_side_ratios", [0.45, 0.65, 0.75])
        )
        self.face_crop_try_rotations = bool(rospy.get_param("~face_crop_try_rotations", True))
        self.face_candidate_enhance = bool(rospy.get_param("~face_candidate_enhance", True))
        self.face_fast_pass_enabled = bool(rospy.get_param("~face_fast_pass_enabled", True))
        self.face_fast_pass_reject_threshold = float(
            rospy.get_param("~face_fast_pass_reject_threshold", 0.30)
        )

        self.detection_min_score = float(rospy.get_param("~detection_min_score", 0.30))
        self.detection_min_area_ratio = float(rospy.get_param("~detection_min_area_ratio", 0.015))
        self.verify_top_k = max(1, int(rospy.get_param("~verify_top_k", 1)))
        self.verify_during_scan = bool(rospy.get_param("~verify_during_scan", True))
        self.verify_after_scan_top_k = max(1, int(rospy.get_param("~verify_after_scan_top_k", 1)))
        self.candidate_cooldown = float(rospy.get_param("~candidate_cooldown", 1.0))
        self.candidate_collection_interval = float(rospy.get_param("~candidate_collection_interval", 0.5))
        self.scan_candidate_pool_size = max(1, int(rospy.get_param("~scan_candidate_pool_size", 6)))

        self.navigate_enabled = bool(rospy.get_param("~navigate_enabled", True))
        self.navigate_timeout = float(rospy.get_param("~navigate_timeout", 120.0))
        self.scan_angular_speed = float(rospy.get_param("~scan_angular_speed", -0.25))
        self.scan_duration = float(rospy.get_param("~scan_duration", 13.0))
        self.scan_total_angle = float(rospy.get_param("~scan_total_angle", math.pi))
        self.scan_timeout = float(rospy.get_param("~scan_timeout", 35.0))
        self.scan_after_arrival_delay = float(rospy.get_param("~scan_after_arrival_delay", 0.2))
        self.scan_return_to_owner = bool(rospy.get_param("~scan_return_to_owner", True))
        self.return_angular_speed = abs(float(rospy.get_param("~return_angular_speed", 0.25)))

        self.center_owner_enabled = bool(rospy.get_param("~center_owner_enabled", True))
        self.center_owner_timeout = float(rospy.get_param("~center_owner_timeout", 5.0))
        self.center_owner_tolerance = float(rospy.get_param("~center_owner_tolerance", 0.06))
        self.center_owner_angular_gain = float(rospy.get_param("~center_owner_angular_gain", 0.65))
        self.center_owner_max_angular_speed = abs(float(rospy.get_param("~center_owner_max_angular_speed", 0.30)))
        self.center_owner_lost_turn_speed = abs(float(rospy.get_param("~center_owner_lost_turn_speed", 0.10)))

        self.action_recognition_enabled = bool(rospy.get_param("~action_recognition_enabled", True))
        self.action_model_path = os.path.expanduser(rospy.get_param("~action_model_path", ""))
        self.action_core_path = rospy.get_param("~action_core_path", "")
        self.action_llm_url = rospy.get_param(
            "~action_llm_url", "http://127.0.0.1:11434/api/chat"
        )
        self.action_llm_model = str(
            rospy.get_param("~action_llm_model", DEFAULT_QWEN_MODEL)
        ).strip() or DEFAULT_QWEN_MODEL
        self.action_llm_timeout = float(rospy.get_param("~action_llm_timeout", 45.0))
        self.action_llm_keep_alive = rospy.get_param("~action_llm_keep_alive", "30m")
        self.action_llm_max_tokens = int(rospy.get_param("~action_llm_max_tokens", 20))
        self.action_llm_num_ctx = int(rospy.get_param("~action_llm_num_ctx", 4096))
        self.action_llm_num_gpu = max(
            0, int(rospy.get_param("~action_llm_num_gpu", 0))
        )
        self.action_device = rospy.get_param("~action_device", "cuda:0")
        self.action_require_gpu = bool(rospy.get_param("~action_require_gpu", True))
        self.action_imgsz = int(rospy.get_param("~action_imgsz", 416))
        self.action_conf = float(rospy.get_param("~action_conf", 0.25))
        self.action_iou = float(rospy.get_param("~action_iou", 0.45))
        self.action_half = bool(rospy.get_param("~action_half", True))
        self.action_sample_seconds = float(rospy.get_param("~action_sample_seconds", 5.0))
        self.action_sample_rate = float(rospy.get_param("~action_sample_rate", 3.0))
        self.action_frame_count = max(1, int(rospy.get_param("~action_frame_count", 9)))
        self.action_llm_frame_count = max(
            1, int(rospy.get_param("~action_llm_frame_count", 3))
        )
        self.action_jpeg_quality = int(rospy.get_param("~action_jpeg_quality", 65))
        self.action_image_max_width = int(rospy.get_param("~action_image_max_width", 320))
        self.action_warmup_retries = max(1, int(rospy.get_param("~action_warmup_retries", 3)))
        self.action_warmup_retry_delay = max(
            0.0, float(rospy.get_param("~action_warmup_retry_delay", 2.0))
        )
        self.action_warmup_wait_timeout = max(
            0.0, float(rospy.get_param("~action_warmup_wait_timeout", 30.0))
        )
        self.action_pose_enabled = bool(rospy.get_param("~action_pose_enabled", True))
        self.action_pointcloud_enabled = bool(
            rospy.get_param("~action_pointcloud_enabled", True)
        )
        self.action_pointcloud_camera_height = float(
            rospy.get_param("~action_pointcloud_camera_height", 0.85)
        )
        self.action_pointcloud_ground_height_limit = float(
            rospy.get_param("~action_pointcloud_ground_height_limit", 0.35)
        )
        self.action_pointcloud_furniture_height_limit = float(
            rospy.get_param("~action_pointcloud_furniture_height_limit", 0.42)
        )
        self.action_pointcloud_max_age = max(
            0.1, float(rospy.get_param("~action_pointcloud_max_age", 1.0))
        )
        self.action_pointcloud_stride = max(
            1, int(rospy.get_param("~action_pointcloud_stride", 8))
        )
        self.action_pointcloud_min_samples = max(
            8, int(rospy.get_param("~action_pointcloud_min_samples", 30))
        )
        self.action_pointcloud_roi_padding = float(
            rospy.get_param("~action_pointcloud_roi_padding", 0.20)
        )
        self.action_pointcloud_anchor_radius = float(
            rospy.get_param("~action_pointcloud_anchor_radius", 14.0)
        )
        self.action_pointcloud_depth_percentile = float(
            rospy.get_param("~action_pointcloud_depth_percentile", 20.0)
        )
        self.action_pointcloud_surface_band = float(
            rospy.get_param("~action_pointcloud_surface_band", 0.25)
        )
        self.action_pointcloud_local_ground_padding = float(
            rospy.get_param("~action_pointcloud_local_ground_padding", 0.85)
        )
        self.action_pointcloud_elevated_delta = float(
            rospy.get_param("~action_pointcloud_elevated_delta", 0.24)
        )
        self.action_pointcloud_frame_mode = rospy.get_param(
            "~action_pointcloud_frame_mode", "auto"
        )
        self.action_min_keypoint_conf = float(rospy.get_param("~action_min_keypoint_conf", 0.25))
        self.action_max_det = int(rospy.get_param("~action_max_det", 4))
        self.action_pause_yolo = bool(rospy.get_param("~action_pause_yolo", True))
        self.action_yolo_pause_settle_seconds = max(
            0.0,
            float(rospy.get_param("~action_yolo_pause_settle_seconds", 0.35)),
        )
        self.action_use_owner_roi = bool(rospy.get_param("~action_use_owner_roi", False))
        self.action_roi_padding = float(rospy.get_param("~action_roi_padding", 0.55))
        self.action_min_pose_samples = int(rospy.get_param("~action_min_pose_samples", 5))
        self.action_static_required_ratio = float(rospy.get_param("~action_static_required_ratio", 0.65))
        self.action_sitting_min_torso_verticality = float(
            rospy.get_param("~action_sitting_min_torso_verticality", 0.45)
        )
        self.action_sitting_knee_angle_max = float(rospy.get_param("~action_sitting_knee_angle_max", 150.0))
        self.action_sitting_knee_hip_y_ratio = float(rospy.get_param("~action_sitting_knee_hip_y_ratio", 0.45))
        self.action_sitting_ankle_hip_y_ratio = float(rospy.get_param("~action_sitting_ankle_hip_y_ratio", 0.62))
        self.action_sitting_thigh_horizontal_ratio = float(
            rospy.get_param("~action_sitting_thigh_horizontal_ratio", 0.65)
        )
        self.action_sitting_compact_aspect_min = float(
            rospy.get_param("~action_sitting_compact_aspect_min", 0.65)
        )
        self.action_sitting_support_score = float(rospy.get_param("~action_sitting_support_score", 0.90))
        self.action_sitting_relaxed_ratio = float(rospy.get_param("~action_sitting_relaxed_ratio", 0.60))
        self.action_fall_center_drop = float(rospy.get_param("~action_fall_center_drop", 0.06))
        self.action_fall_torso_drop = float(rospy.get_param("~action_fall_torso_drop", 0.20))
        self.action_fall_aspect_gain = float(rospy.get_param("~action_fall_aspect_gain", 0.25))
        self.action_fall_lie_ratio_gain = float(rospy.get_param("~action_fall_lie_ratio_gain", 0.35))
        self.action_fall_late_lie_ratio = float(rospy.get_param("~action_fall_late_lie_ratio", 0.45))
        self.action_fall_height_shrink_ratio = float(rospy.get_param("~action_fall_height_shrink_ratio", 0.82))
        self.action_fall_early_upright_ratio = float(rospy.get_param("~action_fall_early_upright_ratio", 0.50))
        self.action_fall_early_lie_ratio_max = float(rospy.get_param("~action_fall_early_lie_ratio_max", 0.25))
        self.action_fall_static_lie_ratio = float(rospy.get_param("~action_fall_static_lie_ratio", 0.65))
        self.action_fall_early_det_aspect_max = float(rospy.get_param("~action_fall_early_det_aspect_max", 1.10))
        self.action_fall_static_det_aspect_min = float(rospy.get_param("~action_fall_static_det_aspect_min", 1.20))
        self.action_fall_min_transition_signals = max(
            1,
            int(rospy.get_param("~action_fall_min_transition_signals", 2)),
        )
        self.already_fallen_surface_labels = self.parse_string_list(
            rospy.get_param("~already_fallen_surface_labels", ["lying", "unknown"])
        )
        self.lying_surface_classification_enabled = bool(
            rospy.get_param("~lying_surface_classification_enabled", True)
        )
        self.lying_surface_camera_height = float(rospy.get_param("~lying_surface_camera_height", 0.85))
        self.lying_ground_max_surface_height = float(rospy.get_param("~lying_ground_max_surface_height", 0.35))
        self.lying_furniture_min_surface_height = float(rospy.get_param("~lying_furniture_min_surface_height", 0.42))
        self.lying_ground_bbox_bottom_ratio = float(rospy.get_param("~lying_ground_bbox_bottom_ratio", 0.88))
        self.lying_ground_bbox_center_ratio = float(rospy.get_param("~lying_ground_bbox_center_ratio", 0.62))
        self.action_report_standing = bool(rospy.get_param("~action_report_standing", False))
        self.action_speech_hold = float(rospy.get_param("~action_speech_hold", 1.2))

        self.approach_on_waving_enabled = bool(rospy.get_param("~approach_on_waving_enabled", True))
        self.approach_timeout = float(rospy.get_param("~approach_timeout", 25.0))
        self.approach_linear_speed = abs(float(rospy.get_param("~approach_linear_speed", 0.18)))
        self.approach_angular_gain = float(rospy.get_param("~approach_angular_gain", 0.45))
        self.approach_max_angular_speed = abs(float(rospy.get_param("~approach_max_angular_speed", 0.25)))
        self.approach_stop_distance = float(rospy.get_param("~approach_stop_distance", 0.45))
        self.approach_standoff_distance = float(rospy.get_param("~approach_standoff_distance", self.approach_stop_distance))
        self.approach_distance_tolerance = float(rospy.get_param("~approach_distance_tolerance", 0.05))
        self.approach_fast_finish_tolerance = float(
            rospy.get_param("~approach_fast_finish_tolerance", self.approach_distance_tolerance)
        )
        self.approach_bearing_tolerance = float(rospy.get_param("~approach_bearing_tolerance", 0.06))
        self.approach_forward_bearing_limit = float(rospy.get_param("~approach_forward_bearing_limit", 0.45))
        self.approach_arrival_stable_cycles = max(1, int(rospy.get_param("~approach_arrival_stable_cycles", 1)))
        self.approach_missing_data_grace = float(rospy.get_param("~approach_missing_data_grace", 0.7))
        self.approach_missing_linear_scale = float(rospy.get_param("~approach_missing_linear_scale", 0.35))
        self.approach_command_smoothing = float(rospy.get_param("~approach_command_smoothing", 0.45))
        self.approach_center_tolerance = float(rospy.get_param("~approach_center_tolerance", 0.12))
        self.approach_target_height_ratio = float(rospy.get_param("~approach_target_height_ratio", 0.55))
        self.approach_lost_turn_speed = abs(float(rospy.get_param("~approach_lost_turn_speed", 0.10)))
        self.approach_front_scan_degrees = float(rospy.get_param("~approach_front_scan_degrees", 25.0))
        self.approach_scan_max_age = float(rospy.get_param("~approach_scan_max_age", 0.8))
        self.approach_require_lidar = bool(rospy.get_param("~approach_require_lidar", False))
        self.approach_lidar_stop_distance = float(rospy.get_param("~approach_lidar_stop_distance", 0.55))
        self.approach_lidar_margin = float(rospy.get_param("~approach_lidar_margin", 0.08))
        self.approach_pointcloud_mode = str(rospy.get_param("~approach_pointcloud_mode", "auto")).lower()
        self.approach_pointcloud_max_age = float(rospy.get_param("~approach_pointcloud_max_age", 1.0))
        self.approach_pointcloud_min_samples = max(5, int(rospy.get_param("~approach_pointcloud_min_samples", 30)))
        self.approach_pointcloud_stride = max(1, int(rospy.get_param("~approach_pointcloud_stride", 8)))
        self.approach_pointcloud_roi_x_margin = float(rospy.get_param("~approach_pointcloud_roi_x_margin", 0.25))
        self.approach_pointcloud_roi_y_min_ratio = float(rospy.get_param("~approach_pointcloud_roi_y_min_ratio", 0.15))
        self.approach_pointcloud_roi_y_max_ratio = float(rospy.get_param("~approach_pointcloud_roi_y_max_ratio", 0.85))
        self.approach_pointcloud_depth_percentile = float(rospy.get_param("~approach_pointcloud_depth_percentile", 35.0))
        self.approach_pointcloud_surface_band = float(rospy.get_param("~approach_pointcloud_surface_band", 0.30))
        self.approach_min_depth = float(rospy.get_param("~approach_min_depth", 0.35))
        self.approach_max_depth = float(rospy.get_param("~approach_max_depth", 5.0))
        self.approach_linear_gain = float(rospy.get_param("~approach_linear_gain", 0.25))
        self.approach_min_linear_speed = abs(float(rospy.get_param("~approach_min_linear_speed", 0.06)))
        self.approach_slow_finish_enabled = bool(rospy.get_param("~approach_slow_finish_enabled", True))
        self.approach_slow_finish_tolerance = max(
            0.0,
            float(rospy.get_param("~approach_slow_finish_tolerance", 0.05)),
        )
        self.approach_slow_finish_linear_speed = abs(
            float(rospy.get_param("~approach_slow_finish_linear_speed", 0.03))
        )
        self.approach_slow_finish_cycles = max(1, int(rospy.get_param("~approach_slow_finish_cycles", 3)))
        self.approach_odom_speed_max_age = max(0.1, float(rospy.get_param("~approach_odom_speed_max_age", 0.6)))
        self.approach_navigation_enabled = bool(rospy.get_param("~approach_navigation_enabled", True))
        self.approach_navigation_frame = rospy.get_param("~approach_navigation_frame", "map")
        self.approach_navigation_base_frame = rospy.get_param("~approach_navigation_base_frame", "base_footprint")
        if self.approach_navigation_frame in ("base_footprint", "base_link"):
            rospy.logwarn(
                "approach_navigation_frame=%s is a robot frame and will be rejected by move_base; using map as the goal frame and %s as the base frame",
                self.approach_navigation_frame,
                self.approach_navigation_frame,
            )
            self.approach_navigation_base_frame = self.approach_navigation_frame
            self.approach_navigation_frame = "map"
        self.approach_navigation_tf_timeout = float(rospy.get_param("~approach_navigation_tf_timeout", 0.6))
        self.approach_navigation_timeout = float(rospy.get_param("~approach_navigation_timeout", 18.0))
        self.approach_navigation_server_timeout = float(rospy.get_param("~approach_navigation_server_timeout", 3.0))
        self.approach_navigation_retry_after_clear = bool(
            rospy.get_param("~approach_navigation_retry_after_clear", True)
        )
        self.approach_navigation_clear_costmaps_before_goal = bool(
            rospy.get_param("~approach_navigation_clear_costmaps_before_goal", False)
        )
        self.approach_navigation_min_distance = max(
            0.0,
            float(rospy.get_param("~approach_navigation_min_distance", 0.05)),
        )
        self.approach_navigation_lidar_guard_enabled = bool(
            rospy.get_param("~approach_navigation_lidar_guard_enabled", False)
        )
        self.approach_navigation_stuck_timeout = float(
            rospy.get_param("~approach_navigation_stuck_timeout", 4.0)
        )
        self.approach_navigation_stuck_min_progress = float(
            rospy.get_param("~approach_navigation_stuck_min_progress", 0.05)
        )
        self.approach_navigation_stuck_linear_speed = abs(
            float(rospy.get_param("~approach_navigation_stuck_linear_speed", 0.025))
        )
        self.approach_direct_fallback_enabled = bool(rospy.get_param("~approach_direct_fallback_enabled", False))
        self.waving_approach_candidate_enabled = bool(
            rospy.get_param("~waving_approach_candidate_enabled", True)
        )
        self.waving_approach_min_owner_distance = clamp(
            abs(float(rospy.get_param("~waving_approach_min_owner_distance", 0.45))),
            0.35,
            0.50,
        )
        self.waving_approach_max_owner_distance = clamp(
            abs(float(rospy.get_param("~waving_approach_max_owner_distance", 0.50))),
            self.waving_approach_min_owner_distance,
            0.50,
        )
        self.waving_approach_candidate_distances = self.parse_float_list(
            rospy.get_param("~waving_approach_candidate_distances", [0.45, 0.48, 0.50])
        )
        self.waving_approach_candidate_angles_deg = self.parse_float_list(
            rospy.get_param("~waving_approach_candidate_angles_deg", [65, -65, 95, -95, 35, -35, 0, 125, -125])
        )
        self.waving_approach_plan_check = bool(rospy.get_param("~waving_approach_plan_check", True))
        self.waving_approach_plan_service = rospy.get_param(
            "~waving_approach_plan_service", "/move_base/make_plan"
        )
        self.waving_approach_plan_tolerance = max(
            0.0,
            float(rospy.get_param("~waving_approach_plan_tolerance", 0.20)),
        )
        self.waving_approach_safety_radius = clamp(
            abs(float(rospy.get_param("~waving_approach_safety_radius", 0.48))),
            self.waving_approach_min_owner_distance,
            self.waving_approach_max_owner_distance,
        )
        self.waving_approach_still_duration = max(
            0.2,
            float(rospy.get_param("~waving_approach_still_duration", 0.8)),
        )
        self.waving_approach_plan_detour_ratio = max(
            1.0,
            float(rospy.get_param("~waving_approach_plan_detour_ratio", 2.0)),
        )
        self.waving_approach_plan_detour_margin = max(
            0.0,
            float(rospy.get_param("~waving_approach_plan_detour_margin", 0.8)),
        )
        self.waving_approach_plan_turn_limit = max(
            math.pi,
            float(rospy.get_param("~waving_approach_plan_turn_limit", 4.5)),
        )
        self.waving_approach_retry_after_clear = bool(
            rospy.get_param("~waving_approach_retry_after_clear", False)
        )
        self.approach_help_prompt = rospy.get_param("~approach_help_prompt", "请问您需要什么帮助？")
        self.waving_help_pause_seconds = max(0.0, float(rospy.get_param("~waving_help_pause_seconds", 3.0)))

        self.fall_approach_enabled = bool(rospy.get_param("~fall_approach_enabled", True))
        self.fall_approach_action_labels = self.parse_string_list(
            rospy.get_param(
                "~fall_approach_action_labels",
                ["sudden_fall", "fallen", "falling", "lying_ground", "lying", "sitting"],
            )
        )
        self.fall_approach_position_sample_seconds = float(rospy.get_param("~fall_approach_position_sample_seconds", 0.9))
        self.fall_approach_min_position_samples = max(
            1,
            int(rospy.get_param("~fall_approach_min_position_samples", 2)),
        )
        self.fall_approach_standoff_distance = float(rospy.get_param("~fall_approach_standoff_distance", 0.75))
        self.fall_approach_distance_tolerance = float(rospy.get_param("~fall_approach_distance_tolerance", 0.08))
        self.fall_approach_fast_finish_tolerance = float(
            rospy.get_param("~fall_approach_fast_finish_tolerance", self.fall_approach_distance_tolerance)
        )
        self.fall_approach_linear_speed = abs(float(rospy.get_param("~fall_approach_linear_speed", 0.14)))
        self.fall_approach_min_linear_speed = abs(float(rospy.get_param("~fall_approach_min_linear_speed", 0.05)))
        self.fall_approach_linear_gain = float(rospy.get_param("~fall_approach_linear_gain", self.approach_linear_gain))
        self.fall_approach_turn_timeout = float(rospy.get_param("~fall_approach_turn_timeout", 8.0))
        self.fall_approach_drive_timeout = float(rospy.get_param("~fall_approach_drive_timeout", 18.0))
        self.fall_approach_max_travel_distance = float(rospy.get_param("~fall_approach_max_travel_distance", 1.50))
        self.fall_approach_lidar_stop_distance = float(
            rospy.get_param("~fall_approach_lidar_stop_distance", self.approach_lidar_stop_distance)
        )
        self.fall_approach_lidar_margin = float(rospy.get_param("~fall_approach_lidar_margin", self.approach_lidar_margin))
        self.fall_approach_extra_close_enabled = bool(rospy.get_param("~fall_approach_extra_close_enabled", True))
        self.fall_approach_extra_close_distance = float(rospy.get_param("~fall_approach_extra_close_distance", 0.18))
        self.fall_approach_extra_close_speed = abs(float(rospy.get_param("~fall_approach_extra_close_speed", 0.08)))
        self.fall_approach_extra_close_timeout = float(rospy.get_param("~fall_approach_extra_close_timeout", 6.0))
        self.fall_approach_extra_close_finish_tolerance = float(
            rospy.get_param("~fall_approach_extra_close_finish_tolerance", 0.05)
        )

        self.fall_assist_arm_enabled = bool(rospy.get_param("~fall_assist_arm_enabled", True))
        self.fall_assist_arm_action_labels = self.parse_string_list(
            rospy.get_param(
                "~fall_assist_arm_action_labels",
                ["sudden_fall", "fallen", "falling", "lying_ground"],
            )
        )
        self.mani_ctrl_topic = rospy.get_param("~mani_ctrl_topic", "/wpb_home/mani_ctrl")
        self.fall_assist_arm_extend_lift = float(rospy.get_param("~fall_assist_arm_extend_lift", 0.50))
        self.fall_assist_arm_extend_gripper = float(rospy.get_param("~fall_assist_arm_extend_gripper", 0.12))
        self.fall_assist_arm_retract_lift = float(rospy.get_param("~fall_assist_arm_retract_lift", 0.0))
        self.fall_assist_arm_retract_gripper = float(rospy.get_param("~fall_assist_arm_retract_gripper", 0.12))
        self.fall_assist_arm_lift_velocity = float(rospy.get_param("~fall_assist_arm_lift_velocity", 0.5))
        self.fall_assist_arm_gripper_velocity = float(rospy.get_param("~fall_assist_arm_gripper_velocity", 5.0))
        self.fall_assist_arm_extend_wait = float(rospy.get_param("~fall_assist_arm_extend_wait", 3.0))
        self.fall_assist_arm_hold_seconds = float(rospy.get_param("~fall_assist_arm_hold_seconds", 4.0))
        self.fall_assist_arm_retract_wait = float(rospy.get_param("~fall_assist_arm_retract_wait", 3.0))
        self.fall_assist_arm_completion_wait = max(
            0.0,
            float(rospy.get_param("~fall_assist_arm_completion_wait", 1.0)),
        )
        self.fall_assist_arm_command_rate = float(rospy.get_param("~fall_assist_arm_command_rate", 5.0))

        self.electrical_switch_instruction_enabled = bool(
            rospy.get_param("~electrical_switch_instruction_enabled", True)
        )
        self.electrical_switch_instruction_source = str(
            rospy.get_param("~electrical_switch_instruction_source", "direct_asr")
        ).strip().lower()
        self.electrical_switch_instruction_timeout = float(
            rospy.get_param("~electrical_switch_instruction_timeout", 0.0)
        )
        self.electrical_switch_instruction_window_seconds = float(
            rospy.get_param("~electrical_switch_instruction_window_seconds", 5.0)
        )
        self.electrical_switch_instruction_asr_settle_seconds = float(
            rospy.get_param("~electrical_switch_instruction_asr_settle_seconds", 1.0)
        )
        self.electrical_switch_instruction_max_empty_windows = int(
            rospy.get_param("~electrical_switch_instruction_max_empty_windows", 0)
        )
        self.electrical_switch_prompt = rospy.get_param("~electrical_switch_prompt", "请指示。")
        self.electrical_switch_prompt_hold = float(rospy.get_param("~electrical_switch_prompt_hold", 1.5))
        self.electrical_switch_ready_ding_enabled = bool(
            rospy.get_param("~electrical_switch_ready_ding_enabled", True)
        )
        self.electrical_switch_ready_ding_wav = os.path.expanduser(
            rospy.get_param("~electrical_switch_ready_ding_wav", "/dev/shm/task1_switch_ready_ding.wav")
        )
        self.electrical_switch_ready_ding_frequency = float(
            rospy.get_param("~electrical_switch_ready_ding_frequency", 880.0)
        )
        self.electrical_switch_ready_ding_duration = float(
            rospy.get_param("~electrical_switch_ready_ding_duration", 0.16)
        )
        self.electrical_switch_ready_ding_volume = float(
            rospy.get_param("~electrical_switch_ready_ding_volume", 0.65)
        )
        self.electrical_switch_pause_yolo_during_voice = bool(
            rospy.get_param("~electrical_switch_pause_yolo_during_voice", True)
        )
        self.electrical_switch_pause_yolo_settle_seconds = float(
            rospy.get_param("~electrical_switch_pause_yolo_settle_seconds", 0.15)
        )
        self.electrical_switch_script_filter_output = bool(
            rospy.get_param("~electrical_switch_script_filter_output", True)
        )
        self.electrical_switch_script_output_pattern = str(
            rospy.get_param(
                "~electrical_switch_script_output_pattern",
                r"(模型|预加载|开始测试|第\s*\d+\s*轮|录音|RMS|识别文本|判断结果|没有识别|没有听出|LLM|ASR|TTS|播报|提示音|ERROR|WARN)",
            )
        )
        self.electrical_switch_ready_ding_player = str(
            rospy.get_param("~electrical_switch_ready_ding_player", "aplay")
        ).strip()
        self.electrical_switch_ready_ding_speaker_device = str(
            rospy.get_param("~electrical_switch_ready_ding_speaker_device", "default")
        ).strip()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        task_package_dir = os.path.abspath(os.path.join(script_dir, ".."))
        self.task_package_dir = task_package_dir
        catkin_src_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
        default_switch_script_path = os.path.join(
            task_package_dir,
            "tools",
            "local_switch_command_test.py",
        )
        default_switch_asr_model = os.path.join(
            catkin_src_dir,
            "offline_voice_bridge",
            "models",
            "whisper",
            "faster-whisper-small",
        )
        self.electrical_switch_script_path = os.path.expanduser(
            rospy.get_param("~electrical_switch_script_path", default_switch_script_path)
        )
        self.electrical_switch_script_python = str(
            rospy.get_param("~electrical_switch_script_python", "python3")
        ).strip() or "python3"
        self.electrical_switch_script_until_result = bool(
            rospy.get_param("~electrical_switch_script_until_result", True)
        )
        self.electrical_switch_asr_capture_device = rospy.get_param(
            "~electrical_switch_asr_capture_device", rospy.get_param("/asr/capture_device", "default")
        )
        self.electrical_switch_asr_energy_threshold = int(
            rospy.get_param("~electrical_switch_asr_energy_threshold", rospy.get_param("/asr/energy_threshold", 300))
        )
        self.electrical_switch_asr_sample_rate = int(rospy.get_param("~electrical_switch_asr_sample_rate", 16000))
        self.electrical_switch_asr_channels = int(rospy.get_param("~electrical_switch_asr_channels", 1))
        self.electrical_switch_asr_model_path = os.path.expanduser(
            rospy.get_param("~electrical_switch_asr_model_path", default_switch_asr_model)
        )
        self.electrical_switch_asr_hf_endpoint = rospy.get_param(
            "~electrical_switch_asr_hf_endpoint", "https://huggingface.co"
        )
        self.electrical_switch_asr_device = rospy.get_param("~electrical_switch_asr_device", "cpu")
        self.electrical_switch_asr_compute_type = rospy.get_param("~electrical_switch_asr_compute_type", "int8")
        self.electrical_switch_asr_language = rospy.get_param("~electrical_switch_asr_language", "zh")
        self.electrical_switch_asr_beam_size = int(rospy.get_param("~electrical_switch_asr_beam_size", 1))
        self.electrical_switch_asr_vad_filter = bool(rospy.get_param("~electrical_switch_asr_vad_filter", False))
        self.electrical_switch_asr_no_speech_threshold = float(
            rospy.get_param("~electrical_switch_asr_no_speech_threshold", 0.8)
        )
        self.electrical_switch_asr_keep_wav = bool(rospy.get_param("~electrical_switch_asr_keep_wav", False))
        self.electrical_switch_asr_wav_dir = rospy.get_param("~electrical_switch_asr_wav_dir", "/dev/shm")
        self.electrical_switch_asr_input_gain = max(
            1.0,
            float(rospy.get_param("~electrical_switch_asr_input_gain", 3.0)),
        )
        self.electrical_switch_asr_auto_gain_target_peak = clamp(
            float(rospy.get_param("~electrical_switch_asr_auto_gain_target_peak", 0.70)),
            0.0,
            0.98,
        )
        self.electrical_switch_asr_max_gain = max(
            self.electrical_switch_asr_input_gain,
            float(rospy.get_param("~electrical_switch_asr_max_gain", 8.0)),
        )
        self.electrical_switch_ollama_url = rospy.get_param(
            "~electrical_switch_ollama_url", "http://127.0.0.1:11434/api/chat"
        )
        self.electrical_switch_ollama_enabled = bool(
            rospy.get_param("~electrical_switch_ollama_enabled", True)
        )
        self.electrical_switch_ollama_model = str(
            rospy.get_param("~electrical_switch_ollama_model", DEFAULT_QWEN_MODEL)
        ).strip() or DEFAULT_QWEN_MODEL
        self.electrical_switch_ollama_autoselect_model = bool(
            rospy.get_param("~electrical_switch_ollama_autoselect_model", True)
        )
        self.electrical_switch_ollama_model_fallbacks = self.parse_string_list(
            rospy.get_param("~electrical_switch_ollama_model_fallbacks", [])
        )
        self.electrical_switch_ollama_tags_timeout = float(
            rospy.get_param("~electrical_switch_ollama_tags_timeout", 2.0)
        )
        self.electrical_switch_ollama_num_gpu = int(
            rospy.get_param("~electrical_switch_ollama_num_gpu", 0)
        )
        self.electrical_switch_ollama_timeout = float(
            rospy.get_param("~electrical_switch_ollama_timeout", 30.0)
        )
        self.electrical_switch_ollama_keep_alive = rospy.get_param(
            "~electrical_switch_ollama_keep_alive", "10m"
        )
        self.electrical_switch_ollama_max_tokens = max(
            8, int(rospy.get_param("~electrical_switch_ollama_max_tokens", 20))
        )
        self.electrical_switch_ollama_warmup_enabled = bool(
            rospy.get_param("~electrical_switch_ollama_warmup_enabled", True)
        )
        self.electrical_switch_ollama_warmup_timeout = float(
            rospy.get_param(
                "~electrical_switch_ollama_warmup_timeout",
                max(15.0, self.electrical_switch_ollama_timeout),
            )
        )
        self.electrical_switch_ollama_warmup_retries = max(
            1, int(rospy.get_param("~electrical_switch_ollama_warmup_retries", 3))
        )
        self.electrical_switch_ollama_warmup_retry_delay = float(
            rospy.get_param("~electrical_switch_ollama_warmup_retry_delay", 5.0)
        )
        self.electrical_switch_ollama_failure_cooldown = float(
            rospy.get_param("~electrical_switch_ollama_failure_cooldown", 60.0)
        )
        self.electrical_switch_fast_keyword_first = bool(
            rospy.get_param("~electrical_switch_fast_keyword_first", True)
        )
        self.electrical_switch_preload_enabled = bool(
            rospy.get_param("~electrical_switch_preload_enabled", True)
        )
        self.electrical_switch_preload_wait_before_prompt = bool(
            rospy.get_param("~electrical_switch_preload_wait_before_prompt", True)
        )
        self.electrical_switch_preload_wait_timeout = float(
            rospy.get_param("~electrical_switch_preload_wait_timeout", 20.0)
        )
        self.electrical_switch_asr_transcribe_from_memory = bool(
            rospy.get_param("~electrical_switch_asr_transcribe_from_memory", True)
        )
        self.electrical_switch_reply_on = rospy.get_param(
            "~electrical_switch_reply_on", "好的，已开启电气开关。"
        )
        self.electrical_switch_reply_off = rospy.get_param(
            "~electrical_switch_reply_off", "好的，已关闭电气开关。"
        )
        self.electrical_switch_reply_unknown = rospy.get_param(
            "~electrical_switch_reply_unknown", "抱歉，我没有听出开关指令。"
        )
        self.latest_asr_text = ""
        self.latest_asr_time = None
        self.asr_sequence = 0
        self.asr_history = []
        self.electrical_switch_asr_model = None
        self.electrical_switch_asr_ready = False
        self.electrical_switch_asr_lock = threading.Lock()
        self.electrical_switch_ollama_lock = threading.Lock()
        self.ollama_request_lock = threading.Lock()
        self.electrical_switch_ollama_available = None
        self.electrical_switch_ollama_last_failure = 0.0
        self.electrical_switch_preload_done = threading.Event()
        self.electrical_switch_preload_started = False
        self.electrical_switch_preload_thread = None
        self.electrical_switch_state = "unknown"

        self.speak_on_start = bool(rospy.get_param("~speak_on_start", True))
        self.speak_on_arrival = bool(rospy.get_param("~speak_on_arrival", False))
        self.speak_on_owner_found = bool(rospy.get_param("~speak_on_owner_found", True))
        self.speak_on_finish = bool(rospy.get_param("~speak_on_finish", True))
        self.say_wait_for_subscribers = bool(rospy.get_param("~say_wait_for_subscribers", True))
        self.say_wait_timeout = float(rospy.get_param("~say_wait_timeout", 20.0))
        self.say_repeat_count = max(1, int(rospy.get_param("~say_repeat_count", 1)))
        self.say_repeat_interval = float(rospy.get_param("~say_repeat_interval", 0.25))
        self.say_after_publish_delay = float(rospy.get_param("~say_after_publish_delay", 1.0))

        self.owner_reference_images = []
        self.owner_name = ""
        self.face_app = None
        self.face_model_ready = False
        self.owner_face_embedding = None
        self.owner_face_embeddings = []
        self.face_ready = False
        self.action_pose_model = None
        self.action_ready = False
        self.qwen_action_core = None
        self.qwen_action_core_path = ""
        self.qwen_pose_analyzer = None
        self.qwen_pointcloud_analyzer = None
        self.qwen_warmup_done = threading.Event()
        self.qwen_warmup_error = ""
        self.last_owner_action_place = "unknown"
        self.last_owner_action_result = {}

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.say_pub = rospy.Publisher(self.say_topic, String, queue_size=5)
        self.electrical_switch_state_pub = rospy.Publisher(
            self.electrical_switch_state_topic, String, queue_size=1, latch=True
        )
        self.pause_yolo_pub = rospy.Publisher(self.pause_yolo_topic, Bool, queue_size=1, latch=True)
        self.mani_ctrl_pub = rospy.Publisher(self.mani_ctrl_topic, JointState, queue_size=5)

        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.det_sub = rospy.Subscriber(self.detections_topic, Detection2DArray, self.detections_callback, queue_size=1)
        self.points_sub = rospy.Subscriber(self.points_topic, PointCloud2, self.pointcloud_callback, queue_size=1)
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.asr_sub = rospy.Subscriber(self.asr_topic, String, self.asr_callback, queue_size=10)

        self.publish_electrical_switch_state()
        self.start_electrical_switch_preload()

        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.tf_listener = tf.TransformListener() if tf is not None else None
        self.waving_make_plan = rospy.ServiceProxy(self.waving_approach_plan_service, GetPlan)

        if self.face_verify_enabled:
            if not self.owner_enrollment_enabled:
                self.owner_reference_images = self.load_owner_images()
            self.init_face_recognizer()
        if (
            self.face_verify_required
            and not self.face_ready
            and not self.owner_enrollment_enabled
            and not self.allow_unverified_owner
        ):
            raise RuntimeError(
                "face verification is required but not ready; check owner_image_path and InsightFace model cache"
            )
    def load_owner_images(self):
        if not self.owner_image_path:
            raise RuntimeError("owner_image_path is empty")
        if not os.path.exists(self.owner_image_path):
            raise RuntimeError("owner photo path not found: %s" % self.owner_image_path)

        image_paths = []
        if os.path.isdir(self.owner_image_path):
            extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            for name in sorted(os.listdir(self.owner_image_path)):
                path = os.path.join(self.owner_image_path, name)
                if os.path.isfile(path) and name.lower().endswith(extensions):
                    image_paths.append(path)
            if not image_paths:
                raise RuntimeError("no owner reference photos found under: %s" % self.owner_image_path)
        else:
            image_paths.append(self.owner_image_path)

        images = []
        for path in image_paths:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                rospy.logwarn("Skipping unreadable owner reference photo: %s", path)
                continue
            images.append((path, image))

        if not images:
            raise RuntimeError("failed to read any owner reference photos from: %s" % self.owner_image_path)
        return images

    @staticmethod
    def normalize_embedding(embedding):
        vector = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6:
            return None
        return vector / norm

    @staticmethod
    def parse_float_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            raw_values = value.replace(";", ",").split(",")
        elif isinstance(value, (list, tuple)):
            raw_values = value
        else:
            raw_values = [value]

        parsed = []
        for item in raw_values:
            try:
                parsed.append(float(item))
            except (TypeError, ValueError):
                continue
        return parsed

    @staticmethod
    def parse_string_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            raw_values = value.replace(";", ",").split(",")
        elif isinstance(value, (list, tuple)):
            raw_values = value
        else:
            raw_values = [value]

        parsed = []
        for item in raw_values:
            text = str(item).strip().lower()
            if text:
                parsed.append(text)
        return parsed

    @staticmethod
    def normalize_owner_name(text):
        compact = re.sub(r"\s+", "", str(text or ""))
        compact = re.sub(r"[，。！？,.!?、；;：:“”\"'()（）]", "", compact)
        if not compact:
            return ""

        prefix_match = re.search(
            r"(?:我的名字叫|我的名字是|名字叫|名字是|我叫|我是)([一-龥A-Za-z][一-龥A-Za-z0-9·_-]{0,11})",
            compact,
        )
        if prefix_match:
            candidate = prefix_match.group(1)
        else:
            prompt_match = re.search(
                r"(?:说出|告诉我)(?:您的|你的)?名字([一-龥A-Za-z][一-龥A-Za-z0-9·_-]{0,11})$",
                compact,
            )
            candidate = prompt_match.group(1) if prompt_match else compact

        candidate = re.split(r"(?:请|确认|重新|重说|谢谢|机器人|主人)", candidate)[0]
        candidate = re.sub(r"[^一-龥A-Za-z0-9·_-]", "", candidate)
        if not candidate or len(candidate) > 12:
            return ""
        if any(marker in candidate for marker in ("名字", "姓名", "什么", "叫我")):
            return ""
        if not prefix_match and len(candidate) > 8:
            return ""
        return candidate

    @staticmethod
    def parse_owner_confirmation(text):
        compact = re.sub(r"\s+", "", str(text or ""))
        compact = re.sub(r"[，。！？,.!?、；;：:“”\"'()（）]", "", compact)
        if not compact:
            return None
        if compact in ("不是", "不对", "错了", "重新", "重说", "重来", "否"):
            return False
        if compact in ("确认", "确定", "是", "是的", "对", "对的", "正确", "没错", "好的", "我确认"):
            return True
        if compact.endswith(("确认", "确定", "是", "是的", "对", "对的", "正确", "没错", "好的")):
            return True
        if compact.endswith(("不是", "不对", "错了", "重新", "重说", "重来", "否")):
            return False
        if compact.startswith(("确认", "确定", "是的", "对的", "正确", "没错")):
            return True
        if compact.startswith(("不是", "不对", "错了", "重新", "重说", "重来", "否")):
            return False
        if "正确请回答确认" in compact or "不正确请回答重来" in compact:
            if compact.endswith("重来") and compact.count("重来") >= 2:
                return False
            if compact.endswith("确认") and compact.count("确认") >= 2:
                return True
            return None
        return None

    def current_asr_sequence(self):
        with self.lock:
            return int(self.asr_sequence)

    def wait_for_asr_publisher(self):
        deadline = time.time() + max(1.0, self.owner_enrollment_name_timeout)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            publisher_count = self.asr_sub.get_num_connections()
            if publisher_count > 0:
                rospy.loginfo(
                    "Owner enrollment ASR publisher ready: topic=%s publishers=%d",
                    self.asr_topic,
                    publisher_count,
                )
                return True
            rate.sleep()
        rospy.logwarn(
            "Owner enrollment has no ASR publisher on %s; check start_asr/start_voice and microphone input",
            self.asr_topic,
        )
        return False

    def wait_for_owner_name(self, sequence_after, timeout):
        cursor = int(sequence_after)
        deadline = time.time() + max(0.1, float(timeout))
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            transcript, cursor = self.collect_asr_since(cursor)
            if transcript:
                candidate_name = self.normalize_owner_name(transcript)
                if candidate_name:
                    return candidate_name, cursor
            rate.sleep()
        return "", cursor

    def wait_for_owner_confirmation(self, sequence_after, timeout):
        cursor = int(sequence_after)
        deadline = time.time() + max(0.1, float(timeout))
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            transcript, cursor = self.collect_asr_since(cursor)
            if transcript:
                decision = self.parse_owner_confirmation(transcript)
                if decision is not None:
                    return decision, cursor
            rate.sleep()
        return None, cursor

    def capture_owner_face_embeddings(self):
        if not self.face_model_ready or self.face_app is None:
            raise RuntimeError("InsightFace model is not ready for owner enrollment")

        deadline = time.time() + self.owner_enrollment_capture_seconds
        next_sample_time = 0.0
        last_image_time = 0.0
        embeddings = []
        sample_index = 0
        rate = rospy.Rate(10)

        while not rospy.is_shutdown() and time.time() < deadline:
            now = time.time()
            if now < next_sample_time:
                rate.sleep()
                continue

            with self.lock:
                image = None if self.latest_image is None else self.latest_image.copy()
                image_time = self.latest_image_time

            if image is None or image_time is None or image_time <= last_image_time:
                rate.sleep()
                continue
            last_image_time = float(image_time)

            try:
                faces = self.face_app.get(image)
            except Exception as exc:
                rospy.logwarn_throttle(2.0, "Owner enrollment face detection failed: %s", exc)
                rate.sleep()
                continue

            if len(faces or []) != 1:
                rospy.logwarn_throttle(
                    2.0,
                    "Owner enrollment requires exactly one visible face; detected=%d",
                    len(faces or []),
                )
                rate.sleep()
                continue

            face = self.select_largest_face(faces)
            bbox = getattr(face, "bbox", None)
            if bbox is None or len(bbox) < 4:
                rate.sleep()
                continue

            face_width = float(bbox[2]) - float(bbox[0])
            face_height = float(bbox[3]) - float(bbox[1])
            det_score = float(getattr(face, "det_score", 0.0) or 0.0)
            if (
                min(face_width, face_height) < self.owner_enrollment_min_face_size
                or det_score < self.face_det_thresh
            ):
                rospy.loginfo_throttle(
                    2.0,
                    "Owner enrollment face quality too low: size=%.0fx%.0f score=%.2f",
                    face_width,
                    face_height,
                    det_score,
                )
                rate.sleep()
                continue

            embedding = self.normalize_embedding(getattr(face, "embedding", None))
            if embedding is None:
                rate.sleep()
                continue

            embeddings.append(
                {
                    "path": "runtime_enrollment_%03d" % sample_index,
                    "embedding": embedding,
                }
            )
            sample_index += 1
            next_sample_time = now + self.owner_enrollment_sample_interval
            rospy.loginfo(
                "Owner enrollment captured face sample %d/%d",
                len(embeddings),
                self.owner_enrollment_min_samples,
            )
            rate.sleep()

        if len(embeddings) < self.owner_enrollment_min_samples:
            raise RuntimeError(
                "owner enrollment captured only %d/%d usable face samples"
                % (len(embeddings), self.owner_enrollment_min_samples)
            )

        mean_embedding = self.normalize_embedding(
            np.mean(
                np.asarray([item["embedding"] for item in embeddings], dtype=np.float32),
                axis=0,
            )
        )
        if mean_embedding is None:
            raise RuntimeError("owner enrollment produced an invalid face embedding")

        self.owner_face_embedding = mean_embedding
        self.owner_face_embeddings = embeddings
        self.face_ready = True
        rospy.loginfo(
            "Runtime owner enrollment complete: name=%s samples=%d threshold=%.2f",
            self.owner_name,
            len(self.owner_face_embeddings),
            self.face_accept_threshold,
        )

    def enroll_owner(self):
        if not self.owner_enrollment_enabled:
            return
        if not self.face_verify_enabled or not self.face_model_ready or self.face_app is None:
            raise RuntimeError("owner enrollment requires a ready InsightFace model")
        if not self.wait_for_asr_publisher():
            raise RuntimeError("owner enrollment ASR topic is not connected")

        confirmed_name = ""
        for attempt in range(self.owner_enrollment_retries):
            sequence_before_prompt = self.current_asr_sequence()
            if attempt == 0:
                self.say("请在我前方说出您的名字。")
            else:
                self.say("我没有听清，请再说一次您的名字。")
            rospy.sleep(self.owner_enrollment_asr_settle_seconds)
            candidate_name, sequence_after_name = self.wait_for_owner_name(
                sequence_before_prompt,
                self.owner_enrollment_name_timeout,
            )
            if not candidate_name:
                rospy.logwarn("Could not extract owner name from ASR")
                continue

            sequence_before_confirm = self.current_asr_sequence()
            self.say(
                "您说的是%s。请确认这个名字是否正确。正确请回答确认，不正确请回答重来。"
                % candidate_name
            )
            rospy.sleep(self.owner_enrollment_asr_settle_seconds)
            decision, _ = self.wait_for_owner_confirmation(
                sequence_before_confirm,
                self.owner_enrollment_confirm_timeout,
            )
            if decision is True:
                confirmed_name = candidate_name
                break
            if attempt + 1 < self.owner_enrollment_retries:
                self.say("好的，请重新说出您的名字。")

        if not confirmed_name:
            raise RuntimeError("owner name enrollment was not confirmed")

        self.owner_name = confirmed_name
        self.say(
            "好的，%s，请保持一个人站在我前方。我现在采集您的脸部特征。"
            % self.owner_name
        )
        self.capture_owner_face_embeddings()
        self.say("主人注册完成，我记住您的名字是%s。" % self.owner_name)

    @staticmethod
    def parse_waypoint_name_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            raw_values = value.replace(";", ",").split(",")
        elif isinstance(value, (list, tuple)):
            raw_values = value
        else:
            raw_values = [value]

        parsed = []
        for item in raw_values:
            text = str(item).strip()
            if text:
                parsed.append(text)
        return parsed

    @staticmethod
    def parse_waypoint_speech_names(value):
        if not isinstance(value, dict):
            return {}
        parsed = {}
        for waypoint_name, speech_name in value.items():
            key = str(waypoint_name).strip()
            text = str(speech_name).strip()
            if key and text:
                parsed[key] = text
        return parsed

    @staticmethod
    def select_largest_face(faces):
        if not faces:
            return None

        def face_priority(face):
            bbox = getattr(face, "bbox", None)
            if bbox is None or len(bbox) < 4:
                return 0.0
            width = max(0.0, float(bbox[2]) - float(bbox[0]))
            height = max(0.0, float(bbox[3]) - float(bbox[1]))
            det_score = float(getattr(face, "det_score", 1.0) or 1.0)
            return width * height * det_score

        return max(faces, key=face_priority)

    def face_reference_orientations(self, path, image):
        yield path, image
        if not self.face_reference_try_rotations:
            return
        yield path + ":rot90", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        yield path + ":rot270", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        yield path + ":rot180", cv2.rotate(image, cv2.ROTATE_180)

    def init_face_recognizer(self):
        model_dir = os.path.join(self.face_model_root, "models", self.face_model_name)
        cached_models = []
        if os.path.isdir(model_dir):
            cached_models = [name for name in os.listdir(model_dir) if name.endswith(".onnx")]
        if not cached_models and not self.face_auto_download:
            rospy.logwarn(
                "InsightFace model %s is not cached under %s. Run the install tool first or set face_auto_download=true.",
                self.face_model_name,
                model_dir,
            )
            return

        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            rospy.logwarn("InsightFace is not available: %s", exc)
            return

        providers = ["CPUExecutionProvider"] if self.face_ctx_id < 0 else None
        try:
            self.face_app = FaceAnalysis(
                name=self.face_model_name,
                root=self.face_model_root,
                providers=providers,
            )
            self.face_app.prepare(
                ctx_id=self.face_ctx_id,
                det_thresh=self.face_det_thresh,
                det_size=(self.face_det_size, self.face_det_size),
            )
            self.face_model_ready = True
            if not self.owner_reference_images:
                self.face_ready = False
                rospy.loginfo(
                    "InsightFace model ready for runtime owner enrollment: model=%s ctx_id=%d",
                    self.face_model_name,
                    self.face_ctx_id,
                )
                return
            owner_embeddings = []
            for path, image in self.owner_reference_images:
                loaded_for_photo = 0
                for oriented_path, oriented_image in self.face_reference_orientations(path, image):
                    faces = self.face_app.get(oriented_image)
                    owner_face = self.select_largest_face(faces)
                    if owner_face is None:
                        continue
                    embedding = self.normalize_embedding(owner_face.embedding)
                    if embedding is None:
                        continue
                    owner_embeddings.append({"path": oriented_path, "embedding": embedding})
                    loaded_for_photo += 1
                if loaded_for_photo == 0:
                    rospy.logwarn("No face found in owner reference photo, skipping: %s", path)
            if not owner_embeddings:
                rospy.logwarn("No usable owner face embeddings loaded from: %s", self.owner_image_path)
                return
            mean_embedding = self.normalize_embedding(
                np.mean(np.asarray([item["embedding"] for item in owner_embeddings], dtype=np.float32), axis=0)
            )
            self.owner_face_embedding = mean_embedding
            self.owner_face_embeddings = owner_embeddings
            self.face_ready = True
            rospy.loginfo(
                "Face verification ready: model=%s ctx_id=%d det_size=%d det_thresh=%.2f threshold=%.2f references=%d photos=%d ref_rotations=%s",
                self.face_model_name,
                self.face_ctx_id,
                self.face_det_size,
                self.face_det_thresh,
                self.face_accept_threshold,
                len(self.owner_face_embeddings),
                len(self.owner_reference_images),
                str(self.face_reference_try_rotations),
            )
        except Exception as exc:
            self.face_app = None
            self.face_model_ready = False
            self.owner_face_embedding = None
            self.owner_face_embeddings = []
            self.face_ready = False
            rospy.logwarn("Failed to initialize face verification: %s", exc)

    def init_action_recognizer(self):
        if not self.action_recognition_enabled:
            rospy.loginfo("Owner action recognition disabled")
            return
        try:
            self.qwen_action_core, self.qwen_action_core_path = load_qwen_action_core(
                self.action_core_path
            )
            if self.action_pose_enabled:
                pose_model_path = self.action_model_path
                if not pose_model_path:
                    pose_model_path = self.qwen_action_core.PoseActionAnalyzer.resolve_model_path("")
                self.qwen_pose_analyzer = self.qwen_action_core.PoseActionAnalyzer(
                    pose_model_path,
                    self.action_device,
                    max(160, self.action_imgsz),
                    self.action_conf,
                    self.action_iou,
                    max(1, self.action_max_det),
                )
                pose_ready, pose_message = self.qwen_pose_analyzer.initialize()
                if pose_ready:
                    rospy.loginfo(
                        "Qwen pose helper ready: model=%s device=%s",
                        pose_model_path,
                        self.action_device,
                    )
                else:
                    self.qwen_pose_analyzer = None
                    rospy.logwarn("Qwen pose helper disabled: %s", pose_message)

            if self.action_pointcloud_enabled:
                self.qwen_pointcloud_analyzer = self.qwen_action_core.PointCloudGroundAnalyzer(
                    self.action_pointcloud_camera_height,
                    self.action_pointcloud_ground_height_limit,
                    self.action_pointcloud_furniture_height_limit,
                    self.action_pointcloud_max_age,
                    self.action_pointcloud_stride,
                    self.action_pointcloud_min_samples,
                    self.action_pointcloud_roi_padding,
                    self.action_pointcloud_depth_percentile,
                    self.action_pointcloud_surface_band,
                    self.action_pointcloud_frame_mode,
                    self.action_pointcloud_anchor_radius,
                    self.action_pointcloud_local_ground_padding,
                    self.action_pointcloud_elevated_delta,
                )

            self.action_ready = True
            self.warmup_qwen_action_model()
            rospy.loginfo(
                "Owner Qwen action recognition ready: model=%s core=%s seconds=%.1f frames=%d llm_frames=%d",
                self.action_llm_model,
                self.qwen_action_core_path,
                self.action_sample_seconds,
                self.action_frame_count,
                self.action_llm_frame_count,
            )
        except Exception as exc:
            self.qwen_action_core = None
            self.qwen_action_core_path = ""
            self.qwen_pose_analyzer = None
            self.qwen_pointcloud_analyzer = None
            self.action_ready = False
            rospy.logwarn("Failed to initialize owner Qwen action recognition: %s", exc)

    def get_latest_action_image(self):
        with self.lock:
            if self.latest_image is None:
                return None
            return self.latest_image.copy()

    def encode_qwen_action_frame(self, frame):
        image = frame
        if self.action_image_max_width > 0 and image.shape[1] > self.action_image_max_width:
            scale = float(self.action_image_max_width) / float(image.shape[1])
            image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.action_jpeg_quality],
        )
        if not ok:
            raise RuntimeError("could not encode owner action frame")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def call_qwen_action(self, images_b64, prompt=None, max_tokens=None):
        images = [images_b64] if isinstance(images_b64, str) else list(images_b64)
        if not images:
            raise RuntimeError("no image supplied to Qwen action recognizer")
        rospy.loginfo(
            "Sending owner action request to Qwen: model=%s images=%d image_bytes≈%dKB num_ctx=%d",
            self.action_llm_model,
            len(images),
            sum(len(image) for image in images) // 1024,
            self.action_llm_num_ctx,
        )
        payload = {
            "model": self.action_llm_model,
            "stream": False,
            "think": False,
            "format": "json",
            "keep_alive": self.action_llm_keep_alive,
            "messages": [
                {
                    "role": "user",
                    "content": prompt or self.qwen_action_core.PROMPT,
                    "images": images,
                }
            ],
            "options": {
                "temperature": 0,
                "top_p": 0.7,
                "num_predict": max_tokens or self.action_llm_max_tokens,
                "num_ctx": self.action_llm_num_ctx,
                "num_gpu": self.action_llm_num_gpu,
            },
        }
        request = urllib.request.Request(
            self.action_llm_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.time()
        try:
            with self.ollama_request_lock:
                with urllib.request.urlopen(
                    request,
                    timeout=max(1.0, float(self.action_llm_timeout)),
                ) as response:
                    raw = response.read().decode("utf-8")
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                "Qwen action request timed out after %.1fs: %s"
                % (time.time() - started, exc)
            )
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:
                detail = ""
            rospy.logerr(
                "Qwen owner action HTTP error: code=%s model=%s detail=%s",
                exc.code,
                self.action_llm_model,
                detail,
            )
            raise RuntimeError(
                "Qwen action HTTP %s for model %s: %s"
                % (exc.code, self.action_llm_model, detail or exc.reason)
            )
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Qwen action request failed; check Ollama and model %s: %s"
                % (self.action_llm_model, exc)
            )

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid Qwen action response: %s" % exc)
        content = result.get("message", {}).get("content", "").strip()
        if not content:
            raise RuntimeError("Qwen action returned empty content: %s" % raw)
        rospy.loginfo(
            "Qwen owner action response received in %.2fs",
            time.time() - started,
        )
        return content, time.time() - started

    def warmup_qwen_action_model(self):
        yolo_paused = False
        try:
            if self.action_pause_yolo:
                self.set_yolo_paused(True)
                yolo_paused = True
                if self.action_yolo_pause_settle_seconds > 0.0:
                    rospy.sleep(self.action_yolo_pause_settle_seconds)
            deadline = time.time() + max(5.0, self.action_warmup_wait_timeout)
            frame = self.get_latest_action_image()
            while frame is None and not rospy.is_shutdown() and time.time() < deadline:
                rospy.sleep(0.05)
                frame = self.get_latest_action_image()
            if frame is None:
                raise RuntimeError("no camera frame available for Qwen warm-up")
            if self.qwen_pose_analyzer is not None and self.qwen_pose_analyzer.ready:
                rospy.loginfo("Warming up owner pose helper")
                self.qwen_pose_analyzer.warmup(frame)
                rospy.loginfo("Owner pose helper warm-up complete")

            warmup_prompt = (
                "/no_think\n"
                "只输出一行 JSON："
                '{"action":"unknown","place":"unknown"}。'
                "这是视觉模型预热请求，不要解释。"
            )
            encoded = self.encode_qwen_action_frame(frame)
            last_error = None
            for attempt in range(1, self.action_warmup_retries + 1):
                if rospy.is_shutdown():
                    return
                try:
                    content, elapsed = self.call_qwen_action(
                        [encoded],
                        prompt=warmup_prompt,
                        max_tokens=16,
                    )
                    self.qwen_warmup_error = ""
                    self.qwen_warmup_done.set()
                    rospy.loginfo(
                        "Qwen owner-action warm-up complete in %.2fs: %s",
                        elapsed,
                        content,
                    )
                    return
                except Exception as exc:
                    last_error = exc
                    if attempt < self.action_warmup_retries:
                        rospy.logwarn(
                            "Qwen owner-action warm-up attempt %d failed: %s; retrying in %.1fs",
                            attempt,
                            exc,
                            self.action_warmup_retry_delay,
                        )
                        rospy.sleep(self.action_warmup_retry_delay)
            self.qwen_warmup_error = str(last_error)
            self.qwen_warmup_done.set()
            rospy.logwarn("Qwen owner-action warm-up failed: %s", self.qwen_warmup_error)
        except Exception as exc:
            self.qwen_warmup_error = str(exc)
            self.qwen_warmup_done.set()
            rospy.logwarn("Qwen owner-action warm-up failed: %s", exc)
        finally:
            if yolo_paused:
                self.set_yolo_paused(False)

    def wait_for_qwen_action_warmup(self):
        if self.qwen_warmup_done.wait(timeout=self.action_warmup_wait_timeout):
            if self.qwen_warmup_error:
                rospy.logwarn(
                    "Continuing owner action recognition after warm-up failure: %s",
                    self.qwen_warmup_error,
                )
            return not self.qwen_warmup_error
        rospy.logwarn(
            "Qwen owner-action warm-up is still running after %.1fs",
            self.action_warmup_wait_timeout,
        )
        return False

    def ensure_qwen_action_warmup(self):
        if not self.action_ready or self.qwen_action_core is None:
            return False
        if self.qwen_warmup_done.is_set() and self.qwen_warmup_error:
            rospy.logwarn(
                "Qwen owner-action warm-up already failed; continuing without a duplicate startup retry"
            )
        return self.wait_for_qwen_action_warmup()

    def extract_owner_pose_features(self, frames, samples, target_centers):
        if self.qwen_pose_analyzer is None or not self.qwen_pose_analyzer.ready:
            return [], "unknown", 0.0, "pose disabled"

        rospy.loginfo("Running pose helper on %d owner action frames", len(frames))
        pose_features = self.qwen_pose_analyzer.extract_features(
            frames,
            target_centers=target_centers,
        )
        mapped_features = []
        for index, feature in enumerate(pose_features):
            frame_index = int(feature.get("frame_index", index))
            sample = samples[frame_index] if frame_index < len(samples) else samples[index]
            mapped_features.append(
                self.map_qwen_pose_feature_to_full_frame(feature, sample)
            )
        pose_action, pose_confidence, pose_reason = self.qwen_pose_analyzer.classify(
            mapped_features
        )
        rospy.loginfo(
            "Pose fusion complete: features=%d action=%s confidence=%.2f",
            len(mapped_features),
            pose_action,
            pose_confidence,
        )
        return mapped_features, pose_action, pose_confidence, pose_reason

    def extract_owner_pose_features_worker(
        self,
        result_holder,
        frames,
        samples,
        target_centers,
    ):
        try:
            result_holder["value"] = self.extract_owner_pose_features(
                frames,
                samples,
                target_centers,
            )
        except Exception as exc:
            result_holder["error"] = exc

    def capture_qwen_action_frames(self):
        frame_count = max(1, self.action_frame_count)
        duration = max(0.0, self.action_sample_seconds)
        if frame_count == 1:
            sample_times = [0.0]
        else:
            interval = duration / float(frame_count - 1)
            sample_times = [interval * index for index in range(frame_count)]

        samples = []
        capture_started = time.time()
        for sample_index, sample_time in enumerate(sample_times, start=1):
            while not rospy.is_shutdown():
                remaining = sample_time - (time.time() - capture_started)
                if remaining <= 0.0:
                    break
                rospy.sleep(min(0.05, remaining))
            if rospy.is_shutdown():
                break
            frame, source_meta = self.snapshot_owner_action_image()
            if frame is not None:
                samples.append(
                    {
                        "image": frame,
                        "full_image": source_meta.get("debug_image", frame),
                        "source_meta": source_meta,
                    }
                )
            rospy.loginfo_throttle(
                1.0,
                "Captured owner action frame %d/%d",
                sample_index,
                frame_count,
            )

        if not samples:
            raise RuntimeError("no camera frames available for owner action recognition")
        return samples

    @staticmethod
    def map_qwen_pose_feature_to_full_frame(feature, sample):
        mapped = dict(feature)
        source_meta = sample.get("source_meta") or {}
        full_image = sample.get("full_image")
        if full_image is None:
            return mapped
        full_height, full_width = full_image.shape[:2]
        crop_box = source_meta.get("crop_box")
        if crop_box is None:
            return mapped

        crop_x1, crop_y1, _crop_x2, _crop_y2 = [
            float(value) for value in crop_box
        ]
        crop_width = max(1.0, float(crop_box[2]) - crop_x1)
        crop_height = max(1.0, float(crop_box[3]) - crop_y1)

        def map_point(point):
            if point is None:
                return None
            return (float(point[0]) + crop_x1, float(point[1]) + crop_y1)

        bbox = mapped.get("bbox")
        if bbox is not None and len(bbox) == 4:
            mapped_bbox = (
                float(bbox[0]) + crop_x1,
                float(bbox[1]) + crop_y1,
                float(bbox[2]) + crop_x1,
                float(bbox[3]) + crop_y1,
            )
            mapped["bbox"] = mapped_bbox
            mapped["center_y"] = (
                0.5 * (mapped_bbox[1] + mapped_bbox[3]) / max(1.0, float(full_height))
            )
            mapped["height"] = (
                max(1.0, mapped_bbox[3] - mapped_bbox[1])
                / max(1.0, float(full_height))
            )
            mapped["aspect"] = max(
                1e-6,
                (mapped_bbox[2] - mapped_bbox[0])
                / max(1.0, mapped_bbox[3] - mapped_bbox[1]),
            )

        body_bbox = mapped.get("body_bbox")
        if body_bbox is not None and len(body_bbox) == 4:
            mapped["body_bbox"] = (
                float(body_bbox[0]) + crop_x1,
                float(body_bbox[1]) + crop_y1,
                float(body_bbox[2]) + crop_x1,
                float(body_bbox[3]) + crop_y1,
            )
        if mapped.get("body_anchors"):
            mapped["body_anchors"] = [
                map_point(point) for point in mapped["body_anchors"]
            ]
        for side in ("left", "right"):
            wrist_x_key = "%s_wrist_x" % side
            wrist_y_key = "%s_wrist_y" % side
            wrist_x = mapped.get(wrist_x_key)
            wrist_y = mapped.get(wrist_y_key)
            if wrist_x is not None:
                mapped[wrist_x_key] = (
                    float(wrist_x) * crop_width + crop_x1
                ) / max(1.0, float(full_width))
            if wrist_y is not None:
                mapped[wrist_y_key] = (
                    float(wrist_y) * crop_height + crop_y1
                ) / max(1.0, float(full_height))
        return mapped

    def get_latest_action_pointcloud(self):
        with self.lock:
            return self.latest_pointcloud, self.latest_pointcloud_time

    def analyze_qwen_ground_relation(self, frame, pose_features):
        if not self.action_pointcloud_enabled or self.qwen_pointcloud_analyzer is None:
            return {"status": "unknown", "reason": "point cloud disabled"}
        if frame is None:
            return {"status": "unknown", "reason": "no final image"}
        if not pose_features:
            return {"status": "unknown", "reason": "no pose bbox"}
        final_feature = pose_features[-1]
        bbox = final_feature.get("body_bbox") or final_feature.get("bbox")
        anchors = final_feature.get("body_anchors") or []
        cloud, cloud_stamp = self.get_latest_action_pointcloud()
        cloud_age = time.time() - cloud_stamp if cloud_stamp else None
        return self.qwen_pointcloud_analyzer.analyze(
            cloud,
            cloud_age,
            frame.shape,
            bbox,
            anchors=anchors,
        )

    def snapshot_owner_action_image(self):
        with self.lock:
            if self.latest_image is None:
                return None, None
            image = self.latest_image.copy()
            detections = list(self.latest_detections)

        height, width = image.shape[:2]
        full_meta = {
            "frame_width": width,
            "frame_height": height,
            "crop_box": None,
            "debug_image": image,
            "pose_target_center_norm": self.owner_track_center if self.owner_track_center is not None else 0.5,
        }

        if not self.action_use_owner_roi or not detections:
            return image, full_meta

        best = None
        best_score = -1e9
        target_center = self.owner_track_center if self.owner_track_center is not None else 0.5
        for det in detections:
            class_name = getattr(det, "class_name", "")
            if class_name and class_name != "person":
                continue
            if float(det.score) < self.detection_min_score:
                continue
            xmin = clamp(int(det.xmin), 0, width - 1)
            ymin = clamp(int(det.ymin), 0, height - 1)
            xmax = clamp(int(det.xmax), 0, width - 1)
            ymax = clamp(int(det.ymax), 0, height - 1)
            if xmax <= xmin or ymax <= ymin:
                continue

            center_norm = float(det.center_x) / max(1.0, float(width))
            area_ratio = float((xmax - xmin) * (ymax - ymin)) / max(1.0, float(width * height))
            score = float(det.score) + area_ratio * 2.0 - abs(center_norm - target_center) * 2.0
            if score > best_score:
                best_score = score
                det_center_y = float(getattr(det, "center_y", (float(ymin) + float(ymax)) * 0.5))
                best = (xmin, ymin, xmax, ymax, center_norm, float(det.center_x), det_center_y)

        if best is None:
            return image, full_meta

        xmin, ymin, xmax, ymax, center_norm, det_center_x, det_center_y = best
        box_w = xmax - xmin
        box_h = ymax - ymin
        pad_x = int(box_w * clamp(self.action_roi_padding, 0.0, 1.0))
        pad_y = int(box_h * clamp(self.action_roi_padding, 0.0, 1.0))
        cx1 = clamp(xmin - pad_x, 0, width - 1)
        cy1 = clamp(ymin - pad_y, 0, height - 1)
        cx2 = clamp(xmax + pad_x, 0, width - 1)
        cy2 = clamp(ymax + pad_y, 0, height - 1)
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return image, full_meta

        self.owner_track_center = center_norm
        crop_width = max(1.0, float(cx2 - cx1))
        pose_target_center_norm = clamp((det_center_x - float(cx1)) / crop_width, 0.0, 1.0)
        return crop, {
            "frame_width": width,
            "frame_height": height,
            "crop_box": (cx1, cy1, cx2, cy2),
            "debug_image": image,
            "det_bbox": (xmin, ymin, xmax, ymax),
            "det_center_y_norm": det_center_y / max(1.0, float(height)),
            "det_aspect": float(box_w) / max(1.0, float(box_h)),
            "det_height_norm": float(box_h) / max(1.0, float(height)),
            "pose_target_center_norm": pose_target_center_norm,
        }

    def select_owner_pose(self, result, image_width, target_center=None):
        if result is None or result.keypoints is None or result.boxes is None:
            return None
        try:
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            keypoints_conf = result.keypoints.conf.cpu().numpy()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            boxes_conf = result.boxes.conf.cpu().numpy()
        except Exception as exc:
            rospy.logwarn("Failed to read pose result tensors: %s", exc)
            return None

        count = min(len(keypoints_xy), len(boxes_xyxy))
        if count <= 0:
            return None

        best = None
        best_score = -1e9
        if target_center is None:
            target_center = self.owner_track_center
        for index in range(count):
            kxy = keypoints_xy[index]
            kconf = keypoints_conf[index]
            valid_count = int(np.sum(kconf >= self.action_min_keypoint_conf))
            if valid_count < 5:
                continue

            bbox = boxes_xyxy[index]
            center_norm = ((float(bbox[0]) + float(bbox[2])) * 0.5) / max(1.0, float(image_width))
            center_penalty = abs(center_norm - target_center) if target_center is not None else abs(center_norm - 0.5)
            score = float(boxes_conf[index]) + valid_count * 0.04 - center_penalty * 1.5
            if score > best_score:
                best_score = score
                best = {
                    "keypoints_xy": kxy,
                    "keypoints_conf": kconf,
                    "bbox": bbox,
                    "box_conf": float(boxes_conf[index]),
                    "center_norm": center_norm,
                    "valid_count": valid_count,
                }
        return best

    @staticmethod
    def angle_at(point_a, point_b, point_c):
        vec_a = np.asarray(point_a, dtype=np.float32) - np.asarray(point_b, dtype=np.float32)
        vec_c = np.asarray(point_c, dtype=np.float32) - np.asarray(point_b, dtype=np.float32)
        denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_c))
        if denom <= 1e-6:
            return None
        cosine = clamp(float(np.dot(vec_a, vec_c) / denom), -1.0, 1.0)
        return math.degrees(math.acos(cosine))

    @staticmethod
    def median_value(values):
        cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        if not cleaned:
            return None
        return float(np.median(np.asarray(cleaned, dtype=np.float32)))

    def build_pose_feature(self, pose, image_shape, source_meta=None):
        height, width = image_shape[:2]
        kxy = pose["keypoints_xy"]
        kconf = pose["keypoints_conf"]
        bbox = pose["bbox"]
        box_w = max(1.0, float(bbox[2]) - float(bbox[0]))
        box_h = max(1.0, float(bbox[3]) - float(bbox[1]))
        aspect = box_w / box_h
        center_y_norm = ((float(bbox[1]) + float(bbox[3])) * 0.5) / max(1.0, float(height))
        full_center_y_norm = center_y_norm
        full_box_height_norm = box_h / max(1.0, float(height))
        full_aspect = aspect
        det_aspect = None
        if source_meta:
            frame_height = float(source_meta.get("frame_height") or height)
            det_aspect = source_meta.get("det_aspect")
            crop_box = source_meta.get("crop_box")
            if crop_box is not None:
                crop_x1, crop_y1, _crop_x2, _crop_y2 = [float(value) for value in crop_box]
                full_x1 = crop_x1 + float(bbox[0])
                full_y1 = crop_y1 + float(bbox[1])
                full_x2 = crop_x1 + float(bbox[2])
                full_y2 = crop_y1 + float(bbox[3])
                full_box_w = max(1.0, full_x2 - full_x1)
                full_box_h = max(1.0, full_y2 - full_y1)
                full_center_y_norm = ((full_y1 + full_y2) * 0.5) / max(1.0, frame_height)
                full_box_height_norm = full_box_h / max(1.0, frame_height)
                full_aspect = full_box_w / full_box_h
            det_center_y_norm = source_meta.get("det_center_y_norm")
            if det_center_y_norm is not None:
                full_center_y_norm = float(det_center_y_norm)
            det_aspect = source_meta.get("det_aspect")
            if det_aspect is not None and math.isfinite(float(det_aspect)):
                full_aspect = max(full_aspect, float(det_aspect))
            det_height_norm = source_meta.get("det_height_norm")
            if det_height_norm is not None and math.isfinite(float(det_height_norm)):
                full_box_height_norm = float(det_height_norm)

        def point(index):
            if index >= len(kxy) or float(kconf[index]) < self.action_min_keypoint_conf:
                return None
            return (float(kxy[index][0]), float(kxy[index][1]))

        def midpoint(left, right):
            if left is None or right is None:
                return None
            return ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)

        left_shoulder = point(5)
        right_shoulder = point(6)
        left_wrist = point(9)
        right_wrist = point(10)
        left_hip = point(11)
        right_hip = point(12)
        left_knee = point(13)
        right_knee = point(14)
        left_ankle = point(15)
        right_ankle = point(16)
        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip)
        knee_mid = midpoint(left_knee, right_knee)

        torso_verticality = None
        if shoulder_mid is not None and hip_mid is not None:
            torso_dx = hip_mid[0] - shoulder_mid[0]
            torso_dy = hip_mid[1] - shoulder_mid[1]
            torso_dist = math.hypot(torso_dx, torso_dy)
            if torso_dist > 1e-6:
                torso_verticality = abs(torso_dy) / torso_dist

        knee_angles = []
        left_knee_angle = self.angle_at(left_hip, left_knee, left_ankle) if left_hip and left_knee and left_ankle else None
        right_knee_angle = self.angle_at(right_hip, right_knee, right_ankle) if right_hip and right_knee and right_ankle else None
        if left_knee_angle is not None:
            knee_angles.append(left_knee_angle)
        if right_knee_angle is not None:
            knee_angles.append(right_knee_angle)
        knee_angle = self.median_value(knee_angles)

        left_wrist_above = bool(left_wrist and left_shoulder and left_wrist[1] < left_shoulder[1] - box_h * 0.05)
        right_wrist_above = bool(right_wrist and right_shoulder and right_wrist[1] < right_shoulder[1] - box_h * 0.05)
        left_wrist_x_norm = left_wrist[0] / max(1.0, float(width)) if left_wrist else None
        right_wrist_x_norm = right_wrist[0] / max(1.0, float(width)) if right_wrist else None

        torso_v = torso_verticality if torso_verticality is not None else 1.0
        posture_aspect = max(aspect, full_aspect)
        lying_like = (torso_v < 0.42) or (posture_aspect > 1.20 and torso_v < 0.68)

        knees_near_hips = False
        any_knee_near_hip = False
        if hip_mid is not None and knee_mid is not None:
            knees_near_hips = abs(knee_mid[1] - hip_mid[1]) < box_h * self.action_sitting_knee_hip_y_ratio

        thigh_horizontal = False
        ankle_near_hip = False
        for hip, knee, ankle in ((left_hip, left_knee, left_ankle), (right_hip, right_knee, right_ankle)):
            if hip is not None and knee is not None:
                thigh_dx = abs(knee[0] - hip[0])
                thigh_dy = abs(knee[1] - hip[1])
                thigh_len = math.hypot(thigh_dx, thigh_dy)
                if thigh_dy < box_h * self.action_sitting_knee_hip_y_ratio:
                    any_knee_near_hip = True
                if (
                    thigh_len > box_h * 0.08
                    and thigh_dx > thigh_dy * self.action_sitting_thigh_horizontal_ratio
                ):
                    thigh_horizontal = True
            if hip is not None and ankle is not None:
                if abs(ankle[1] - hip[1]) < box_h * self.action_sitting_ankle_hip_y_ratio:
                    ankle_near_hip = True

        bent_knee = knee_angle is not None and knee_angle < self.action_sitting_knee_angle_max
        compact_seated_box = (
            posture_aspect >= self.action_sitting_compact_aspect_min
            and posture_aspect <= 1.20
            and torso_v < 0.97
        )
        sitting_score = 0.0
        if bent_knee:
            sitting_score += 1.0
        if knees_near_hips:
            sitting_score += 1.0
        elif any_knee_near_hip:
            sitting_score += 0.75
        if thigh_horizontal:
            sitting_score += 0.55
        if ankle_near_hip:
            sitting_score += 0.45
        if compact_seated_box:
            sitting_score += 0.25

        sitting_structural_support = (
            bent_knee
            or knees_near_hips
            or (any_knee_near_hip and (thigh_horizontal or ankle_near_hip))
        )
        sitting_min_torso = self.action_sitting_min_torso_verticality
        sitting_support_threshold = self.action_sitting_support_score
        sitting_support_like = (
            torso_v > sitting_min_torso
            and sitting_score >= sitting_support_threshold
            and sitting_structural_support
            and not lying_like
        )
        sitting_like = (
            torso_v > sitting_min_torso
            and sitting_score >= 1.0
            and sitting_structural_support
            and not lying_like
        )
        upright_like = torso_v > 0.65 and posture_aspect < 1.05

        return {
            "aspect": aspect,
            "full_aspect": full_aspect,
            "det_aspect": det_aspect,
            "center_y_norm": center_y_norm,
            "full_center_y_norm": full_center_y_norm,
            "full_box_height_norm": full_box_height_norm,
            "torso_verticality": torso_verticality,
            "knee_angle": knee_angle,
            "left_wrist_above": left_wrist_above,
            "right_wrist_above": right_wrist_above,
            "left_wrist_x_norm": left_wrist_x_norm,
            "right_wrist_x_norm": right_wrist_x_norm,
            "lying_like": lying_like,
            "any_knee_near_hip": any_knee_near_hip,
            "thigh_horizontal": thigh_horizontal,
            "ankle_near_hip": ankle_near_hip,
            "compact_seated_box": compact_seated_box,
            "sitting_structural_support": sitting_structural_support,
            "sitting_score": sitting_score,
            "sitting_support_like": sitting_support_like,
            "sitting_like": sitting_like,
            "upright_like": upright_like,
            "box_conf": pose["box_conf"],
            "valid_count": pose["valid_count"],
        }

    def predict_owner_pose_feature(self, image, source_meta=None):
        if image is None or image.size == 0 or not self.action_ready or self.action_pose_model is None:
            return None
        try:
            use_half = self.action_half and str(self.action_device).startswith("cuda")
            results = self.action_pose_model.predict(
                image,
                imgsz=self.action_imgsz,
                conf=self.action_conf,
                iou=self.action_iou,
                device=self.action_device,
                half=use_half,
                max_det=max(1, self.action_max_det),
                verbose=False,
            )
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Owner action pose inference failed: %s", exc)
            return None
        if not results:
            return None
        target_center = source_meta.get("pose_target_center_norm") if source_meta else None
        pose = self.select_owner_pose(results[0], image.shape[1], target_center=target_center)
        if pose is None:
            return None
        return self.build_pose_feature(pose, image.shape, source_meta=source_meta)

    def classify_owner_action(self, features):
        usable = [feature for feature in features if feature is not None]
        if not usable:
            return "unknown", 0.0, "no pose"
        if len(usable) < max(1, self.action_min_pose_samples):
            return "unknown", 0.15, "too few pose samples"

        sample_count = float(len(usable))
        first = usable[:max(1, len(usable) // 3)]
        last = usable[-max(1, len(usable) // 3):]

        first_center = self.median_value([feature["center_y_norm"] for feature in first])
        last_center = self.median_value([feature["center_y_norm"] for feature in last])
        first_full_center = self.median_value([feature.get("full_center_y_norm") for feature in first])
        last_full_center = self.median_value([feature.get("full_center_y_norm") for feature in last])
        first_torso = self.median_value([feature["torso_verticality"] for feature in first])
        last_torso = self.median_value([feature["torso_verticality"] for feature in last])
        first_aspect = self.median_value([feature.get("full_aspect", feature["aspect"]) for feature in first])
        last_aspect = self.median_value([feature.get("full_aspect", feature["aspect"]) for feature in last])
        first_height = self.median_value([feature.get("full_box_height_norm") for feature in first])
        last_height = self.median_value([feature.get("full_box_height_norm") for feature in last])
        first_det_aspect = self.median_value([feature.get("det_aspect") for feature in first])
        last_det_aspect = self.median_value([feature.get("det_aspect") for feature in last])
        first_lie_ratio = sum(1 for feature in first if feature["lying_like"]) / float(len(first))
        last_lie_ratio = sum(1 for feature in last if feature["lying_like"]) / float(len(last))
        first_upright_ratio = sum(1 for feature in first if feature["upright_like"]) / float(len(first))
        median_aspect = self.median_value([feature.get("full_aspect", feature["aspect"]) for feature in usable])
        median_torso = self.median_value([feature["torso_verticality"] for feature in usable])
        median_knee = self.median_value([feature["knee_angle"] for feature in usable])

        center_start = first_full_center if first_full_center is not None else first_center
        center_end = last_full_center if last_full_center is not None else last_center
        center_drop = (center_end - center_start) if center_start is not None and center_end is not None else 0.0
        torso_drop = (first_torso - last_torso) if first_torso is not None and last_torso is not None else 0.0
        aspect_gain = (last_aspect - first_aspect) if first_aspect is not None and last_aspect is not None else 0.0
        lie_ratio_gain = last_lie_ratio - first_lie_ratio
        height_ratio = (last_height / first_height) if first_height and last_height else None
        height_shrunk = height_ratio is not None and height_ratio < self.action_fall_height_shrink_ratio
        became_lying = (
            last_lie_ratio >= self.action_fall_late_lie_ratio
            and lie_ratio_gain >= self.action_fall_lie_ratio_gain
        )
        transition_signals = [
            center_drop > self.action_fall_center_drop,
            torso_drop > self.action_fall_torso_drop,
            aspect_gain > self.action_fall_aspect_gain,
            height_shrunk,
            became_lying,
        ]
        transition_signal_count = sum(1 for matched in transition_signals if matched)
        early_detection_not_upright = (
            first_det_aspect is not None
            and first_det_aspect > self.action_fall_early_det_aspect_max
        )
        static_detection_lying = (
            first_det_aspect is not None
            and last_det_aspect is not None
            and first_det_aspect >= self.action_fall_static_det_aspect_min
            and last_det_aspect >= self.action_fall_static_det_aspect_min
        )
        static_lying = (
            first_lie_ratio >= self.action_fall_static_lie_ratio
            and last_lie_ratio >= self.action_fall_static_lie_ratio
        ) or static_detection_lying
        plausible_start = (
            first_upright_ratio >= self.action_fall_early_upright_ratio
            and first_lie_ratio <= self.action_fall_early_lie_ratio_max
            and not early_detection_not_upright
        )

        rospy.loginfo(
            "Owner action fall metrics: samples=%d first_lie=%.2f last_lie=%.2f first_upright=%.2f "
            "center_drop=%.2f torso_drop=%.2f aspect_gain=%.2f height_ratio=%s det_aspect=%s->%s transition_signals=%d static_lying=%s",
            len(usable),
            first_lie_ratio,
            last_lie_ratio,
            first_upright_ratio,
            center_drop,
            torso_drop,
            aspect_gain,
            "%.2f" % height_ratio if height_ratio is not None else "NA",
            "%.2f" % first_det_aspect if first_det_aspect is not None else "NA",
            "%.2f" % last_det_aspect if last_det_aspect is not None else "NA",
            transition_signal_count,
            static_lying,
        )
        if (
            not static_lying
            and plausible_start
            and last_lie_ratio >= self.action_fall_late_lie_ratio
            and transition_signal_count >= self.action_fall_min_transition_signals
        ):
            confidence = clamp(
                0.58
                + max(0.0, center_drop) * 1.4
                + max(0.0, torso_drop) * 0.5
                + max(0.0, aspect_gain) * 0.35
                + max(0.0, lie_ratio_gain) * 0.25,
                0.0,
                0.98,
            )
            return "falling", confidence, "fall transition"

        lie_ratio = sum(1 for feature in usable if feature["lying_like"]) / sample_count
        sit_ratio = sum(1 for feature in usable if feature["sitting_like"]) / sample_count
        sit_support_ratio = sum(1 for feature in usable if feature.get("sitting_support_like")) / sample_count
        median_sitting_score = self.median_value([feature.get("sitting_score") for feature in usable])
        required_ratio = clamp(self.action_static_required_ratio, 0.50, 0.95)
        sitting_relaxed_ratio = clamp(self.action_sitting_relaxed_ratio, 0.35, required_ratio)
        sitting_support_score = max(0.0, self.action_sitting_support_score)
        rospy.loginfo(
            "Owner action feature summary: samples=%d lie_ratio=%.2f sit_ratio=%.2f sit_support=%.2f "
            "aspect=%s torso=%s knee=%s sit_score=%s",
            len(usable),
            lie_ratio,
            sit_ratio,
            sit_support_ratio,
            "%.2f" % median_aspect if median_aspect is not None else "NA",
            "%.2f" % median_torso if median_torso is not None else "NA",
            "%.0f" % median_knee if median_knee is not None else "NA",
            "%.2f" % median_sitting_score if median_sitting_score is not None else "NA",
        )
        if lie_ratio >= required_ratio and (median_aspect is None or median_aspect > 1.05):
            return "lying", clamp(0.50 + lie_ratio * 0.45, 0.0, 0.95), "horizontal body posture"

        wave_scores = []
        for side in ("left", "right"):
            above_key = "%s_wrist_above" % side
            x_key = "%s_wrist_x_norm" % side
            wrist_x = [feature[x_key] for feature in usable if feature[above_key] and feature[x_key] is not None]
            if len(wrist_x) >= 3:
                x_range = max(wrist_x) - min(wrist_x)
                above_ratio = len(wrist_x) / sample_count
                if x_range > 0.11 and above_ratio >= 0.45:
                    wave_scores.append(clamp(0.50 + x_range * 2.0 + above_ratio * 0.25, 0.0, 0.95))
        if wave_scores:
            return "waving", max(wave_scores), "raised wrist motion"

        if sit_ratio >= required_ratio and median_torso is not None and median_torso > self.action_sitting_min_torso_verticality:
            return "sitting", clamp(0.50 + sit_ratio * 0.45, 0.0, 0.95), "bent seated posture"
        if (
            lie_ratio < required_ratio
            and median_torso is not None
            and median_torso > self.action_sitting_min_torso_verticality
            and median_sitting_score is not None
            and median_sitting_score >= sitting_support_score
            and (sit_support_ratio >= required_ratio or sit_ratio >= sitting_relaxed_ratio)
        ):
            confidence = clamp(
                0.46 + max(sit_ratio, sit_support_ratio) * 0.38 + min(1.5, median_sitting_score) * 0.06,
                0.0,
                0.92,
            )
            return "sitting", confidence, "supported seated posture"

        upright_ratio = sum(1 for feature in usable if feature["upright_like"]) / sample_count
        if self.action_report_standing and upright_ratio >= required_ratio:
            return "standing", clamp(0.45 + upright_ratio * 0.35, 0.0, 0.85), "upright posture"
        return "unknown", 0.25, "no reliable target action"

    def recognize_owner_action(self, owner_candidate):
        if owner_candidate is not None and "det" in owner_candidate:
            det = owner_candidate["det"]
            image_width = float(owner_candidate.get("image_width", 0.0))
            if image_width > 0:
                self.owner_track_center = float(det.center_x) / image_width

        if not self.action_recognition_enabled:
            self.last_owner_action_place = "unknown"
            self.last_owner_action_result = {}
            return "unknown", 0.0, "disabled"
        if not self.action_ready or self.qwen_action_core is None:
            self.last_owner_action_place = "unknown"
            self.last_owner_action_result = {}
            return "unknown", 0.0, "Qwen action recognizer unavailable"

        self.stop_base()
        yolo_paused = False
        try:
            if self.action_pause_yolo:
                self.set_yolo_paused(True)
                yolo_paused = True
                if self.action_yolo_pause_settle_seconds > 0.0:
                    rospy.sleep(self.action_yolo_pause_settle_seconds)

            started = time.time()
            samples = self.capture_qwen_action_frames()
            frames = [sample["image"] for sample in samples]
            full_frames = [sample["full_image"] for sample in samples]
            target_centers = [
                (sample.get("source_meta") or {}).get("pose_target_center_norm")
                for sample in samples
            ]
            capture_elapsed = time.time() - started

            try:
                pose_features, pose_action, pose_confidence, pose_reason = (
                    self.extract_owner_pose_features(
                        frames,
                        samples,
                        target_centers,
                    )
                )
            except Exception as exc:
                pose_features = []
                pose_action = "unknown"
                pose_confidence = 0.0
                pose_reason = "pose error: %s" % exc
                rospy.logwarn(
                    "Owner pose action analysis failed; continuing with Qwen static-action recognition: %s",
                    exc,
                )

            if pose_action in ("waving", "sudden_fall"):
                place = "floor" if pose_action == "sudden_fall" else "unknown"
                total_elapsed = time.time() - started
                self.last_owner_action_place = place
                self.last_owner_action_result = {
                    "action": pose_action,
                    "place": place,
                    "recognizer": "yolo_pose",
                    "pose_action": pose_action,
                    "pose_confidence": round(float(pose_confidence), 3),
                    "pose_reason": pose_reason,
                    "pose_feature_count": len(pose_features),
                    "capture_sec": round(capture_elapsed, 3),
                    "total_sec": round(total_elapsed, 3),
                    "frame_count": len(frames),
                }
                reason = (
                    "YOLO-Pose=%s(%.2f); %s; frames=%d; total=%.2fs"
                    % (
                        pose_action,
                        pose_confidence,
                        pose_reason,
                        len(frames),
                        total_elapsed,
                    )
                )
                rospy.loginfo(
                    "Owner dynamic action verdict: action=%s place=%s confidence=%.2f "
                    "features=%d frames=%d total=%.2fs",
                    pose_action,
                    place,
                    pose_confidence,
                    len(pose_features),
                    len(frames),
                    total_elapsed,
                )
                return pose_action, float(pose_confidence), reason

            warmup_ready = self.wait_for_qwen_action_warmup()
            if not warmup_ready and not self.qwen_warmup_done.is_set():
                rospy.loginfo(
                    "Waiting for the existing Qwen warm-up request to finish before static action recognition"
                )
                while not rospy.is_shutdown() and not self.qwen_warmup_done.wait(0.5):
                    pass
            if rospy.is_shutdown():
                return "unknown", 0.0, "ROS shutdown during Qwen warm-up"

            llm_frames = frames
            if len(frames) > self.action_llm_frame_count:
                indexes = [
                    int(
                        round(
                            index
                            * (len(frames) - 1)
                            / float(self.action_llm_frame_count - 1)
                        )
                )
                for index in range(self.action_llm_frame_count)
            ]
            llm_frames = [frames[index] for index in indexes]

            images_b64 = [self.encode_qwen_action_frame(frame) for frame in llm_frames]
            rospy.loginfo(
                "Owner action frames encoded: capture=%d, qwen=%d; starting Qwen inference",
                len(frames),
                len(llm_frames),
            )

            try:
                content, qwen_elapsed = self.call_qwen_action(images_b64)
            except RuntimeError as exc:
                error_text = str(exc).lower()
                if len(images_b64) <= 1 or "context" not in error_text:
                    raise
                rospy.logwarn(
                    "Qwen multi-frame request was rejected by context limits; retrying with the latest frame: %s",
                    exc,
                )
                content, qwen_elapsed = self.call_qwen_action([images_b64[-1]])

            qwen_action, place = self.qwen_action_core.parse_result(content)
            qwen_action_raw = qwen_action
            if qwen_action in ("waving", "sudden_fall"):
                rospy.logwarn(
                    "Ignoring Qwen dynamic action=%s because dynamic actions require YOLO-Pose evidence",
                    qwen_action,
                )
                qwen_action = "unknown"
                place = "unknown"
            rospy.loginfo(
                "Qwen owner static action parsed: action=%s raw_action=%s place=%s; starting point-cloud support analysis",
                qwen_action,
                qwen_action_raw,
                place,
            )

            rospy.loginfo("Starting owner action point-cloud support-surface analysis")
            ground_relation = self.analyze_qwen_ground_relation(
                full_frames[-1],
                pose_features,
            )
            rospy.loginfo(
                "Point-cloud support-surface analysis complete: status=%s reason=%s",
                ground_relation.get("status", "unknown"),
                ground_relation.get("reason", ""),
            )
            final_pose_is_horizontal = bool(
                pose_features and pose_features[-1].get("lying_like")
            )
            action, place, ground_fallen, elevated_lying = (
                self.qwen_action_core.merge_action_result(
                    qwen_action,
                    place,
                    "unknown",
                    ground_relation,
                    final_pose_is_horizontal,
                )
            )
            total_elapsed = time.time() - started
            self.last_owner_action_place = place
            self.last_owner_action_result = {
                "action": action,
                "place": place,
                "recognizer": "qwen_static",
                "raw": content,
                "qwen_action": qwen_action_raw,
                "qwen_action_used": qwen_action,
                "pose_action": "unknown",
                "pose_confidence": round(pose_confidence, 3),
                "pose_reason": pose_reason,
                "pose_feature_count": len(pose_features),
                "ground_relation": ground_relation,
                "ground_fallen": ground_fallen,
                "elevated_lying": elevated_lying,
                "capture_sec": round(capture_elapsed, 3),
                "qwen_sec": round(qwen_elapsed, 3),
                "total_sec": round(total_elapsed, 3),
                "frame_count": len(frames),
                "llm_frame_count": len(llm_frames),
            }
            confidence = max(
                0.35,
                float(pose_confidence) if action in ("waving", "sudden_fall") else 0.0,
            )
            if action == qwen_action and qwen_action not in ("unknown", "waving", "sudden_fall"):
                confidence = max(confidence, 0.70)
            reason = (
                "Qwen=%s/%s; pose_dynamic=%s(%.2f); ground=%s; frames=%d/%d; total=%.2fs"
                % (
                    qwen_action_raw,
                    place,
                    pose_action,
                    pose_confidence,
                    ground_relation.get("status", "unknown"),
                    len(frames),
                    len(llm_frames),
                    total_elapsed,
                )
            )
            rospy.loginfo(
                "Owner action verdict: action=%s place=%s qwen=%s used=%s pose_dynamic=%s(%.2f) "
                "ground=%s frames=%d/%d total=%.2fs",
                action,
                place,
                qwen_action_raw,
                qwen_action,
                pose_action,
                pose_confidence,
                ground_relation.get("status", "unknown"),
                len(frames),
                len(llm_frames),
                total_elapsed,
            )
            return action, confidence, reason
        except Exception as exc:
            self.last_owner_action_place = "unknown"
            self.last_owner_action_result = {"error": str(exc)}
            rospy.logwarn("Owner Qwen action recognition failed: %s", exc)
            return "unknown", 0.0, "Qwen action error: %s" % exc
        finally:
            if yolo_paused:
                self.set_yolo_paused(False)

    @staticmethod
    def action_to_speech(label, place="unknown"):
        messages = {
            "falling": "识别到主人摔倒。",
            "lying_ground": "识别到主人摔倒。",
            "sudden_fall": "主人突然摔倒在地上。",
            "fallen": "主人已经摔倒在地上。",
            "standing": "主人当前站立，没有检测到指定异常动作。",
            "unknown": "我已识别到主人，但动作不确定。",
        }
        if label == "waving":
            if place in ("chair", "sofa", "bed"):
                return "主人正在挥手，人在%s。" % {
                    "chair": "椅子上",
                    "sofa": "沙发上",
                    "bed": "床上",
                }[place]
            return "主人正在挥手。"
        if label == "lying":
            if place in ("chair", "sofa", "bed"):
                return "主人正躺在%s。" % {
                    "chair": "椅子上",
                    "sofa": "沙发上",
                    "bed": "床上",
                }[place]
            return "主人正在躺着。"
        if label == "sitting":
            if place in ("chair", "sofa", "bed"):
                return "主人正坐在%s。" % {
                    "chair": "椅子上",
                    "sofa": "沙发上",
                    "bed": "床上",
                }[place]
            return "主人正在坐着。"
        return messages.get(label, messages["unknown"])

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "image conversion failed: %s", exc)
            return
        with self.lock:
            self.latest_image = image
            self.latest_image_time = time.time()

    def detections_callback(self, msg):
        with self.lock:
            self.latest_detections = list(msg.detections)
            self.latest_detections_time = time.time()

    def pointcloud_callback(self, msg):
        with self.lock:
            self.latest_pointcloud = msg
            self.latest_pointcloud_time = time.time()

    def scan_callback(self, msg):
        with self.lock:
            self.latest_scan = msg
            self.latest_scan_time = time.time()

    def odom_callback(self, msg):
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        position = msg.pose.pose.position
        linear = msg.twist.twist.linear
        with self.lock:
            self.latest_yaw = yaw
            self.latest_odom_xy = (float(position.x), float(position.y))
            self.latest_odom_linear_speed = math.hypot(float(linear.x), float(linear.y))
            self.latest_odom_time = time.time()

    def asr_callback(self, msg):
        text = str(msg.data).strip()
        if not text:
            return
        now = time.time()
        with self.lock:
            self.asr_sequence += 1
            self.latest_asr_text = text
            self.latest_asr_time = now
            self.asr_history.append({"sequence": self.asr_sequence, "time": now, "text": text})
            if len(self.asr_history) > 100:
                self.asr_history = self.asr_history[-100:]
        rospy.loginfo("ASR text received on %s: %s", self.asr_topic, text)

    def collect_asr_since(self, sequence_after):
        with self.lock:
            entries = [entry for entry in self.asr_history if entry["sequence"] > sequence_after]
        if not entries:
            return "", sequence_after

        parts = []
        for entry in entries:
            text = str(entry.get("text", "")).strip()
            if text:
                parts.append(text)
        transcript = " ".join(parts).strip()
        return transcript, int(entries[-1]["sequence"])

    @staticmethod
    def write_pcm_wav(path, raw_audio, sample_rate, channels):
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(int(channels))
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(raw_audio)

    def ensure_electrical_switch_ready_ding_wav(self):
        wav_path = self.electrical_switch_ready_ding_wav
        sample_rate = 16000
        duration = max(0.03, float(self.electrical_switch_ready_ding_duration))
        frequency = max(100.0, float(self.electrical_switch_ready_ding_frequency))
        volume = clamp(float(self.electrical_switch_ready_ding_volume), 0.0, 1.0)
        frame_count = max(1, int(sample_rate * duration))
        attack_frames = max(1, int(sample_rate * 0.01))
        release_frames = max(1, int(sample_rate * 0.04))
        amplitude = int(32767 * volume)
        frames = bytearray()

        for index in range(frame_count):
            attack = min(1.0, float(index + 1) / attack_frames)
            release = min(1.0, float(frame_count - index) / release_frames)
            envelope = min(attack, release)
            sample = int(amplitude * envelope * math.sin(2.0 * math.pi * frequency * index / sample_rate))
            frames.extend(struct.pack("<h", sample))

        try:
            out_dir = os.path.dirname(wav_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            self.write_pcm_wav(wav_path, bytes(frames), sample_rate, 1)
            return wav_path
        except Exception as exc:
            rospy.logwarn("Failed to create electrical switch ready ding wav: %s", exc)
            return ""

    def play_electrical_switch_ready_ding(self):
        if not self.electrical_switch_ready_ding_enabled:
            return
        player = self.electrical_switch_ready_ding_player
        if not player:
            return

        wav_path = self.ensure_electrical_switch_ready_ding_wav()
        if not wav_path:
            return

        cmd = [player, "-q"]
        if self.electrical_switch_ready_ding_speaker_device and os.path.basename(player) == "aplay":
            cmd.extend(["-D", self.electrical_switch_ready_ding_speaker_device])
        cmd.append(wav_path)
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=max(1.0, float(self.electrical_switch_ready_ding_duration) + 2.0),
            )
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="replace").strip()
                rospy.logwarn("Electrical switch ready ding playback failed: %s", stderr or proc.returncode)
        except (OSError, subprocess.TimeoutExpired) as exc:
            rospy.logwarn("Unable to play electrical switch ready ding: %s", exc)

    def start_electrical_switch_preload(self):
        if not self.electrical_switch_preload_enabled or self.electrical_switch_preload_started:
            return
        self.electrical_switch_preload_started = True
        self.electrical_switch_preload_thread = threading.Thread(
            target=self.preload_electrical_switch_voice_stack,
            name="electrical_switch_voice_preload",
        )
        self.electrical_switch_preload_thread.daemon = True
        self.electrical_switch_preload_thread.start()

    def preload_electrical_switch_voice_stack(self):
        rospy.loginfo("Preloading electrical switch voice stack in background")
        try:
            if self.electrical_switch_instruction_source in (
                "direct",
                "direct_asr",
                "mic",
                "microphone",
            ):
                self.ensure_electrical_switch_asr_model()
            if self.electrical_switch_ollama_warmup_enabled:
                for attempt in range(self.electrical_switch_ollama_warmup_retries):
                    if rospy.is_shutdown():
                        break
                    if self.warmup_electrical_switch_ollama():
                        break
                    if attempt + 1 < self.electrical_switch_ollama_warmup_retries:
                        rospy.sleep(max(0.0, float(self.electrical_switch_ollama_warmup_retry_delay)))
        finally:
            self.electrical_switch_preload_done.set()
            rospy.loginfo("Electrical switch voice stack preload finished")

    def wait_for_electrical_switch_preload(self):
        if not self.electrical_switch_preload_enabled:
            return
        self.start_electrical_switch_preload()
        if self.electrical_switch_preload_done.is_set() or not self.electrical_switch_preload_wait_before_prompt:
            return

        timeout = max(0.0, float(self.electrical_switch_preload_wait_timeout))
        rospy.loginfo("Waiting up to %.1fs for electrical switch voice preload", timeout)
        if timeout <= 0.0:
            self.electrical_switch_preload_done.wait()
        elif not self.electrical_switch_preload_done.wait(timeout):
            rospy.logwarn("Electrical switch voice preload is still running; continuing before it finishes")

    def electrical_switch_ollama_tags_url(self):
        url = str(self.electrical_switch_ollama_url).strip()
        if re.search(r"/api/(chat|generate)/?$", url):
            return re.sub(r"/api/(chat|generate)/?$", "/api/tags", url)
        return url.rstrip("/") + "/api/tags"

    def select_available_electrical_switch_ollama_model(self):
        if not self.electrical_switch_ollama_autoselect_model:
            return True

        request = urllib.request.Request(self.electrical_switch_ollama_tags_url(), method="GET")
        try:
            with urllib.request.urlopen(
                request,
                timeout=max(0.5, float(self.electrical_switch_ollama_tags_timeout)),
            ) as response:
                raw = response.read().decode("utf-8")
            result = json.loads(raw)
        except Exception as exc:
            rospy.logwarn("Unable to list local Ollama models; trying configured model directly: %s", exc)
            return True

        names = []
        model_items = result.get("models", []) if isinstance(result, dict) else []
        for item in model_items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if name:
                names.append(name)
        if not names:
            rospy.logwarn("Ollama returned no local models; skipping LLM for switch command classification")
            self.electrical_switch_ollama_available = False
            self.electrical_switch_ollama_last_failure = time.time()
            return False

        by_lower_name = {name.lower(): name for name in names}
        candidates = [self.electrical_switch_ollama_model] + list(self.electrical_switch_ollama_model_fallbacks)
        for candidate in candidates:
            model_name = by_lower_name.get(str(candidate).strip().lower())
            if not model_name:
                continue
            if model_name != self.electrical_switch_ollama_model:
                rospy.logwarn(
                    "Configured Ollama model %s is not local; using available fallback %s",
                    self.electrical_switch_ollama_model,
                    model_name,
                )
                self.electrical_switch_ollama_model = model_name
            return True

        rospy.logwarn(
            "No configured switch-command Ollama model is installed locally; candidates=%s local=%s",
            ", ".join(str(item) for item in candidates if str(item).strip()),
            ", ".join(names[:8]),
        )
        self.electrical_switch_ollama_available = False
        self.electrical_switch_ollama_last_failure = time.time()
        return False

    def warmup_electrical_switch_ollama(self):
        if not self.electrical_switch_ollama_enabled:
            self.electrical_switch_ollama_available = False
            return False
        with self.electrical_switch_ollama_lock:
            if self.electrical_switch_ollama_available is True:
                return True
            if not self.select_available_electrical_switch_ollama_model():
                return False

            prompt = (
                "/no_think\n"
                "只输出一行 JSON：{\"action\":\"unknown\"}。这是启动预热，不是用户指令。"
            )
            payload = {
                "model": self.electrical_switch_ollama_model,
                "stream": False,
                "think": False,
                "format": "json",
                "keep_alive": self.electrical_switch_ollama_keep_alive,
                "messages": [{"role": "user", "content": prompt}],
                "options": {
                    "temperature": 0,
                    "top_p": 0.7,
                    "num_predict": max(8, self.electrical_switch_ollama_max_tokens),
                    "num_ctx": 512,
                    "num_gpu": max(0, self.electrical_switch_ollama_num_gpu),
                },
            }
            request = urllib.request.Request(
                self.electrical_switch_ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                start = time.time()
                with self.ollama_request_lock:
                    with urllib.request.urlopen(
                        request,
                        timeout=max(1.0, float(self.electrical_switch_ollama_warmup_timeout)),
                    ) as response:
                        response.read()
                self.electrical_switch_ollama_available = True
                rospy.loginfo(
                    "Electrical switch Ollama model warmed in %.2fs: %s",
                    time.time() - start,
                    self.electrical_switch_ollama_model,
                )
                return True
            except Exception as exc:
                self.electrical_switch_ollama_available = False
                self.electrical_switch_ollama_last_failure = time.time()
                rospy.logwarn(
                    "Electrical switch Ollama warmup failed; keyword fallback will be used first: %s",
                    exc,
                )
                return False

    def should_try_electrical_switch_ollama(self):
        if not self.electrical_switch_ollama_enabled:
            return False
        if (
            self.electrical_switch_ollama_warmup_enabled
            and self.electrical_switch_preload_started
            and not self.electrical_switch_preload_done.is_set()
            and self.electrical_switch_ollama_available is None
        ):
            return False
        if self.electrical_switch_ollama_available is not False:
            return True
        cooldown = max(0.0, float(self.electrical_switch_ollama_failure_cooldown))
        return time.time() - self.electrical_switch_ollama_last_failure >= cooldown

    def begin_electrical_switch_voice_phase(self):
        if not self.electrical_switch_pause_yolo_during_voice:
            return
        self.set_yolo_paused(True)
        settle_seconds = max(0.0, float(self.electrical_switch_pause_yolo_settle_seconds))
        if settle_seconds > 0.0:
            rospy.sleep(settle_seconds)

    def end_electrical_switch_voice_phase(self):
        if self.electrical_switch_pause_yolo_during_voice:
            self.set_yolo_paused(False)

    def should_print_electrical_switch_script_line(self, line):
        if not self.electrical_switch_script_filter_output:
            return True
        try:
            return re.search(self.electrical_switch_script_output_pattern, line) is not None
        except re.error:
            return True

    def ensure_electrical_switch_asr_model(self):
        if self.electrical_switch_asr_ready and self.electrical_switch_asr_model is not None:
            return True

        with self.electrical_switch_asr_lock:
            if self.electrical_switch_asr_ready and self.electrical_switch_asr_model is not None:
                return True

            if self.electrical_switch_asr_hf_endpoint:
                os.environ.setdefault("HF_ENDPOINT", self.electrical_switch_asr_hf_endpoint)
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                rospy.logerr("faster-whisper is not available for direct switch ASR: %s", exc)
                return False

            model_path = self.electrical_switch_asr_model_path
            if os.path.sep in model_path and not os.path.exists(model_path):
                rospy.logerr("Direct switch ASR model path does not exist: %s", model_path)
                return False

            try:
                rospy.loginfo(
                    "Loading direct switch ASR model: path=%s device=%s compute=%s",
                    model_path,
                    self.electrical_switch_asr_device,
                    self.electrical_switch_asr_compute_type,
                )
                start = time.time()
                self.electrical_switch_asr_model = WhisperModel(
                    model_path,
                    device=self.electrical_switch_asr_device,
                    compute_type=self.electrical_switch_asr_compute_type,
                )
                self.electrical_switch_asr_ready = True
                rospy.loginfo("Direct switch ASR model loaded in %.2fs", time.time() - start)
                return True
            except Exception as exc:
                self.electrical_switch_asr_model = None
                self.electrical_switch_asr_ready = False
                rospy.logerr("Failed to load direct switch ASR model: %s", exc)
                return False

    def amplify_electrical_switch_audio(self, raw_audio, bytes_per_sample, window_index):
        if bytes_per_sample != 2 or not raw_audio:
            return raw_audio, 1.0, None, None

        samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return raw_audio, 1.0, None, None

        input_peak = float(np.max(np.abs(samples)))
        gain = max(1.0, float(self.electrical_switch_asr_input_gain))
        target_peak = float(self.electrical_switch_asr_auto_gain_target_peak) * 32767.0
        if target_peak > 0.0 and input_peak > 0.0:
            gain = max(gain, target_peak / input_peak)
        gain = clamp(gain, 1.0, float(self.electrical_switch_asr_max_gain))
        if gain <= 1.0001:
            return raw_audio, 1.0, int(input_peak), int(input_peak)

        amplified = np.clip(samples * gain, -32768.0, 32767.0).astype(np.int16)
        output_peak = int(np.max(np.abs(amplified.astype(np.float32))))
        rospy.loginfo(
            "Direct switch ASR window %d input gain: gain=%.2fx peak=%d->%d target_peak=%.0f",
            window_index,
            gain,
            int(input_peak),
            output_peak,
            target_peak,
        )
        return amplified.tobytes(), gain, int(input_peak), output_peak

    def record_electrical_switch_audio_window(self, window_seconds, window_index):
        sample_rate = max(8000, int(self.electrical_switch_asr_sample_rate))
        channels = max(1, int(self.electrical_switch_asr_channels))
        bytes_per_sample = 2
        target_seconds = max(0.5, float(window_seconds))
        arecord_seconds = max(1, int(math.ceil(target_seconds)))
        target_bytes = int(target_seconds * sample_rate * channels * bytes_per_sample)
        cmd = [
            "arecord",
            "-q",
            "-D",
            str(self.electrical_switch_asr_capture_device),
            "-d",
            str(arecord_seconds),
            "-f",
            "S16_LE",
            "-r",
            str(sample_rate),
            "-c",
            str(channels),
            "-t",
            "raw",
        ]

        rospy.loginfo(
            "Direct switch ASR window %d: recording %.1fs from ALSA device %s",
            window_index,
            target_seconds,
            self.electrical_switch_asr_capture_device,
        )
        self.play_electrical_switch_ready_ding()
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=arecord_seconds + 4,
            )
        except subprocess.TimeoutExpired as exc:
            rospy.logerr("Direct switch ASR arecord timed out: %s", exc)
            return None
        except OSError as exc:
            rospy.logerr("Direct switch ASR failed to start arecord: %s", exc)
            return None

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            rospy.logerr("Direct switch ASR arecord failed: %s", stderr or "unknown error")
            return None

        raw_audio = proc.stdout[:target_bytes]
        if not raw_audio:
            rospy.logerr("Direct switch ASR arecord returned no audio")
            return None

        input_rms = audioop.rms(raw_audio, bytes_per_sample)
        input_peak = audioop.max(raw_audio, bytes_per_sample)
        raw_audio, applied_gain, _pre_gain_peak, _post_gain_peak = self.amplify_electrical_switch_audio(
            raw_audio,
            bytes_per_sample,
            window_index,
        )
        rms = audioop.rms(raw_audio, bytes_per_sample)
        peak = audioop.max(raw_audio, bytes_per_sample)
        rospy.loginfo(
            "Direct switch ASR window %d audio stats: raw_rms=%d raw_peak=%d gain=%.2fx rms=%d peak=%d",
            window_index,
            input_rms,
            input_peak,
            applied_gain,
            rms,
            peak,
        )

        if (
            self.electrical_switch_asr_transcribe_from_memory
            and not self.electrical_switch_asr_keep_wav
            and sample_rate == 16000
            and channels == 1
        ):
            audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            return {"audio": audio, "wav_path": "", "rms": rms, "peak": peak}

        try:
            os.makedirs(self.electrical_switch_asr_wav_dir, exist_ok=True)
            wav_path = os.path.join(
                self.electrical_switch_asr_wav_dir,
                "task1_switch_instruction_%03d.wav" % window_index,
            )
            self.write_pcm_wav(wav_path, raw_audio, sample_rate, channels)
            return wav_path
        except Exception as exc:
            rospy.logerr("Failed to write direct switch ASR wav: %s", exc)
            return None

    def transcribe_electrical_switch_audio(self, audio_input, window_index):
        if not self.ensure_electrical_switch_asr_model():
            return None

        wav_path = audio_input
        audio_source = audio_input
        if isinstance(audio_input, dict):
            wav_path = audio_input.get("wav_path", "")
            audio_source = audio_input.get("audio")

        try:
            start = time.time()
            segments, _ = self.electrical_switch_asr_model.transcribe(
                audio_source,
                language=self.electrical_switch_asr_language,
                vad_filter=self.electrical_switch_asr_vad_filter,
                beam_size=self.electrical_switch_asr_beam_size,
                condition_on_previous_text=False,
                no_speech_threshold=self.electrical_switch_asr_no_speech_threshold,
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            rospy.loginfo(
                "Direct switch ASR window %d transcript: %s (%.2fs)",
                window_index,
                transcript or "<empty>",
                time.time() - start,
            )
            return transcript
        except Exception as exc:
            rospy.logerr("Direct switch ASR transcription failed: %s", exc)
            return None
        finally:
            if isinstance(wav_path, str) and wav_path and not self.electrical_switch_asr_keep_wav:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    def collect_direct_electrical_switch_transcript(self):
        if not self.ensure_electrical_switch_asr_model():
            return ""

        window_seconds = max(0.5, self.electrical_switch_instruction_window_seconds)
        max_empty_windows = max(0, self.electrical_switch_instruction_max_empty_windows)
        total_timeout = max(0.0, self.electrical_switch_instruction_timeout)
        deadline = time.time() + total_timeout if total_timeout > 0.0 else None
        empty_windows = 0
        window_index = 1

        while not rospy.is_shutdown():
            if deadline is not None and time.time() >= deadline:
                rospy.logwarn("Direct switch ASR wait timed out after %.1fs", total_timeout)
                break

            wav_path = self.record_electrical_switch_audio_window(window_seconds, window_index)
            if wav_path is None:
                break

            transcript = self.transcribe_electrical_switch_audio(wav_path, window_index)
            if transcript is None:
                break
            if transcript:
                return transcript

            empty_windows += 1
            rospy.logwarn(
                "Direct switch ASR window %d had no recognized speech; continuing with next %.1fs window",
                window_index,
                window_seconds,
            )
            if max_empty_windows > 0 and empty_windows >= max_empty_windows:
                rospy.logwarn("Direct switch ASR stopped after %d empty windows", empty_windows)
                break
            window_index += 1

        return ""

    def collect_ros_topic_electrical_switch_transcript(self):
        with self.lock:
            sequence_after_prompt = self.asr_sequence
            self.latest_asr_text = ""
            self.latest_asr_time = None

        window_seconds = max(0.5, self.electrical_switch_instruction_window_seconds)
        settle_seconds = max(0.0, self.electrical_switch_instruction_asr_settle_seconds)
        max_empty_windows = max(0, self.electrical_switch_instruction_max_empty_windows)
        total_timeout = max(0.0, self.electrical_switch_instruction_timeout)
        deadline = time.time() + total_timeout if total_timeout > 0.0 else None
        rate = rospy.Rate(10)
        empty_windows = 0
        window_index = 1

        while not rospy.is_shutdown():
            if deadline is not None and time.time() >= deadline:
                rospy.logwarn("Electrical switch topic-ASR wait timed out after %.1fs", total_timeout)
                break

            rospy.loginfo(
                "Electrical switch topic-ASR window %d: listening for %.1fs on %s",
                window_index,
                window_seconds,
                self.asr_topic,
            )
            self.play_electrical_switch_ready_ding()
            window_end = time.time() + window_seconds
            while not rospy.is_shutdown() and time.time() < window_end:
                if deadline is not None and time.time() >= deadline:
                    break
                rate.sleep()

            if settle_seconds > 0.0:
                settle_end = time.time() + settle_seconds
                while not rospy.is_shutdown() and time.time() < settle_end:
                    rate.sleep()

            transcript, sequence_after_prompt = self.collect_asr_since(sequence_after_prompt)
            if transcript:
                rospy.loginfo("Electrical switch topic-ASR window %d transcript: %s", window_index, transcript)
                return transcript

            empty_windows += 1
            rospy.logwarn(
                "Electrical switch topic-ASR window %d had no ASR text; continuing with next %.1fs window",
                window_index,
                window_seconds,
            )
            if max_empty_windows > 0 and empty_windows >= max_empty_windows:
                rospy.logwarn("Electrical switch topic-ASR stopped after %d empty windows", empty_windows)
                break
            window_index += 1

        return ""

    def build_local_switch_command_test_cmd(self):
        script_path = os.path.expanduser(self.electrical_switch_script_path)
        if not os.path.isabs(script_path):
            script_path = os.path.join(self.task_package_dir, script_path)
        script_path = os.path.abspath(script_path)

        window_seconds = max(0.5, self.electrical_switch_instruction_window_seconds)
        cmd = [self.electrical_switch_script_python, "-u", script_path]
        if self.electrical_switch_script_until_result:
            cmd.append("--until-result")
        cmd.extend(["--count", "1"])

        def add_arg(flag, value):
            value = "" if value is None else str(value)
            if value:
                cmd.extend([flag, value])

        add_arg("--mic-device", self.electrical_switch_asr_capture_device)
        add_arg("--sample-rate", self.electrical_switch_asr_sample_rate)
        add_arg("--seconds", window_seconds)
        add_arg("--energy-threshold", self.electrical_switch_asr_energy_threshold)
        add_arg("--model-size", self.electrical_switch_asr_model_path)
        add_arg("--hf-endpoint", self.electrical_switch_asr_hf_endpoint)
        add_arg("--asr-device", self.electrical_switch_asr_device)
        add_arg("--compute-type", self.electrical_switch_asr_compute_type)
        add_arg("--language", self.electrical_switch_asr_language)
        add_arg("--beam-size", self.electrical_switch_asr_beam_size)
        add_arg("--no-speech-threshold", self.electrical_switch_asr_no_speech_threshold)
        add_arg("--llm-url", self.electrical_switch_ollama_url)
        add_arg("--llm-model", self.electrical_switch_ollama_model)
        add_arg("--llm-timeout", self.electrical_switch_ollama_timeout)
        add_arg("--llm-max-tokens", self.electrical_switch_ollama_max_tokens)
        add_arg("--llm-keep-alive", self.electrical_switch_ollama_keep_alive)
        if self.electrical_switch_asr_vad_filter:
            cmd.append("--vad-filter")
        else:
            cmd.append("--no-vad")
        if self.electrical_switch_asr_keep_wav:
            cmd.append("--keep-wav")
        return script_path, cmd

    def run_local_switch_command_test(self):
        script_path, cmd = self.build_local_switch_command_test_cmd()
        if not os.path.exists(script_path):
            rospy.logerr("Local switch command script does not exist: %s", script_path)
            return "unknown"

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env["LOCAL_SWITCH_READY_DING"] = "1" if self.electrical_switch_ready_ding_enabled else "0"
        env["LOCAL_SWITCH_READY_DING_WAV"] = self.electrical_switch_ready_ding_wav
        env["LOCAL_SWITCH_READY_DING_FREQUENCY"] = str(self.electrical_switch_ready_ding_frequency)
        env["LOCAL_SWITCH_READY_DING_SECONDS"] = str(self.electrical_switch_ready_ding_duration)
        env["LOCAL_SWITCH_READY_DING_VOLUME"] = str(self.electrical_switch_ready_ding_volume)
        env["LOCAL_SWITCH_READY_DING_PLAYER"] = self.electrical_switch_ready_ding_player
        env["LOCAL_SWITCH_READY_DING_SPEAKER_DEVICE"] = self.electrical_switch_ready_ding_speaker_device
        rospy.loginfo("Starting local switch command test subprocess: %s", " ".join(cmd))

        action = None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=self.task_package_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=0,
            )
        except OSError as exc:
            rospy.logerr("Failed to start local switch command test subprocess: %s", exc)
            return "unknown"

        try:
            for raw_line in iter(proc.stdout.readline, b""):
                line = raw_line.decode("utf-8", errors="replace")
                if self.should_print_electrical_switch_script_line(line):
                    try:
                        sys.stdout.buffer.write(raw_line)
                        sys.stdout.buffer.flush()
                    except Exception:
                        sys.stdout.write(line)
                        sys.stdout.flush()

                match = re.search(r"判断结果:\s*(on|off|unknown)\b", line)
                if match:
                    action = match.group(1)
            returncode = proc.wait()
        except Exception as exc:
            rospy.logerr("Error while reading local switch command test output: %s", exc)
            try:
                proc.terminate()
            except OSError:
                pass
            return "unknown"

        if returncode != 0:
            rospy.logwarn("local_switch_command_test.py exited with code %s", returncode)
        if action not in ("on", "off", "unknown"):
            action = "unknown"
        return action

    def publish_electrical_switch_state(self):
        self.electrical_switch_state_pub.publish(String(data=self.electrical_switch_state))
        rospy.loginfo(
            "Electrical switch state: %s (topic=%s)",
            self.electrical_switch_state,
            self.electrical_switch_state_topic,
        )

    @staticmethod
    def parse_electrical_switch_action(content):
        """Normalize the short JSON response returned by the local LLM."""
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            match = re.search(r"\{.*\}", str(content), re.DOTALL)
            if not match:
                return "unknown"
            try:
                parsed = json.loads(match.group(0))
            except (TypeError, ValueError):
                return "unknown"

        if not isinstance(parsed, dict):
            return "unknown"

        action_map = {
            "on": "on",
            "开": "on",
            "开启": "on",
            "打开": "on",
            "open": "on",
            "1": "on",
            "true": "on",
            "接通": "on",
            "上电": "on",
            "off": "off",
            "关": "off",
            "关闭": "off",
            "关掉": "off",
            "close": "off",
            "0": "off",
            "false": "off",
            "断开": "off",
            "断电": "off",
        }
        values = [parsed.get("action")]
        values.extend(value for value in parsed.values() if not isinstance(value, (dict, list)))
        for value in values:
            action = action_map.get(str(value).strip().lower())
            if action:
                return action
        return "unknown"

    @staticmethod
    def electrical_switch_keyword_fallback(transcript):
        """Conservative fallback used when Ollama is unavailable or malformed."""
        text = str(transcript).strip()
        negation = re.search(r"(不要|不用|别|不想|无需|不需要|禁止|不能|别把|不用把)", text)
        if re.search(r"(关闭|关掉|关上|关了|关一下|断开|断电|停掉|停止|拔掉|熄灭|灭掉)", text):
            return "off"
        if re.search(r"(打开|开启|开一下|开开|开机|开起来|接通|上电|启动|点亮|亮起)", text) and not negation:
            return "on"

        text_without_switch_noun = re.sub(r"开关", "", text)
        if re.search(r"(关|灭)", text_without_switch_noun) and not re.search(r"(开|接通|上电|亮)", text_without_switch_noun):
            return "off"
        if (
            re.search(r"(开|接通|上电|启动|亮)", text_without_switch_noun)
            and not negation
            and not re.search(r"(关|断开|断电|灭)", text_without_switch_noun)
        ):
            return "on"
        return "unknown"

    def classify_electrical_switch_instruction(self, transcript):
        """Use local Ollama first, with the reference script's keyword fallback."""
        keyword_action = self.electrical_switch_keyword_fallback(transcript)
        if self.electrical_switch_fast_keyword_first and keyword_action in ("on", "off"):
            rospy.loginfo("Electrical switch instruction classified by fast keyword path: %s", keyword_action)
            return keyword_action
        if not self.should_try_electrical_switch_ollama():
            rospy.logwarn("Skipping electrical switch Ollama classification due to recent warmup/request failure")
            return keyword_action

        prompt = (
            "/no_think\n"
            "判断说话内容是否要开启或关闭电气开关。只输出一行 JSON，不要任何解释。\n"
            '{"action":"on"} 表示开启；{"action":"off"} 表示关闭；'
            '{"action":"unknown"} 表示与开关无关或无法判断。\n'
            '示例：“把灯打开” -> {"action":"on"}；“关掉电源” -> {"action":"off"}；'
            '“今天天气怎么样” -> {"action":"unknown"}\n'
            "说话内容：%s" % transcript
        )
        payload = {
            "model": self.electrical_switch_ollama_model,
            "stream": False,
            "think": False,
            "format": "json",
            "keep_alive": self.electrical_switch_ollama_keep_alive,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": 0,
                "top_p": 0.7,
                "num_predict": self.electrical_switch_ollama_max_tokens,
                "num_ctx": 1024,
                "num_gpu": max(0, self.electrical_switch_ollama_num_gpu),
            },
        }
        request = urllib.request.Request(
            self.electrical_switch_ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.ollama_request_lock:
                with urllib.request.urlopen(
                    request, timeout=max(1.0, self.electrical_switch_ollama_timeout)
                ) as response:
                    raw = response.read().decode("utf-8")
            result = json.loads(raw)
            content = result.get("message", {}).get("content", "").strip()
            action = self.parse_electrical_switch_action(content)
            if action == "unknown":
                rospy.logwarn("Ollama returned an unknown electrical switch action: %s", content)
                if keyword_action in ("on", "off"):
                    rospy.loginfo("Using keyword fallback after unknown Ollama result: %s", keyword_action)
                    return keyword_action
            else:
                rospy.loginfo("Electrical switch instruction classified by Ollama: %s", action)
            self.electrical_switch_ollama_available = True
            return action
        except (OSError, ValueError, TypeError, urllib.error.URLError) as exc:
            self.electrical_switch_ollama_available = False
            self.electrical_switch_ollama_last_failure = time.time()
            rospy.logwarn("Electrical switch Ollama classification failed: %s", exc)
            rospy.loginfo("Using keyword fallback for electrical switch instruction: %s", keyword_action)
            return keyword_action

    def wait_for_electrical_switch_instruction(self):
        """Ask once, then listen until an on/off switch command is recognized."""
        if not self.electrical_switch_instruction_enabled:
            rospy.loginfo("Electrical switch voice interaction is disabled")
            return "unknown"

        source = self.electrical_switch_instruction_source
        self.wait_for_electrical_switch_preload()
        self.begin_electrical_switch_voice_phase()
        try:
            if source in ("local_script", "script", "local_switch", "local_switch_command_test"):
                self.say(self.electrical_switch_prompt, hold=self.electrical_switch_prompt_hold)
                action = self.run_local_switch_command_test()
                self.electrical_switch_state = action
                self.publish_electrical_switch_state()
                return action

            action = "unknown"
            if source in ("direct", "direct_asr", "mic", "microphone"):
                if not self.ensure_electrical_switch_asr_model():
                    action = "unknown"
                else:
                    self.say(self.electrical_switch_prompt, hold=self.electrical_switch_prompt_hold)
                    while not rospy.is_shutdown():
                        transcript = self.collect_direct_electrical_switch_transcript()
                        if not transcript:
                            rospy.logwarn(
                                "No electrical switch instruction text received by source=%s",
                                source,
                            )
                            break
                        rospy.loginfo("Electrical switch instruction transcript for LLM: %s", transcript)
                        action = self.classify_electrical_switch_instruction(transcript)
                        if action in ("on", "off"):
                            break
                        rospy.logwarn("No switch command recognized from transcript; listening again")
            elif source in ("topic", "ros", "ros_topic", "voice_topic"):
                with self.lock:
                    self.latest_asr_text = ""
                    self.latest_asr_time = None
                self.say(self.electrical_switch_prompt, hold=self.electrical_switch_prompt_hold)
                while not rospy.is_shutdown():
                    transcript = self.collect_ros_topic_electrical_switch_transcript()
                    if not transcript:
                        rospy.logwarn(
                            "No electrical switch instruction text received by source=%s",
                            source,
                        )
                        break
                    rospy.loginfo("Electrical switch instruction transcript for LLM: %s", transcript)
                    action = self.classify_electrical_switch_instruction(transcript)
                    if action in ("on", "off"):
                        break
                    rospy.logwarn("No switch command recognized from transcript; listening again")
            else:
                rospy.logwarn(
                    "Unknown electrical_switch_instruction_source=%s; falling back to direct_asr",
                    source,
                )
                if not self.ensure_electrical_switch_asr_model():
                    action = "unknown"
                else:
                    self.say(self.electrical_switch_prompt, hold=self.electrical_switch_prompt_hold)
                    while not rospy.is_shutdown():
                        transcript = self.collect_direct_electrical_switch_transcript()
                        if not transcript:
                            rospy.logwarn(
                                "No electrical switch instruction text received by source=%s",
                                source,
                            )
                            break
                        rospy.loginfo("Electrical switch instruction transcript for LLM: %s", transcript)
                        action = self.classify_electrical_switch_instruction(transcript)
                        if action in ("on", "off"):
                            break
                        rospy.logwarn("No switch command recognized from transcript; listening again")

            if action not in ("on", "off"):
                action = "unknown"

            self.electrical_switch_state = action
            self.publish_electrical_switch_state()
            reply = {
                "on": self.electrical_switch_reply_on,
                "off": self.electrical_switch_reply_off,
                "unknown": self.electrical_switch_reply_unknown,
            }.get(action, self.electrical_switch_reply_unknown)
            self.say(reply)
            return action
        finally:
            self.end_electrical_switch_voice_phase()

    def get_latest_yaw(self):
        with self.lock:
            return self.latest_yaw

    def get_latest_odom_xy(self):
        with self.lock:
            return self.latest_odom_xy

    def get_latest_odom_linear_speed(self, max_age=None):
        with self.lock:
            if self.latest_odom_linear_speed is None or self.latest_odom_time is None:
                return None
            if max_age is not None and time.time() - self.latest_odom_time > max_age:
                return None
            return self.latest_odom_linear_speed

    def update_approach_slow_finish_cycles(self, remaining_distance, commanded_linear_speed, previous_cycles):
        if not self.approach_slow_finish_enabled:
            return 0, None, "disabled"

        remaining_distance = max(0.0, float(remaining_distance))
        if remaining_distance > self.approach_slow_finish_tolerance:
            return 0, None, "not_near"

        commanded_speed = abs(float(commanded_linear_speed))
        actual_speed = self.get_latest_odom_linear_speed(max_age=self.approach_odom_speed_max_age)
        command_slow = commanded_speed <= self.approach_slow_finish_linear_speed
        odom_slow = actual_speed is not None and actual_speed <= self.approach_slow_finish_linear_speed
        if command_slow or odom_slow:
            if command_slow and odom_slow:
                reason = "command_and_odom_slow"
            elif command_slow:
                reason = "command_slow"
            else:
                reason = "odom_slow"
            return previous_cycles + 1, actual_speed, reason
        return 0, actual_speed, "moving"

    def wait_for_odom_xy(self, timeout=1.0):
        deadline = time.time() + max(0.0, timeout)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() < deadline:
            odom_xy = self.get_latest_odom_xy()
            if odom_xy is not None:
                return odom_xy
            rate.sleep()
        return self.get_latest_odom_xy()

    @staticmethod
    def xy_distance(start_xy, end_xy):
        if start_xy is None or end_xy is None:
            return None
        return math.hypot(float(end_xy[0]) - float(start_xy[0]), float(end_xy[1]) - float(start_xy[1]))

    def stop_base(self):
        self.cmd_pub.publish(Twist())

    def front_scan_distance(self):
        with self.lock:
            scan = self.latest_scan
            scan_time = self.latest_scan_time
        if scan is None or not scan.ranges:
            return None
        if scan_time is None or time.time() - scan_time > self.approach_scan_max_age:
            rospy.logwarn_throttle(2.0, "Front lidar data is stale during owner approach")
            return None

        half_window = math.radians(max(1.0, self.approach_front_scan_degrees))
        values = []
        for index, distance in enumerate(scan.ranges):
            angle = scan.angle_min + index * scan.angle_increment
            if abs(angle) <= half_window and math.isfinite(distance) and scan.range_min < distance < scan.range_max:
                values.append(float(distance))
        if not values:
            return None
        return min(values)

    def estimate_owner_position_from_pointcloud(self, det, image_width, image_height):
        with self.lock:
            cloud = self.latest_pointcloud
            cloud_time = self.latest_pointcloud_time
        if cloud is None:
            self.last_pointcloud_reason = "no point cloud on %s" % self.points_topic
            rospy.logwarn_throttle(2.0, "No Kinect point cloud received on %s", self.points_topic)
            return None
        if cloud.height <= 1 or cloud.width <= 1:
            self.last_pointcloud_reason = "point cloud is not organized"
            rospy.logwarn_throttle(2.0, "Point cloud on %s is not organized; cannot sample by image bbox", self.points_topic)
            return None

        age = time.time() - cloud_time if cloud_time is not None else None
        if age is not None and age > self.approach_pointcloud_max_age:
            self.last_pointcloud_reason = "point cloud is stale: age=%.2fs" % age
            rospy.logwarn_throttle(2.0, "Kinect point cloud is stale during owner approach: age=%.2fs", age)
            return None

        if image_width is None or image_height is None or image_width <= 0 or image_height <= 0:
            self.last_pointcloud_reason = "invalid image size for point cloud ROI"
            return None

        scale_x = float(cloud.width) / float(image_width)
        scale_y = float(cloud.height) / float(image_height)
        xmin = clamp(int(float(det.xmin) * scale_x), 0, int(cloud.width) - 1)
        xmax = clamp(int(float(det.xmax) * scale_x), xmin + 1, int(cloud.width))
        ymin = clamp(int(float(det.ymin) * scale_y), 0, int(cloud.height) - 1)
        ymax = clamp(int(float(det.ymax) * scale_y), ymin + 1, int(cloud.height))

        box_w = max(1, xmax - xmin)
        box_h = max(1, ymax - ymin)
        x_margin = int(box_w * clamp(self.approach_pointcloud_roi_x_margin, 0.0, 0.45))
        roi_x1 = clamp(xmin + x_margin, 0, int(cloud.width) - 1)
        roi_x2 = clamp(xmax - x_margin, roi_x1 + 1, int(cloud.width))

        y_min_ratio = clamp(self.approach_pointcloud_roi_y_min_ratio, 0.0, 0.95)
        y_max_ratio = clamp(self.approach_pointcloud_roi_y_max_ratio, y_min_ratio + 0.01, 1.0)
        roi_y1 = clamp(ymin + int(box_h * y_min_ratio), 0, int(cloud.height) - 1)
        roi_y2 = clamp(ymin + int(box_h * y_max_ratio), roi_y1 + 1, int(cloud.height))

        stride = max(1, self.approach_pointcloud_stride)
        uvs = [(u, v) for v in range(roi_y1, roi_y2, stride) for u in range(roi_x1, roi_x2, stride)]
        if not uvs:
            self.last_pointcloud_reason = "empty point cloud ROI"
            return None

        points = []
        try:
            for x, y, z in pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True, uvs=uvs):
                x = float(x)
                y = float(y)
                z = float(z)
                if all(math.isfinite(value) for value in (x, y, z)):
                    points.append((x, y, z))
        except Exception as exc:
            self.last_pointcloud_reason = "failed to read point cloud ROI: %s" % exc
            rospy.logwarn_throttle(2.0, "Failed to read owner ROI from point cloud: %s", exc)
            return None

        if len(points) < self.approach_pointcloud_min_samples:
            self.last_pointcloud_reason = "too few valid ROI points: %d < %d" % (
                len(points),
                self.approach_pointcloud_min_samples,
            )
            rospy.logwarn_throttle(
                2.0,
                "Owner point cloud ROI has too few valid points: %d < %d",
                len(points),
                self.approach_pointcloud_min_samples,
            )
            return None

        raw = np.asarray(points, dtype=np.float32)
        raw_x = float(np.median(raw[:, 0]))
        raw_y = float(np.median(raw[:, 1]))
        raw_z = float(np.median(raw[:, 2]))
        mode = self.approach_pointcloud_mode
        if mode not in ("auto", "optical", "base"):
            mode = "auto"

        frame = (cloud.header.frame_id or "").lower()
        horizontal = math.hypot(raw_x, raw_y)
        depth_dominant = raw_z > max(0.8, horizontal * 1.35)
        use_optical = mode == "optical" or (mode == "auto" and ("optical" in frame or depth_dominant))
        if use_optical:
            forward_values = raw[:, 2]
            lateral_values = -raw[:, 0]
            # Kinect optical frame uses +Y downward; estimate height above floor from camera height.
            height_values = float(self.lying_surface_camera_height) - raw[:, 1]
            used_mode = "optical"
        else:
            forward_values = raw[:, 0]
            lateral_values = raw[:, 1]
            height_values = raw[:, 2]
            used_mode = "base"

        valid = []
        for forward, lateral, height in zip(forward_values, lateral_values, height_values):
            forward = float(forward)
            lateral = float(lateral)
            height = float(height)
            if (
                self.approach_min_depth <= forward <= self.approach_max_depth
                and abs(lateral) <= self.approach_max_depth
                and math.isfinite(height)
                and -0.20 <= height <= 2.00
            ):
                valid.append((forward, lateral, height))
        if len(valid) < self.approach_pointcloud_min_samples:
            self.last_pointcloud_reason = "too few in-range ROI points: %d < %d mode=%s raw=(%.2f, %.2f, %.2f)" % (
                len(valid),
                self.approach_pointcloud_min_samples,
                used_mode,
                raw_x,
                raw_y,
                raw_z,
            )
            rospy.logwarn_throttle(
                2.0,
                "Owner point cloud ROI has too few in-range points: %d < %d mode=%s raw=(%.2f, %.2f, %.2f)",
                len(valid),
                self.approach_pointcloud_min_samples,
                used_mode,
                raw_x,
                raw_y,
                raw_z,
            )
            return None

        valid = np.asarray(valid, dtype=np.float32)
        depth_percentile = clamp(self.approach_pointcloud_depth_percentile, 5.0, 50.0)
        forward_surface = float(np.percentile(valid[:, 0], depth_percentile))
        surface_band = max(0.05, self.approach_pointcloud_surface_band)
        surface = valid[valid[:, 0] <= forward_surface + surface_band]
        if len(surface) >= self.approach_pointcloud_min_samples:
            valid = surface

        person_x = float(np.median(valid[:, 0]))
        person_y = float(np.median(valid[:, 1]))
        surface_height_median = float(np.median(valid[:, 2]))
        surface_height_p20 = float(np.percentile(valid[:, 2], 20.0))
        surface_height_p80 = float(np.percentile(valid[:, 2], 80.0))
        distance = math.hypot(person_x, person_y)
        bearing = math.atan2(person_y, max(0.05, person_x))
        self.last_pointcloud_reason = ""
        return {
            "x": person_x,
            "y": person_y,
            "distance": distance,
            "bearing": bearing,
            "surface_height_median": surface_height_median,
            "surface_height_p20": surface_height_p20,
            "surface_height_p80": surface_height_p80,
            "mode": used_mode,
            "samples": int(len(valid)),
            "frame": cloud.header.frame_id,
        }

    def set_yolo_paused(self, paused):
        self.pause_yolo_pub.publish(Bool(data=bool(paused)))

    def wait_for_tts_subscriber(self):
        if not self.say_wait_for_subscribers:
            return True
        deadline = time.time() + self.say_wait_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            if self.say_pub.get_num_connections() > 0:
                return True
            time.sleep(0.05)
        rospy.logwarn("No subscribers connected to %s; TTS message may not be spoken", self.say_topic)
        return False

    def say(self, text, hold=None):
        rospy.loginfo("TTS: %s", text)
        self.wait_for_tts_subscriber()
        for index in range(self.say_repeat_count):
            self.say_pub.publish(String(data=text))
            if index + 1 < self.say_repeat_count and self.say_repeat_interval > 0:
                rospy.sleep(self.say_repeat_interval)
        delay = self.say_after_publish_delay if hold is None else float(hold)
        if delay > 0:
            rospy.sleep(delay)

    def publish_manipulator_command(self, lift, gripper):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ["lift", "gripper"]
        msg.position = [float(lift), float(gripper)]
        msg.velocity = [float(self.fall_assist_arm_lift_velocity), float(self.fall_assist_arm_gripper_velocity)]
        self.mani_ctrl_pub.publish(msg)

    def hold_manipulator_command(self, lift, gripper, seconds):
        duration = max(0.0, float(seconds))
        rate_hz = max(0.5, float(self.fall_assist_arm_command_rate))
        deadline = time.time() + duration
        rate = rospy.Rate(rate_hz)
        self.publish_manipulator_command(lift, gripper)
        while not rospy.is_shutdown() and time.time() < deadline:
            self.publish_manipulator_command(lift, gripper)
            rate.sleep()

    def perform_fall_assist_arm_motion(self):
        if not self.fall_assist_arm_enabled:
            rospy.loginfo("Fall assist arm motion is disabled")
            return True

        rospy.loginfo(
            "Fall assist arm motion: extend lift=%.2f gripper=%.2f hold=%.1fs then retract lift=%.2f gripper=%.2f topic=%s",
            self.fall_assist_arm_extend_lift,
            self.fall_assist_arm_extend_gripper,
            self.fall_assist_arm_hold_seconds,
            self.fall_assist_arm_retract_lift,
            self.fall_assist_arm_retract_gripper,
            self.mani_ctrl_topic,
        )

        self.hold_manipulator_command(
            self.fall_assist_arm_extend_lift,
            self.fall_assist_arm_extend_gripper,
            self.fall_assist_arm_extend_wait + self.fall_assist_arm_hold_seconds,
        )
        self.hold_manipulator_command(
            self.fall_assist_arm_retract_lift,
            self.fall_assist_arm_retract_gripper,
            self.fall_assist_arm_retract_wait,
        )
        if self.fall_assist_arm_completion_wait > 0:
            rospy.loginfo(
                "Fall assist arm retract command complete; waiting %.1fs before next waypoint",
                self.fall_assist_arm_completion_wait,
            )
            rospy.sleep(self.fall_assist_arm_completion_wait)
        return True

    def wait_for_hardware_inputs(self):
        deadline = time.time() + max(1.0, self.hardware_check_timeout)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                has_image = self.latest_image_time is not None
                has_scan = self.latest_scan_time is not None
                has_odom = self.latest_odom_time is not None

            image_ok = has_image or not self.require_kinect
            scan_ok = has_scan or not self.require_lidar
            odom_ok = has_odom or not self.require_odom
            if image_ok and scan_ok and odom_ok:
                rospy.loginfo(
                    "Real robot inputs ready: image=%s scan=%s odom=%s",
                    has_image,
                    has_scan,
                    has_odom,
                )
                return True
            rospy.loginfo_throttle(
                2.0,
                "Waiting for real robot inputs: image=%s scan=%s odom=%s",
                has_image,
                has_scan,
                has_odom,
            )
            rate.sleep()

        missing = []
        details = []
        with self.lock:
            if self.require_kinect and self.latest_image_time is None:
                missing.append(self.image_topic)
                details.append(self.describe_missing_topic(self.image_topic, "Kinect color image"))
            if self.require_lidar and self.latest_scan_time is None:
                missing.append(self.scan_topic)
                details.append(self.describe_missing_topic(self.scan_topic, "lidar scan"))
            if self.require_odom and self.latest_odom_time is None:
                missing.append(self.odom_topic)
                details.append(self.describe_missing_topic(self.odom_topic, "wheel odometry"))

        message = "required real robot inputs are not receiving messages: %s" % ", ".join(missing)
        if details:
            message += "; " + "; ".join(details)
        raise MissingHardwareError(message)

    def is_topic_advertised(self, topic_name):
        try:
            published_topics = rospy.get_published_topics("/")
        except Exception as exc:
            rospy.logwarn("Unable to query published topics while checking %s: %s", topic_name, exc)
            return False

        normalized = topic_name.rstrip("/") or "/"
        for name, _topic_type in published_topics:
            if (name.rstrip("/") or "/") == normalized:
                return True
        return False

    def describe_missing_topic(self, topic_name, label):
        if self.is_topic_advertised(topic_name):
            return "%s topic %s is advertised but produced no messages; check `rostopic hz %s`" % (
                label,
                topic_name,
                topic_name,
            )
        return "%s topic %s is not advertised; check the corresponding driver launch" % (label, topic_name)

    def waypoint_lookup_names(self, waypoint_name):
        target_name = str(waypoint_name).strip()
        lookup_names = [target_name]
        if target_name == self.exit_waypoint_name:
            for alias in self.exit_waypoint_aliases:
                if alias and alias not in lookup_names:
                    lookup_names.append(alias)
        return lookup_names

    def waypoint_display_name(self, waypoint_name):
        waypoint_name = str(waypoint_name).strip()
        return self.waypoint_speech_names.get(waypoint_name, waypoint_name)

    def rename_exit_waypoint_alias_if_needed(self):
        if not self.rename_exit_waypoint_alias or not self.exit_waypoint_name or not self.exit_waypoint_aliases:
            return False
        if not os.path.exists(self.waypoint_file):
            return False

        try:
            with open(self.waypoint_file, "r", encoding="utf-8") as waypoint_file:
                content = waypoint_file.read()
        except Exception as exc:
            rospy.logwarn("Unable to read waypoint file for exit rename: %s", exc)
            return False

        exit_pattern = r"(<Name>\s*)%s(\s*</Name>)" % re.escape(self.exit_waypoint_name)
        if re.search(exit_pattern, content):
            return False

        for alias in self.exit_waypoint_aliases:
            alias_pattern = r"(<Name>\s*)%s(\s*</Name>)" % re.escape(alias)

            def replace_name(match):
                return match.group(1) + self.exit_waypoint_name + match.group(2)

            renamed_content, count = re.subn(alias_pattern, replace_name, content, count=1)
            if count <= 0:
                continue
            try:
                with open(self.waypoint_file, "w", encoding="utf-8") as waypoint_file:
                    waypoint_file.write(renamed_content)
                rospy.loginfo(
                    "Renamed waypoint alias %s to %s in %s",
                    alias,
                    self.exit_waypoint_name,
                    self.waypoint_file,
                )
                return True
            except Exception as exc:
                rospy.logwarn(
                    "Unable to rename waypoint alias %s to %s in %s: %s",
                    alias,
                    self.exit_waypoint_name,
                    self.waypoint_file,
                    exc,
                )
                return False
        return False

    def load_waypoint_pose(self, waypoint_name=None):
        if not os.path.exists(self.waypoint_file):
            raise RuntimeError("waypoint file not found: %s" % self.waypoint_file)

        target_name = str(waypoint_name or self.waypoint_name).strip()
        lookup_names = self.waypoint_lookup_names(target_name)
        root = ET.parse(self.waypoint_file).getroot()
        for waypoint in root.findall("Waypoint"):
            name = waypoint.findtext("Name", "").strip()
            if name not in lookup_names:
                continue

            pose = Pose()
            pose.position.x = float(waypoint.findtext("Pos_x", "0"))
            pose.position.y = float(waypoint.findtext("Pos_y", "0"))
            pose.position.z = float(waypoint.findtext("Pos_z", "0"))
            pose.orientation.x = float(waypoint.findtext("Ori_x", "0"))
            pose.orientation.y = float(waypoint.findtext("Ori_y", "0"))
            pose.orientation.z = float(waypoint.findtext("Ori_z", "0"))
            pose.orientation.w = float(waypoint.findtext("Ori_w", "1"))
            return pose

        raise RuntimeError("waypoint not found: %s in %s" % (target_name, self.waypoint_file))

    def clear_move_base_costmaps(self, reason):
        try:
            rospy.loginfo("Clearing move_base costmaps %s", reason)
            rospy.wait_for_service(self.clear_costmaps_service, timeout=self.clear_costmaps_timeout)
            clear_costmaps = rospy.ServiceProxy(self.clear_costmaps_service, Empty)
            clear_costmaps()
            rospy.sleep(0.3)
            return True
        except Exception as exc:
            rospy.logwarn("Unable to clear move_base costmaps %s: %s", reason, exc)
            return False

    def send_navigation_goal(self, goal, timeout=None, label=None):
        wait_timeout = self.navigate_timeout if timeout is None else max(0.1, float(timeout))
        target_label = label or self.waypoint_name
        self.move_base.send_goal(goal)
        finished = self.move_base.wait_for_result(rospy.Duration(wait_timeout))
        if not finished:
            self.move_base.cancel_goal()
            return False, "move_base timed out while navigating to %s" % target_label

        state = self.move_base.get_state()
        if state != GoalStatus.SUCCEEDED:
            return False, "move_base failed with state %s" % state
        return True, ""

    def ensure_move_base_for_approach(self):
        if not self.approach_navigation_enabled:
            return False
        if self.move_base.wait_for_server(rospy.Duration(max(0.1, self.approach_navigation_server_timeout))):
            return True
        self.last_approach_failure_reason = "move_base action server unavailable for owner approach"
        rospy.logwarn(self.last_approach_failure_reason)
        return False

    def lookup_robot_navigation_pose(self):
        if self.tf_listener is None:
            self.last_approach_failure_reason = "tf is unavailable; cannot convert owner approach goal to map frame"
            rospy.logwarn(self.last_approach_failure_reason)
            return None
        try:
            self.tf_listener.waitForTransform(
                self.approach_navigation_frame,
                self.approach_navigation_base_frame,
                rospy.Time(0),
                rospy.Duration(max(0.1, self.approach_navigation_tf_timeout)),
            )
            translation, rotation = self.tf_listener.lookupTransform(
                self.approach_navigation_frame,
                self.approach_navigation_base_frame,
                rospy.Time(0),
            )
            yaw = tf.transformations.euler_from_quaternion(rotation)[2]
            return float(translation[0]), float(translation[1]), float(yaw)
        except Exception as exc:
            self.last_approach_failure_reason = "cannot transform %s to %s for owner approach: %s" % (
                self.approach_navigation_base_frame,
                self.approach_navigation_frame,
                exc,
            )
            rospy.logwarn(self.last_approach_failure_reason)
            return None

    def relative_navigation_goal(self, forward, lateral=0.0, yaw=0.0):
        robot_pose = self.lookup_robot_navigation_pose()
        if robot_pose is None:
            return None
        robot_x, robot_y, robot_yaw = robot_pose
        map_x, map_y = self.relative_navigation_xy(robot_x, robot_y, robot_yaw, forward, lateral)
        map_yaw = robot_yaw + float(yaw)

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.approach_navigation_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = map_x
        goal.target_pose.pose.position.y = map_y
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation.z = math.sin(map_yaw * 0.5)
        goal.target_pose.pose.orientation.w = math.cos(map_yaw * 0.5)
        return goal

    @staticmethod
    def relative_navigation_xy(robot_x, robot_y, robot_yaw, forward, lateral):
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        map_x = robot_x + cos_yaw * float(forward) - sin_yaw * float(lateral)
        map_y = robot_y + sin_yaw * float(forward) + cos_yaw * float(lateral)
        return map_x, map_y

    def navigation_goal_remaining_distance(self, goal):
        robot_pose = self.lookup_robot_navigation_pose()
        if robot_pose is None:
            return None
        robot_x, robot_y, _robot_yaw = robot_pose
        goal_x = float(goal.target_pose.pose.position.x)
        goal_y = float(goal.target_pose.pose.position.y)
        return math.hypot(goal_x - robot_x, goal_y - robot_y)

    def navigate_relative_for_approach(
        self,
        forward,
        lateral=0.0,
        yaw=0.0,
        timeout=None,
        label="owner approach",
        lidar_guard_distance=None,
    ):
        move_distance = math.hypot(float(forward), float(lateral))
        if move_distance <= self.approach_navigation_min_distance:
            self.stop_base()
            rospy.loginfo(
                "%s relative navigation skipped near target: move=%.2f min=%.2f",
                label,
                move_distance,
                self.approach_navigation_min_distance,
            )
            return True
        if not self.ensure_move_base_for_approach():
            return False

        if self.clear_costmaps_before_navigation and self.approach_navigation_clear_costmaps_before_goal:
            self.clear_move_base_costmaps("before %s" % label)

        goal = self.relative_navigation_goal(forward, lateral, yaw)
        if goal is None:
            return False
        wait_timeout = self.approach_navigation_timeout if timeout is None else timeout
        rospy.loginfo(
            "%s using move_base map goal from relative command: frame=%s rel=(%.2f, %.2f, %.2f) map=(%.2f, %.2f) timeout=%.1f",
            label,
            self.approach_navigation_frame,
            forward,
            lateral,
            yaw,
            goal.target_pose.pose.position.x,
            goal.target_pose.pose.position.y,
            wait_timeout,
        )
        success, error_message = self.send_approach_navigation_goal(
            goal,
            move_distance,
            timeout=wait_timeout,
            label=label,
            lidar_guard_distance=lidar_guard_distance,
        )
        obstacle_failure = "blocked by close obstacle" in error_message or "stuck or blocked" in error_message
        if not success and self.approach_navigation_retry_after_clear and not obstacle_failure:
            rospy.logwarn("%s; clearing costmaps and retrying %s once", error_message, label)
            self.clear_move_base_costmaps("after %s failure" % label)
            goal.target_pose.header.stamp = rospy.Time.now()
            success, error_message = self.send_approach_navigation_goal(
                goal,
                move_distance,
                timeout=wait_timeout,
                label=label,
                lidar_guard_distance=lidar_guard_distance,
            )

        self.stop_base()
        if not success:
            self.last_approach_failure_reason = error_message
            rospy.logwarn("%s failed: %s", label, error_message)
            return False
        rospy.loginfo("%s relative navigation complete", label)
        return True

    def send_approach_navigation_goal(
        self,
        goal,
        move_distance,
        timeout=None,
        label="owner approach",
        lidar_guard_distance=None,
        accept_near_position=True,
        allow_slow_finish=True,
        stop_still_duration=None,
        early_stop_owner_xy=None,
        early_stop_owner_distance=None,
    ):
        self.move_base.send_goal(goal)
        start_xy = self.wait_for_odom_xy(timeout=0.2)
        deadline = time.time() + max(0.1, float(timeout if timeout is not None else self.approach_navigation_timeout))
        slow_finish_cycles = 0
        best_remaining = float(move_distance)
        last_progress_time = time.time()
        movement_seen = False
        still_since = None
        rate = rospy.Rate(10)

        while not rospy.is_shutdown() and time.time() < deadline:
            state = self.move_base.get_state()
            travelled = None
            remaining = None
            goal_remaining = self.navigation_goal_remaining_distance(goal)
            if goal_remaining is not None:
                remaining = goal_remaining
            if start_xy is not None:
                travelled = self.xy_distance(start_xy, self.get_latest_odom_xy())
                if remaining is None and travelled is not None:
                    remaining = max(0.0, float(move_distance) - travelled)

            current_speed = None
            owner_distance = None
            if stop_still_duration is not None:
                current_speed = self.get_latest_odom_linear_speed(max_age=self.approach_odom_speed_max_age)
                if travelled is not None and travelled >= max(0.03, self.approach_navigation_min_distance):
                    movement_seen = True
                elif current_speed is not None and current_speed > self.approach_slow_finish_linear_speed:
                    movement_seen = True

            if early_stop_owner_xy is not None and early_stop_owner_distance is not None and movement_seen:
                robot_pose = self.lookup_robot_navigation_pose()
                if robot_pose is not None:
                    owner_distance = math.hypot(
                        float(early_stop_owner_xy[0]) - robot_pose[0],
                        float(early_stop_owner_xy[1]) - robot_pose[1],
                    )
                    if owner_distance <= float(early_stop_owner_distance):
                        self.move_base.cancel_goal()
                        self.stop_base()
                        rospy.loginfo(
                            "%s stopped in owner safety circle: owner_distance=%.2f threshold=%.2f",
                            label,
                            owner_distance,
                            float(early_stop_owner_distance),
                        )
                        return True, ""

            if (
                stop_still_duration is not None
                and movement_seen
                and current_speed is not None
                and current_speed <= self.approach_slow_finish_linear_speed
            ):
                near_target = remaining is not None and remaining <= max(0.20, self.approach_slow_finish_tolerance)
                if owner_distance is not None:
                    near_target = near_target or owner_distance <= float(early_stop_owner_distance) + 0.08
                if near_target:
                    if still_since is None:
                        still_since = time.time()
                    elif time.time() - still_since >= float(stop_still_duration):
                        self.move_base.cancel_goal()
                        self.stop_base()
                        rospy.loginfo(
                            "%s accepted as arrived after standing still for %.1fs",
                            label,
                            float(stop_still_duration),
                        )
                        return True, ""
                else:
                    still_since = None
            elif stop_still_duration is not None:
                still_since = None

            if state == GoalStatus.SUCCEEDED:
                success_tolerance = max(self.approach_navigation_min_distance, self.approach_slow_finish_tolerance)
                if remaining is None or remaining <= success_tolerance:
                    return True, ""
                return False, (
                    "move_base reported success before reaching %s approach goal: remaining=%.2f tolerance=%.2f"
                    % (label, remaining, success_tolerance)
                )
            if state in (
                GoalStatus.PREEMPTED,
                GoalStatus.ABORTED,
                GoalStatus.REJECTED,
                GoalStatus.RECALLED,
                GoalStatus.LOST,
            ):
                return False, "move_base failed while navigating to %s with state %s" % (label, state)

            if remaining is not None and remaining <= self.approach_navigation_min_distance:
                self.move_base.cancel_goal()
                return True, ""

            front_distance = None
            if lidar_guard_distance is not None:
                front_distance = self.front_scan_distance()
                if front_distance is not None and front_distance <= lidar_guard_distance:
                    self.move_base.cancel_goal()
                    if remaining is not None and remaining <= self.approach_slow_finish_tolerance:
                        rospy.loginfo(
                            "%s navigation accepted at lidar guard near target: travelled=%.2f target=%.2f remaining=%.2f lidar=%.2f guard=%.2f",
                            label,
                            travelled if travelled is not None else -1.0,
                            move_distance,
                            remaining,
                            front_distance,
                            lidar_guard_distance,
                        )
                        return True, ""
                    return False, "%s blocked by close obstacle: lidar=%.2f guard=%.2f remaining=%s" % (
                        label,
                        front_distance,
                        lidar_guard_distance,
                        "%.2f" % remaining if remaining is not None else "unknown",
                    )

            if remaining is not None:
                if remaining + max(0.0, self.approach_navigation_stuck_min_progress) < best_remaining:
                    best_remaining = remaining
                    last_progress_time = time.time()

                slow_finish_cycles, actual_speed, slow_reason = self.update_approach_slow_finish_cycles(
                    remaining,
                    self.approach_linear_speed,
                    slow_finish_cycles,
                )
                if slow_finish_cycles >= self.approach_slow_finish_cycles:
                    self.move_base.cancel_goal()
                    rospy.loginfo(
                        "%s navigation accepted slow/stop near target: travelled=%.2f target=%.2f remaining=%.2f odom_speed=%.3f reason=%s cycles=%d",
                        label,
                        travelled if travelled is not None else -1.0,
                        move_distance,
                        remaining,
                        actual_speed if actual_speed is not None else -1.0,
                        slow_reason,
                        slow_finish_cycles,
                    )
                    return True, ""

                stuck_timeout = max(0.0, self.approach_navigation_stuck_timeout)
                if stuck_timeout > 0.0 and time.time() - last_progress_time >= stuck_timeout:
                    actual_speed = self.get_latest_odom_linear_speed(max_age=self.approach_odom_speed_max_age)
                    if actual_speed is None or actual_speed <= self.approach_navigation_stuck_linear_speed:
                        if front_distance is None:
                            front_distance = self.front_scan_distance()
                        self.move_base.cancel_goal()
                        return False, (
                            "%s stuck or blocked before target: remaining=%.2f best_remaining=%.2f "
                            "odom_speed=%s lidar=%s"
                            % (
                                label,
                                remaining,
                                best_remaining,
                                "%.3f" % actual_speed if actual_speed is not None else "unknown",
                                "%.2f" % front_distance if front_distance is not None else "unknown",
                            )
                        )

            rate.sleep()

        self.move_base.cancel_goal()
        return False, "move_base timed out while navigating to %s" % label

    def current_pose_for_plan(self, target_frame):
        if self.tf_listener is None:
            return None
        try:
            self.tf_listener.waitForTransform(
                target_frame,
                self.approach_navigation_base_frame,
                rospy.Time(0),
                rospy.Duration(max(0.1, self.approach_navigation_tf_timeout)),
            )
            translation, rotation = self.tf_listener.lookupTransform(
                target_frame,
                self.approach_navigation_base_frame,
                rospy.Time(0),
            )
        except Exception as exc:
            rospy.logwarn_throttle(
                3.0,
                "Cannot transform current robot pose from %s to %s for waving approach plan check: %s",
                self.approach_navigation_base_frame,
                target_frame,
                exc,
            )
            return None

        pose = PoseStamped()
        pose.header.frame_id = target_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = float(translation[0])
        pose.pose.position.y = float(translation[1])
        pose.pose.position.z = float(translation[2])
        pose.pose.orientation.x = float(rotation[0])
        pose.pose.orientation.y = float(rotation[1])
        pose.pose.orientation.z = float(rotation[2])
        pose.pose.orientation.w = float(rotation[3])
        return pose

    def waving_goal_has_global_plan(self, goal):
        if not self.waving_approach_plan_check:
            return True
        frame_id = goal.target_pose.header.frame_id
        start = self.current_pose_for_plan(frame_id)
        if start is None:
            return True
        try:
            rospy.wait_for_service(self.waving_approach_plan_service, timeout=0.3)
            response = self.waving_make_plan(
                start,
                goal.target_pose,
                self.waving_approach_plan_tolerance,
            )
        except Exception as exc:
            rospy.logwarn_throttle(
                3.0,
                "Cannot check waving owner approach plan via %s: %s",
                self.waving_approach_plan_service,
                exc,
            )
            return True

        pose_count = len(response.plan.poses)
        if pose_count <= 1:
            rospy.logwarn(
                "Rejected waving owner candidate with no global plan: frame=%s goal=(%.2f, %.2f)",
                frame_id,
                goal.target_pose.pose.position.x,
                goal.target_pose.pose.position.y,
            )
            return False
        plan_length = 0.0
        total_turn = 0.0
        previous_pose = response.plan.poses[0].pose.position
        previous_heading = None
        for plan_pose in response.plan.poses[1:]:
            current_pose = plan_pose.pose.position
            delta_x = float(current_pose.x) - float(previous_pose.x)
            delta_y = float(current_pose.y) - float(previous_pose.y)
            segment_length = math.hypot(delta_x, delta_y)
            plan_length += segment_length
            if segment_length > 1e-3:
                heading = math.atan2(delta_y, delta_x)
                if previous_heading is not None:
                    total_turn += abs(signed_angle_diff(heading, previous_heading))
                previous_heading = heading
            previous_pose = current_pose
        start_position = start.pose.position
        direct_distance = math.hypot(
            float(goal.target_pose.pose.position.x) - float(start_position.x),
            float(goal.target_pose.pose.position.y) - float(start_position.y),
        )
        if (
            plan_length > max(0.5, direct_distance) * self.waving_approach_plan_detour_ratio
            and plan_length > direct_distance + self.waving_approach_plan_detour_margin
        ):
            rospy.logwarn(
                "Rejected waving owner candidate with large detour: plan=%.2fm direct=%.2fm ratio=%.2f",
                plan_length,
                direct_distance,
                plan_length / max(0.01, direct_distance),
            )
            return False
        if total_turn > self.waving_approach_plan_turn_limit:
            rospy.logwarn(
                "Rejected waving owner candidate with unstable turns: total_turn=%.2frad limit=%.2frad",
                total_turn,
                self.waving_approach_plan_turn_limit,
            )
            return False
        rospy.loginfo_throttle(
            2.0,
            "Waving owner approach candidate has global plan with %d poses length=%.2fm direct=%.2fm turns=%.2frad",
            pose_count,
            plan_length,
            direct_distance,
            total_turn,
        )
        return True

    def waving_owner_standoff_candidates(self, position, standoff_distance):
        distance = max(0.0, float(position.get("distance", 0.0)))
        if distance <= 1e-3:
            return []

        owner_x = float(position.get("x", 0.0))
        owner_y = float(position.get("y", 0.0))
        if not all(math.isfinite(value) for value in (owner_x, owner_y)):
            return []

        min_owner_distance = clamp(float(self.waving_approach_min_owner_distance), 0.35, 0.50)
        max_owner_distance = clamp(float(self.waving_approach_max_owner_distance), min_owner_distance, 0.50)
        requested_standoff = clamp(abs(float(standoff_distance)), min_owner_distance, max_owner_distance)
        raw_distances = [requested_standoff] + list(self.waving_approach_candidate_distances)
        candidate_distances = []
        seen_distances = set()
        for raw_distance in raw_distances:
            try:
                distance_value = float(raw_distance)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(distance_value):
                continue
            owner_clearance = clamp(
                max(abs(distance_value), self.waving_approach_safety_radius),
                min_owner_distance,
                max_owner_distance,
            )
            key = round(owner_clearance, 2)
            if key not in seen_distances:
                seen_distances.add(key)
                candidate_distances.append(key)

        raw_angles = list(self.waving_approach_candidate_angles_deg)
        if not raw_angles:
            raw_angles = [65, -65, 95, -95, 35, -35, 0]
        finite_angles = []
        for raw_angle in raw_angles:
            try:
                angle = float(raw_angle)
            except (TypeError, ValueError):
                continue
            if math.isfinite(angle):
                finite_angles.append(angle)
        if all(abs(angle) > 1e-3 for angle in finite_angles):
            finite_angles.append(0.0)

        unit_x = owner_x / distance
        unit_y = owner_y / distance
        near_side_angle = math.atan2(-unit_y, -unit_x)
        candidates = []
        seen = set()
        for owner_clearance in candidate_distances:
            for angle_deg in finite_angles:
                offset_angle = near_side_angle + math.radians(angle_deg)
                goal_x = owner_x + math.cos(offset_angle) * owner_clearance
                goal_y = owner_y + math.sin(offset_angle) * owner_clearance
                goal_distance = math.hypot(goal_x, goal_y)
                if goal_distance <= self.approach_navigation_min_distance:
                    continue

                face_x = owner_x - goal_x
                face_y = owner_y - goal_y
                yaw = math.atan2(face_y, max(0.05, face_x))
                key = (round(goal_x, 2), round(goal_y, 2), round(yaw, 2))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    "forward": goal_x,
                    "lateral": goal_y,
                    "yaw": yaw,
                    "owner_clearance": owner_clearance,
                    "angle_deg": angle_deg,
                    "goal_distance": goal_distance,
                })

        return candidates

    def navigate_to_waving_owner_candidates(
        self,
        position,
        standoff_distance,
        timeout=None,
        label="waving owner move_base approach",
        lidar_guard_distance=None,
    ):
        if not self.waving_approach_candidate_enabled:
            return self.navigate_to_owner_standoff(
                position,
                standoff_distance,
                timeout=timeout,
                label=label,
                lidar_guard_distance=lidar_guard_distance,
            )
        if not self.ensure_move_base_for_approach():
            return False

        candidates = self.waving_owner_standoff_candidates(position, standoff_distance)
        if not candidates:
            self.last_approach_failure_reason = "no valid waving owner standoff candidates"
            rospy.logwarn(self.last_approach_failure_reason)
            return False

        rospy.loginfo(
            "Generated %d waving owner approach candidates %.2f-%.2fm from owner",
            len(candidates),
            self.waving_approach_min_owner_distance,
            self.waving_approach_max_owner_distance,
        )
        last_error = "no reachable waving owner candidate %.2f-%.2fm from owner" % (
            self.waving_approach_min_owner_distance,
            self.waving_approach_max_owner_distance,
        )
        for index, candidate in enumerate(candidates, start=1):
            goal = self.relative_navigation_goal(
                candidate["forward"],
                candidate["lateral"],
                candidate["yaw"],
            )
            if goal is None:
                last_error = "cannot create waving owner candidate %d goal" % index
                continue
            if not self.waving_goal_has_global_plan(goal):
                last_error = "waving owner candidate %d has no global plan" % index
                continue
            robot_pose = self.lookup_robot_navigation_pose()
            if robot_pose is None:
                last_error = "cannot locate robot pose for waving owner safety circle"
                continue
            owner_map_xy = self.relative_navigation_xy(
                robot_pose[0],
                robot_pose[1],
                robot_pose[2],
                float(position.get("x", 0.0)),
                float(position.get("y", 0.0)),
            )
            rospy.loginfo(
                "%s trying candidate %d/%d: rel=(%.2f, %.2f) owner_clearance=%.2f angle=%.0f yaw=%.2f map=(%.2f, %.2f)",
                label,
                index,
                len(candidates),
                candidate["forward"],
                candidate["lateral"],
                candidate["owner_clearance"],
                candidate["angle_deg"],
                candidate["yaw"],
                goal.target_pose.pose.position.x,
                goal.target_pose.pose.position.y,
            )
            candidate_label = "%s candidate %d" % (label, index)
            success, error_message = self.send_approach_navigation_goal(
                goal,
                candidate["goal_distance"],
                timeout=timeout,
                label=candidate_label,
                lidar_guard_distance=lidar_guard_distance,
                stop_still_duration=self.waving_approach_still_duration,
                early_stop_owner_xy=owner_map_xy,
                early_stop_owner_distance=self.waving_approach_safety_radius,
            )
            obstacle_failure = "blocked by close obstacle" in error_message or "stuck or blocked" in error_message
            if not success and self.waving_approach_retry_after_clear and not obstacle_failure:
                rospy.logwarn("%s; clearing costmaps and retrying candidate %d once", error_message, index)
                self.clear_move_base_costmaps("after %s failure" % candidate_label)
                goal.target_pose.header.stamp = rospy.Time.now()
                success, error_message = self.send_approach_navigation_goal(
                    goal,
                    candidate["goal_distance"],
                    timeout=timeout,
                    label=candidate_label,
                    lidar_guard_distance=lidar_guard_distance,
                    stop_still_duration=self.waving_approach_still_duration,
                    early_stop_owner_xy=owner_map_xy,
                    early_stop_owner_distance=self.waving_approach_safety_radius,
                )

            self.stop_base()
            if success:
                rospy.loginfo(
                    "%s reached candidate %d with owner clearance %.2fm",
                    label,
                    index,
                    candidate["owner_clearance"],
                )
                return True
            last_error = error_message or "waving owner candidate %d failed" % index
            self.last_approach_failure_reason = last_error
            rospy.logwarn("%s candidate %d failed; trying next candidate", label, index)

        self.stop_base()
        self.last_approach_failure_reason = last_error
        rospy.logwarn("%s failed: %s", label, self.last_approach_failure_reason)
        return False

    def navigate_to_owner_standoff(
        self,
        position,
        standoff_distance,
        max_travel=None,
        timeout=None,
        label="owner approach",
        lidar_guard_distance=None,
        lateral_offset=0.0,
    ):
        distance = max(0.0, float(position.get("distance", 0.0)))
        if distance <= 1e-3:
            self.stop_base()
            rospy.loginfo("%s skipped: owner distance unavailable/zero", label)
            return True

        travel_distance = max(0.0, distance - max(0.0, float(standoff_distance)))
        if max_travel is not None:
            travel_distance = min(travel_distance, max(0.0, float(max_travel)))
        if travel_distance <= self.approach_navigation_min_distance:
            self.stop_base()
            rospy.loginfo(
                "%s already within standoff/navigation threshold: distance=%.2f standoff=%.2f travel=%.2f",
                label,
                distance,
                standoff_distance,
                travel_distance,
            )
            return True

        scale = travel_distance / distance
        owner_x = float(position.get("x", 0.0))
        owner_y = float(position.get("y", 0.0))
        offset = float(lateral_offset)
        forward = owner_x * scale
        lateral = owner_y * scale
        if abs(offset) > 1e-3:
            unit_x = owner_x / distance
            unit_y = owner_y / distance
            forward += -unit_y * offset
            lateral += unit_x * offset
        face_x = owner_x - forward
        face_y = owner_y - lateral
        yaw = math.atan2(face_y, max(0.05, face_x))
        if abs(offset) > 1e-3:
            rospy.loginfo(
                "%s using lateral standoff offset %.2fm: owner=(%.2f, %.2f) goal_rel=(%.2f, %.2f)",
                label,
                offset,
                owner_x,
                owner_y,
                forward,
                lateral,
            )
        return self.navigate_relative_for_approach(
            forward,
            lateral,
            yaw=yaw,
            timeout=timeout,
            label=label,
            lidar_guard_distance=lidar_guard_distance,
        )

    def navigate_to_waypoint(self, waypoint_name=None):
        if waypoint_name is not None:
            self.waypoint_name = str(waypoint_name).strip()
        if not self.navigate_enabled:
            rospy.loginfo("Navigation disabled; assuming robot is already at %s", self.waypoint_name)
            return

        rospy.loginfo("Waiting for move_base action server")
        if not self.move_base.wait_for_server(rospy.Duration(25.0)):
            raise RuntimeError("move_base action server is not available")

        if self.clear_costmaps_before_navigation:
            self.clear_move_base_costmaps("before navigating to %s" % self.waypoint_name)

        pose = self.load_waypoint_pose(self.waypoint_name)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = pose

        rospy.loginfo(
            "Navigating real robot to waypoint %s: x=%.3f y=%.3f",
            self.waypoint_name,
            pose.position.x,
            pose.position.y,
        )

        success, error_message = self.send_navigation_goal(goal)
        if not success and self.retry_navigation_after_clear:
            rospy.logwarn("%s; clearing costmaps and retrying once", error_message)
            self.clear_move_base_costmaps("after navigation failure")
            goal.target_pose.header.stamp = rospy.Time.now()
            success, error_message = self.send_navigation_goal(goal)

        if not success:
            raise RuntimeError(error_message)
        rospy.loginfo("Arrived at waypoint %s", self.waypoint_name)

    def snapshot_candidates(self):
        with self.lock:
            if self.latest_image is None or not self.latest_detections:
                return []
            image = self.latest_image.copy()
            detections = list(self.latest_detections)

        height, width = image.shape[:2]
        candidates = []
        for det in detections:
            class_name = getattr(det, "class_name", "")
            if class_name and class_name != "person":
                continue
            if float(det.score) < self.detection_min_score:
                continue

            xmin = clamp(int(det.xmin), 0, width - 1)
            ymin = clamp(int(det.ymin), 0, height - 1)
            xmax = clamp(int(det.xmax), 0, width - 1)
            ymax = clamp(int(det.ymax), 0, height - 1)
            if xmax <= xmin or ymax <= ymin:
                continue

            box_w = xmax - xmin
            box_h = ymax - ymin
            area_ratio = float(box_w * box_h) / float(width * height)
            if area_ratio < self.detection_min_area_ratio:
                continue

            face_crop_padding = clamp(float(self.face_crop_padding), 0.05, 0.40)
            pad_x = int(box_w * face_crop_padding)
            pad_y = int(box_h * face_crop_padding)
            cx1 = clamp(xmin - pad_x, 0, width - 1)
            cy1 = clamp(ymin - pad_y, 0, height - 1)
            cx2 = clamp(xmax + pad_x, 0, width - 1)
            cy2 = clamp(ymax + pad_y, 0, height - 1)
            crop = image[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            center_error = abs((float(det.center_x) / float(width)) - 0.5)
            priority = float(det.score) + area_ratio * 4.0 - center_error * 0.2
            candidates.append({
                "det": det,
                "crop": crop,
                "score": float(det.score),
                "area_ratio": area_ratio,
                "priority": priority,
                "image_width": width,
                "image_height": height,
            })

        candidates.sort(key=lambda item: item["priority"], reverse=True)
        return candidates[:self.verify_top_k]

    def crop_candidate_face_regions(self, candidate, include_extra_side_ratios=True):
        crop = candidate.get("crop")
        if crop is None or crop.size == 0:
            return []
        height, width = crop.shape[:2]
        top_ratio = clamp(float(self.face_crop_top_ratio), 0.35, 1.0)
        side_ratios = [float(self.face_crop_lying_side_ratio)]
        if include_extra_side_ratios:
            side_ratios.extend(float(ratio) for ratio in self.face_crop_lying_extra_side_ratios)

        det = candidate.get("det")
        if det is not None:
            person_width = max(1.0, float(det.xmax) - float(det.xmin))
            person_height = max(1.0, float(det.ymax) - float(det.ymin))
        else:
            person_width = float(width)
            person_height = float(height)

        regions = []
        seen = set()

        def add_region(name, x1, y1, x2, y2, allow_rotations):
            x1 = clamp(int(x1), 0, width - 1)
            y1 = clamp(int(y1), 0, height - 1)
            x2 = clamp(int(x2), x1 + 1, width)
            y2 = clamp(int(y2), y1 + 1, height)
            if x2 - x1 < 8 or y2 - y1 < 8:
                return
            key = (x1, y1, x2, y2)
            if key in seen:
                return
            seen.add(key)
            regions.append((name, crop[y1:y2, x1:x2], bool(allow_rotations)))

        if person_height >= person_width:
            upper_y2 = max(1, int(height * top_ratio))
            add_region("upper", 0, 0, width, upper_y2, allow_rotations=False)
        else:
            for raw_ratio in side_ratios:
                side_ratio = clamp(float(raw_ratio), 0.35, 0.75)
                side_width = max(1, int(width * side_ratio))
                ratio_name = int(round(side_ratio * 100.0))
                add_region("left_side_%d" % ratio_name, 0, 0, side_width, height, allow_rotations=True)
                add_region("right_side_%d" % ratio_name, width - side_width, 0, width, height, allow_rotations=True)

        return regions

    def face_image_variants(self, region_name, face_image, enhanced=False):
        if not enhanced or not self.face_candidate_enhance:
            yield region_name, face_image
            return

        try:
            lab_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
            lightness, channel_a, channel_b = cv2.split(lab_image)
            lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
            yield region_name + ":clahe", cv2.cvtColor(
                cv2.merge((lightness, channel_a, channel_b)),
                cv2.COLOR_LAB2BGR,
            )
        except Exception:
            pass

        yield region_name + ":up1.5", cv2.resize(
            face_image,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC,
        )

    def best_owner_face_match(self, candidate_embedding, owner_embeddings):
        return max(
            (
                (float(np.dot(item["embedding"], candidate_embedding)), item.get("path", "owner"))
                for item in owner_embeddings
            ),
            key=lambda value: value[0],
        )

    def face_region_orientations(self, region_name, face_image, allow_rotations):
        yield region_name, face_image
        if not allow_rotations or not self.face_crop_try_rotations:
            return
        yield region_name + ":rot90", cv2.rotate(face_image, cv2.ROTATE_90_CLOCKWISE)
        yield region_name + ":rot270", cv2.rotate(face_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        yield region_name + ":rot180", cv2.rotate(face_image, cv2.ROTATE_180)

    def verify_candidate_with_face(self, candidate):
        if not self.face_ready or self.face_app is None:
            return None, 0.0, "face unavailable"
        owner_embeddings = list(getattr(self, "owner_face_embeddings", []))
        if not owner_embeddings and self.owner_face_embedding is not None:
            owner_embeddings = [{"path": self.owner_image_path, "embedding": self.owner_face_embedding}]
        if not owner_embeddings:
            return None, 0.0, "face unavailable"

        face_regions = self.crop_candidate_face_regions(candidate)
        if not face_regions:
            return None, 0.0, "empty face crop"

        best_similarity = None
        best_reference_path = None
        best_region_name = None
        face_count = 0
        elapsed_ms = 0.0
        had_error = False

        def run_regions(regions, enhancement_passes):
            nonlocal best_similarity, best_reference_path, best_region_name
            nonlocal face_count, elapsed_ms, had_error
            for enhanced in enhancement_passes:
                for region_name, face_image, allow_rotations in regions:
                    if enhanced and not allow_rotations:
                        continue
                    for variant_name, variant_image in self.face_image_variants(region_name, face_image, enhanced):
                        for oriented_name, oriented_image in self.face_region_orientations(
                            variant_name,
                            variant_image,
                            allow_rotations,
                        ):
                            try:
                                start = time.time()
                                faces = self.face_app.get(oriented_image)
                                elapsed_ms += (time.time() - start) * 1000.0
                            except Exception as exc:
                                had_error = True
                                rospy.logwarn("Face verification failed on %s crop: %s", oriented_name, exc)
                                continue

                            for candidate_face in faces or []:
                                face_count += 1
                                candidate_embedding = self.normalize_embedding(candidate_face.embedding)
                                if candidate_embedding is None:
                                    continue
                                similarity, reference_path = self.best_owner_face_match(
                                    candidate_embedding,
                                    owner_embeddings,
                                )
                                if best_similarity is None or similarity > best_similarity:
                                    best_similarity = similarity
                                    best_reference_path = reference_path
                                    best_region_name = oriented_name
                            if best_similarity is not None and best_similarity >= self.face_accept_threshold:
                                return True
            return False

        # Most non-owners are far below the threshold in the primary side crop.
        # Keep the wider/enhanced search for borderline faces so low-confidence
        # owner views retain the same fallback behavior as before.
        if self.face_fast_pass_enabled:
            fast_regions = self.crop_candidate_face_regions(candidate, include_extra_side_ratios=False)
            accepted = run_regions(fast_regions, [False])
            if accepted:
                rospy.loginfo(
                    "Face owner verdict: accepted by fast pass similarity=%.3f crop=%s faces=%d elapsed=%.1fms",
                    best_similarity,
                    best_region_name,
                    face_count,
                    elapsed_ms,
                )
                return True, best_similarity, "face accepted"
            if (
                best_similarity is not None
                and not had_error
                and best_similarity <= self.face_fast_pass_reject_threshold
            ):
                rospy.loginfo(
                    "Face owner verdict: rejected by fast pass similarity=%.3f crop=%s faces=%d elapsed=%.1fms",
                    best_similarity,
                    best_region_name,
                    face_count,
                    elapsed_ms,
                )
                return False, best_similarity, "face rejected by fast pass"

        if self.face_fast_pass_enabled:
            # The primary regions have already been evaluated above. Continue
            # with only the remaining raw regions, then enhance all eligible
            # side regions so borderline owner views keep the old fallback.
            if run_regions(face_regions[len(fast_regions):], [False]):
                return True, best_similarity, "face accepted"
            if self.face_candidate_enhance:
                if run_regions(face_regions, [True]):
                    return True, best_similarity, "face accepted"
        else:
            enhancement_passes = [False]
            if self.face_candidate_enhance:
                enhancement_passes.append(True)
            run_regions(face_regions, enhancement_passes)

        if best_similarity is None:
            if had_error:
                return None, 0.0, "face verification error"
            rospy.loginfo("Face verifier saw no face in candidate crops")
            return None, 0.0, "no face"

        similarity = best_similarity
        reference_path = best_reference_path or self.owner_image_path
        reference_name = os.path.basename(reference_path)
        if similarity >= self.face_accept_threshold:
            rospy.loginfo(
                "Face owner verdict: accepted similarity=%.3f reference=%s crop=%s faces=%d elapsed=%.1fms",
                similarity,
                reference_name,
                best_region_name,
                face_count,
                elapsed_ms,
            )
            return True, similarity, "face accepted"
        if self.face_fast_reject and similarity <= self.face_reject_threshold:
            rospy.loginfo(
                "Face owner verdict: rejected similarity=%.3f best_reference=%s crop=%s faces=%d elapsed=%.1fms",
                similarity,
                reference_name,
                best_region_name,
                face_count,
                elapsed_ms,
            )
            return False, similarity, "face rejected"

        rospy.loginfo(
            "Face owner verdict: uncertain similarity=%.3f best_reference=%s crop=%s faces=%d elapsed=%.1fms",
            similarity,
            reference_name,
            best_region_name,
            face_count,
            elapsed_ms,
        )
        return None, similarity, "face uncertain"

    def verify_candidate(self, candidate):
        decision, score, reason = self.verify_candidate_with_face(candidate)
        if decision is True:
            return True, score, reason
        if decision is False:
            return False, score, reason
        if self.allow_unverified_owner:
            rospy.logwarn("Accepting unverified owner candidate for hardware debug only: %s", reason)
            return True, float(candidate.get("score", 0.0)), "unverified debug accept"
        return False, score, reason

    def wait_for_odom_yaw(self, timeout=3.0):
        deadline = time.time() + timeout
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() < deadline:
            yaw = self.get_latest_yaw()
            if yaw is not None:
                return yaw
            rate.sleep()
        return None

    def rotate_to_yaw(self, target_yaw, timeout=18.0):
        deadline = time.time() + timeout
        rate = rospy.Rate(15)
        while not rospy.is_shutdown() and time.time() < deadline:
            current_yaw = self.get_latest_yaw()
            if current_yaw is None:
                return False

            error = signed_angle_diff(target_yaw, current_yaw)
            if abs(error) < 0.08:
                self.stop_base()
                return True

            twist = Twist()
            twist.angular.z = clamp(1.0 * error, -self.return_angular_speed, self.return_angular_speed)
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.stop_base()
        return False

    def merge_scan_candidates(self, pool, candidates, yaw):
        for candidate in candidates:
            item = dict(candidate)
            item["yaw"] = yaw
            item["priority"] = float(item.get("priority", 0.0))

            replaced = False
            for index, existing in enumerate(pool):
                existing_yaw = existing.get("yaw")
                same_view = yaw is not None and existing_yaw is not None and abs(signed_angle_diff(yaw, existing_yaw)) < 0.35
                same_center = abs(
                    (float(item["det"].center_x) / float(item["image_width"]))
                    - (float(existing["det"].center_x) / float(existing["image_width"]))
                ) < 0.15
                if same_view and same_center:
                    if item["priority"] > existing.get("priority", 0.0):
                        pool[index] = item
                    replaced = True
                    break
            if not replaced:
                pool.append(item)

        pool.sort(key=lambda value: value.get("priority", 0.0), reverse=True)
        return pool[:self.scan_candidate_pool_size]

    def verify_candidate_pool(self, pool):
        if not pool:
            return None, None

        for candidate in sorted(pool, key=lambda value: value.get("priority", 0.0), reverse=True)[:self.verify_after_scan_top_k]:
            accepted, confidence, reason = self.verify_candidate(candidate)
            if accepted:
                det = candidate["det"]
                self.owner_track_center = float(det.center_x) / float(candidate["image_width"])
                rospy.loginfo(
                    "Owner accepted at bbox=(%d,%d,%d,%d), confidence=%.2f reason=%s",
                    det.xmin,
                    det.ymin,
                    det.xmax,
                    det.ymax,
                    confidence,
                    reason,
                )
                return candidate, candidate.get("yaw")
        return None, None

    def scan_for_owner_by_time(self):
        deadline = time.time() + self.scan_duration
        last_verify_time = 0.0
        last_collect_time = 0.0
        owner_candidate = None
        owner_yaw = None
        scan_candidate_pool = []
        rate = rospy.Rate(10)

        rospy.logwarn("No odom yaw received on %s; falling back to timed scan", self.odom_topic)
        while not rospy.is_shutdown() and time.time() < deadline:
            twist = Twist()
            twist.angular.z = self.scan_angular_speed
            self.cmd_pub.publish(twist)

            now = time.time()
            if now - last_collect_time >= self.candidate_collection_interval:
                candidates = self.snapshot_candidates()
                if candidates:
                    scan_candidate_pool = self.merge_scan_candidates(scan_candidate_pool, candidates, self.get_latest_yaw())
                last_collect_time = now

            if self.verify_during_scan and owner_candidate is None and now - last_verify_time >= self.candidate_cooldown:
                candidates = self.snapshot_candidates()
                if candidates:
                    self.stop_base()
                    rospy.sleep(0.2)
                    current_pool = self.merge_scan_candidates([], candidates, self.get_latest_yaw())
                    owner_candidate, owner_yaw = self.verify_candidate_pool(current_pool)
                    last_verify_time = time.time()
                    if owner_candidate is not None:
                        return owner_candidate
            rate.sleep()

        self.stop_base()
        if owner_candidate is None:
            owner_candidate, owner_yaw = self.verify_candidate_pool(scan_candidate_pool)
        if owner_candidate is not None and owner_yaw is not None and self.scan_return_to_owner:
            self.rotate_to_yaw(owner_yaw)
        return owner_candidate

    def scan_for_owner(self):
        rospy.sleep(self.scan_after_arrival_delay)
        start_yaw = self.wait_for_odom_yaw()
        if start_yaw is None:
            return self.scan_for_owner_by_time()

        deadline = time.time() + self.scan_timeout
        last_verify_time = 0.0
        last_yaw = start_yaw
        rotated_angle = 0.0
        owner_candidate = None
        owner_yaw = None
        last_collect_time = 0.0
        scan_candidate_pool = []
        rate = rospy.Rate(10)

        rospy.loginfo(
            "Scanning %.2f rad at %.2f rad/s using odom yaw from %s",
            self.scan_total_angle,
            self.scan_angular_speed,
            self.odom_topic,
        )
        while not rospy.is_shutdown() and rotated_angle < self.scan_total_angle and time.time() < deadline:
            twist = Twist()
            twist.angular.z = self.scan_angular_speed
            self.cmd_pub.publish(twist)

            current_yaw = self.get_latest_yaw()
            if current_yaw is not None:
                delta = signed_angle_diff(current_yaw, last_yaw)
                if abs(delta) < 0.7:
                    rotated_angle += abs(delta)
                last_yaw = current_yaw

            now = time.time()
            rospy.loginfo_throttle(
                3.0,
                "Owner scan rotated %.0f / %.0f deg",
                math.degrees(rotated_angle),
                math.degrees(self.scan_total_angle),
            )
            if now - last_collect_time >= self.candidate_collection_interval:
                candidates = self.snapshot_candidates()
                if candidates:
                    scan_candidate_pool = self.merge_scan_candidates(scan_candidate_pool, candidates, current_yaw)
                last_collect_time = now

            if self.verify_during_scan and owner_candidate is None and now - last_verify_time >= self.candidate_cooldown:
                candidates = self.snapshot_candidates()
                if candidates:
                    self.stop_base()
                    rospy.sleep(0.2)
                    current_pool = self.merge_scan_candidates([], candidates, current_yaw)
                    owner_candidate, owner_yaw = self.verify_candidate_pool(current_pool)
                    last_verify_time = time.time()
                    if owner_candidate is not None:
                        return owner_candidate
            rate.sleep()

        self.stop_base()
        if rotated_angle < self.scan_total_angle:
            rospy.logwarn(
                "Owner scan stopped before full angle: %.0f / %.0f deg",
                math.degrees(rotated_angle),
                math.degrees(self.scan_total_angle),
            )

        if owner_candidate is None:
            rospy.loginfo(
                "Scan collected %d candidate views; verifying top %d",
                len(scan_candidate_pool),
                self.verify_after_scan_top_k,
            )
            owner_candidate, owner_yaw = self.verify_candidate_pool(scan_candidate_pool)

        if owner_candidate is not None and owner_yaw is not None and self.scan_return_to_owner:
            rospy.loginfo("Scan complete; rotating back to owner heading")
            self.rotate_to_yaw(owner_yaw)
        return owner_candidate

    def select_tracking_detection(self):
        with self.lock:
            if self.latest_image is None or not self.latest_detections:
                return None, None, None
            height, width = self.latest_image.shape[:2]
            detections = list(self.latest_detections)

        best = None
        best_score = -1.0
        for det in detections:
            class_name = getattr(det, "class_name", "")
            if class_name and class_name != "person":
                continue
            if float(det.score) < self.detection_min_score:
                continue
            center_norm = float(det.center_x) / float(width)
            area_ratio = max(0.0, float((det.xmax - det.xmin) * (det.ymax - det.ymin)) / float(width * height))
            if self.owner_track_center is None:
                tracking_penalty = abs(center_norm - 0.5)
            else:
                tracking_penalty = abs(center_norm - self.owner_track_center)
            score = float(det.score) + area_ratio * 5.0 - tracking_penalty * 1.4
            if score > best_score:
                best = det
                best_score = score
        return best, width, height

    def center_owner_in_camera(self, owner_candidate=None):
        if not self.center_owner_enabled:
            return True

        if owner_candidate is not None and "det" in owner_candidate:
            det = owner_candidate["det"]
            image_width = float(owner_candidate.get("image_width", 0.0))
            if image_width > 0:
                self.owner_track_center = float(det.center_x) / image_width

        rospy.loginfo("Centering verified owner in real robot camera")
        deadline = time.time() + max(0.5, self.center_owner_timeout)
        centered_count = 0
        rate = rospy.Rate(12)
        while not rospy.is_shutdown() and time.time() < deadline:
            det, width, _height = self.select_tracking_detection()
            twist = Twist()

            if det is None or width is None or width <= 0:
                last_center = self.owner_track_center if self.owner_track_center is not None else 0.5
                direction = 1.0 if last_center > 0.5 else -1.0
                twist.angular.z = -direction * self.center_owner_lost_turn_speed
                self.cmd_pub.publish(twist)
                centered_count = 0
                rate.sleep()
                continue

            center_norm = float(det.center_x) / float(width)
            self.owner_track_center = center_norm
            x_error = center_norm - 0.5
            if abs(x_error) <= self.center_owner_tolerance:
                self.stop_base()
                centered_count += 1
                if centered_count >= 3:
                    rospy.loginfo("Owner centered in camera: center=%.3f", center_norm)
                    return True
                rate.sleep()
                continue

            centered_count = 0
            twist.angular.z = clamp(
                -self.center_owner_angular_gain * x_error,
                -self.center_owner_max_angular_speed,
                self.center_owner_max_angular_speed,
            )
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.stop_base()
        rospy.logwarn("Owner centering timed out")
        return False

    @staticmethod
    def owner_detection_image_context(det, image_width, image_height):
        width = max(1.0, float(image_width))
        height = max(1.0, float(image_height))
        xmin = float(det.xmin)
        xmax = float(det.xmax)
        ymin = float(det.ymin)
        ymax = float(det.ymax)
        box_w = max(1.0, xmax - xmin)
        box_h = max(1.0, ymax - ymin)
        center_y = float(getattr(det, "center_y", (ymin + ymax) * 0.5))
        return {
            "bbox_bottom_norm": clamp(ymax / height, 0.0, 1.5),
            "bbox_center_y_norm": clamp(center_y / height, 0.0, 1.5),
            "bbox_aspect": box_w / box_h,
            "bbox_height_norm": box_h / height,
            "bbox_center_x_norm": clamp(float(det.center_x) / width, 0.0, 1.5),
        }

    def capture_fallen_owner_position(self, owner_candidate=None):
        deadline = time.time() + max(0.1, self.fall_approach_position_sample_seconds)
        positions = []
        fallback_used = False
        image_context = None
        rate = rospy.Rate(10)

        while not rospy.is_shutdown() and time.time() < deadline:
            det, width, height = self.select_tracking_detection()
            if (det is None or width is None or height is None) and owner_candidate is not None and not positions:
                det = owner_candidate.get("det")
                width = owner_candidate.get("image_width")
                height = owner_candidate.get("image_height")
                fallback_used = det is not None

            if det is None or width is None or height is None or width <= 0 or height <= 0:
                self.last_approach_failure_reason = "no owner detection before fallen-owner approach"
                rate.sleep()
                continue

            self.owner_track_center = float(det.center_x) / float(width)
            image_context = self.owner_detection_image_context(det, width, height)
            position = self.estimate_owner_position_from_pointcloud(det, width, height)
            if position is not None:
                positions.append(position)
                if len(positions) >= self.fall_approach_min_position_samples:
                    break
            else:
                self.last_approach_failure_reason = self.last_pointcloud_reason or "no valid fallen-owner point cloud"
            rate.sleep()

        if not positions:
            if not self.last_approach_failure_reason:
                self.last_approach_failure_reason = "no valid fallen-owner position before blind approach"
            return None

        xs = np.asarray([position["x"] for position in positions], dtype=np.float32)
        ys = np.asarray([position["y"] for position in positions], dtype=np.float32)
        surface_heights = np.asarray(
            [position["surface_height_median"] for position in positions if "surface_height_median" in position],
            dtype=np.float32,
        )
        x = float(np.median(xs))
        y = float(np.median(ys))
        distance = math.hypot(x, y)
        bearing = math.atan2(y, max(0.05, x))
        latest = positions[-1]
        captured = {
            "x": x,
            "y": y,
            "distance": distance,
            "bearing": bearing,
            "surface_height_median": float(np.median(surface_heights)) if len(surface_heights) > 0 else None,
            "surface_height_p20": latest.get("surface_height_p20"),
            "surface_height_p80": latest.get("surface_height_p80"),
            "mode": latest.get("mode", "unknown"),
            "samples": int(sum(position.get("samples", 0) for position in positions)),
            "position_samples": len(positions),
            "frame": latest.get("frame", ""),
            "fallback_detection": fallback_used,
        }

        if image_context:
            captured.update(image_context)
        return captured

    def classify_static_lying_surface(self, owner_candidate=None):
        if not self.lying_surface_classification_enabled:
            return "lying", "surface classification disabled", None

        position = self.capture_fallen_owner_position(owner_candidate)
        if position is None:
            return "lying", self.last_approach_failure_reason or "no point-cloud surface height", None

        surface_height = position.get("surface_height_median")
        bottom_norm = position.get("bbox_bottom_norm")
        center_y_norm = position.get("bbox_center_y_norm")
        ground_height_limit = self.lying_ground_max_surface_height
        furniture_height_limit = self.lying_furniture_min_surface_height

        if surface_height is not None and math.isfinite(float(surface_height)):
            if surface_height <= ground_height_limit:
                return (
                    "lying_ground",
                    "point-cloud surface height %.2fm <= ground limit %.2fm" % (surface_height, ground_height_limit),
                    position,
                )
            if surface_height >= furniture_height_limit:
                return (
                    "lying",
                    "point-cloud surface height %.2fm >= furniture limit %.2fm" % (surface_height, furniture_height_limit),
                    position,
                )

        image_ground_hint = (
            bottom_norm is not None
            and center_y_norm is not None
            and float(bottom_norm) >= self.lying_ground_bbox_bottom_ratio
            and float(center_y_norm) >= self.lying_ground_bbox_center_ratio
        )
        if image_ground_hint:
            return "lying_ground", "ambiguous height with low image bbox", position
        return "lying", "ambiguous/elevated lying posture", position

    def approach_fallen_owner(self, owner_candidate=None, initial_position=None):
        if not self.fall_approach_enabled:
            rospy.loginfo("Owner blind approach is disabled")
            return True

        position = initial_position or self.capture_fallen_owner_position(owner_candidate)
        if position is None:
            rospy.logwarn("Owner blind approach cannot start: %s", self.last_approach_failure_reason)
            return False

        standoff_distance = clamp(abs(self.fall_approach_standoff_distance), 0.45, 1.8)
        distance_tolerance = max(0.02, self.fall_approach_distance_tolerance)
        fast_finish_tolerance = clamp(
            abs(self.fall_approach_fast_finish_tolerance),
            distance_tolerance,
            0.30,
        )
        max_travel = max(0.0, self.fall_approach_max_travel_distance)
        travel_distance = clamp(position["distance"] - standoff_distance, 0.0, max_travel)
        lidar_guard_distance = max(0.30, self.fall_approach_lidar_stop_distance + self.fall_approach_lidar_margin)
        self.last_approach_failure_reason = "owner blind approach did not start"

        rospy.loginfo(
            "Owner blind approach snapshot: mode=%s distance=%.2f bearing=%.3f travel=%.2f standoff=%.2f lidar_guard=%.2f samples=%d/%d fallback_det=%s",
            position["mode"],
            position["distance"],
            position["bearing"],
            travel_distance,
            standoff_distance,
            lidar_guard_distance,
            position["position_samples"],
            position["samples"],
            position["fallback_detection"],
        )

        if travel_distance <= distance_tolerance:
            self.stop_base()
            rospy.loginfo("Owner is already within standoff distance")
            return True

        if travel_distance <= fast_finish_tolerance:
            self.stop_base()
            rospy.loginfo(
                "Owner blind approach near target; accepting without slow finish: travel=%.2f tolerance=%.2f",
                travel_distance,
                fast_finish_tolerance,
            )
            return True

        if self.approach_navigation_enabled:
            if self.navigate_to_owner_standoff(
                position,
                standoff_distance,
                max_travel=max_travel,
                timeout=self.fall_approach_drive_timeout,
                label="owner blind approach",
                lidar_guard_distance=lidar_guard_distance,
            ):
                return True
            if not self.approach_direct_fallback_enabled:
                rospy.logwarn(
                    "Owner blind approach stopped after move_base failure; direct cmd_vel fallback is disabled: %s",
                    self.last_approach_failure_reason,
                )
                return False
            rospy.logwarn(
                "Owner blind approach falling back to direct cmd_vel after move_base failure: %s",
                self.last_approach_failure_reason,
            )

        current_yaw = self.get_latest_yaw()
        if current_yaw is not None and abs(position["bearing"]) > self.approach_bearing_tolerance:
            target_yaw = current_yaw + position["bearing"]
            if not self.rotate_to_yaw(target_yaw, timeout=max(1.0, self.fall_approach_turn_timeout)):
                self.last_approach_failure_reason = "failed to align to owner before blind drive"
                rospy.logwarn("Owner blind approach aborted: %s", self.last_approach_failure_reason)
                return False
        elif current_yaw is None and abs(position["bearing"]) > self.approach_forward_bearing_limit:
            self.last_approach_failure_reason = "no odom yaw for large owner bearing: %.3f" % position["bearing"]
            rospy.logwarn("Owner blind approach aborted: %s", self.last_approach_failure_reason)
            return False

        start_xy = self.wait_for_odom_xy(timeout=1.0)
        start_time = time.time()
        last_loop_time = start_time
        timed_travelled = 0.0
        deadline = start_time + max(1.0, self.fall_approach_drive_timeout)
        rate = rospy.Rate(10)
        used_timed_fallback = start_xy is None
        slow_finish_cycles = 0
        if used_timed_fallback:
            rospy.logwarn("No odom position available; using timed owner blind drive")

        while not rospy.is_shutdown() and time.time() < deadline:
            front_distance = self.front_scan_distance()
            if front_distance is not None and front_distance <= lidar_guard_distance:
                self.stop_base()
                rospy.loginfo(
                    "Owner blind approach stopped at lidar guard: lidar=%.2f guard=%.2f",
                    front_distance,
                    lidar_guard_distance,
                )
                return True

            if start_xy is not None:
                travelled = self.xy_distance(start_xy, self.get_latest_odom_xy())
            else:
                travelled = timed_travelled
            if travelled is None:
                travelled = 0.0

            remaining = travel_distance - travelled
            if remaining <= fast_finish_tolerance:
                self.stop_base()
                rospy.loginfo(
                    "Owner blind approach complete: travelled=%.2f target=%.2f remaining=%.2f tolerance=%.2f timed_fallback=%s",
                    travelled,
                    travel_distance,
                    remaining,
                    fast_finish_tolerance,
                    used_timed_fallback,
                )
                return True

            speed = clamp(
                self.fall_approach_linear_gain * remaining,
                self.fall_approach_min_linear_speed,
                self.fall_approach_linear_speed,
            )
            if front_distance is not None:
                clearance = front_distance - lidar_guard_distance
                speed = min(speed, max(0.0, clearance) * 0.35)

            twist = Twist()
            twist.linear.x = max(0.0, speed)
            slow_finish_cycles, actual_speed, slow_reason = self.update_approach_slow_finish_cycles(
                remaining,
                twist.linear.x,
                slow_finish_cycles,
            )
            if slow_finish_cycles >= self.approach_slow_finish_cycles:
                self.stop_base()
                rospy.loginfo(
                    "Owner blind approach accepted slow/stop near target: travelled=%.2f target=%.2f remaining=%.2f cmd=%.3f odom_speed=%.3f reason=%s cycles=%d",
                    travelled,
                    travel_distance,
                    remaining,
                    twist.linear.x,
                    actual_speed if actual_speed is not None else -1.0,
                    slow_reason,
                    slow_finish_cycles,
                )
                return True
            self.cmd_pub.publish(twist)
            now = time.time()
            if start_xy is None:
                timed_travelled += twist.linear.x * max(0.0, now - last_loop_time)
            last_loop_time = now
            rospy.loginfo_throttle(
                1.0,
                "Owner blind approach: travelled=%.2f target=%.2f remaining=%.2f cmd=%.2f lidar=%.2f",
                travelled,
                travel_distance,
                remaining,
                twist.linear.x,
                front_distance if front_distance is not None else -1.0,
            )
            rate.sleep()

        self.stop_base()
        travelled = self.xy_distance(start_xy, self.get_latest_odom_xy()) if start_xy is not None else None
        if travelled is not None and travelled >= travel_distance - max(0.20, distance_tolerance):
            rospy.logwarn(
                "Owner blind approach timed out near target; accepting stop: travelled=%.2f target=%.2f",
                travelled,
                travel_distance,
            )
            return True
        self.last_approach_failure_reason = "owner blind drive timed out before target: travelled=%s target=%.2f" % (
            "%.2f" % travelled if travelled is not None else "unknown",
            travel_distance,
        )
        rospy.logwarn("Owner blind approach timed out: %s", self.last_approach_failure_reason)
        return False

    def advance_forward_by_distance(self, extra_distance, speed=None, timeout=None, lidar_guard_distance=None, label="owner"):
        target_distance = max(0.0, float(extra_distance))
        if target_distance <= 0.0:
            return True

        move_speed = abs(float(speed)) if speed is not None else self.fall_approach_extra_close_speed
        move_speed = max(0.03, move_speed)
        move_timeout = float(timeout) if timeout is not None else self.fall_approach_extra_close_timeout
        move_timeout = max(1.0, move_timeout)
        finish_tolerance = clamp(
            abs(self.fall_approach_extra_close_finish_tolerance),
            0.02,
            max(0.02, min(0.12, target_distance)),
        )
        guard_distance = lidar_guard_distance
        if guard_distance is None:
            guard_distance = max(0.30, self.fall_approach_lidar_stop_distance + self.fall_approach_lidar_margin)

        if self.approach_navigation_enabled:
            nav_label = "%s extra-close advance" % label
            if self.navigate_relative_for_approach(
                target_distance,
                0.0,
                yaw=0.0,
                timeout=move_timeout,
                label=nav_label,
                lidar_guard_distance=guard_distance,
            ):
                return True
            if not self.approach_direct_fallback_enabled:
                rospy.logwarn(
                    "%s stopped after move_base failure; direct cmd_vel fallback is disabled: %s",
                    nav_label,
                    self.last_approach_failure_reason,
                )
                return False
            rospy.logwarn(
                "%s falling back to direct cmd_vel after move_base failure: %s",
                nav_label,
                self.last_approach_failure_reason,
            )

        start_xy = self.wait_for_odom_xy(timeout=1.0)
        start_time = time.time()
        last_loop_time = start_time
        timed_travelled = 0.0
        deadline = start_time + move_timeout
        rate = rospy.Rate(10)
        used_timed_fallback = start_xy is None
        slow_finish_cycles = 0

        rospy.loginfo(
            "%s extra-close advance: target=%.2fm speed=%.2fm/s finish_tol=%.2fm guard=%.2fm timed_fallback=%s",
            label,
            target_distance,
            move_speed,
            finish_tolerance,
            guard_distance,
            used_timed_fallback,
        )

        while not rospy.is_shutdown() and time.time() < deadline:
            front_distance = self.front_scan_distance()
            if front_distance is not None and front_distance <= guard_distance:
                self.stop_base()
                rospy.loginfo(
                    "%s extra-close advance stopped by lidar guard: lidar=%.2f guard=%.2f",
                    label,
                    front_distance,
                    guard_distance,
                )
                return True

            if start_xy is not None:
                travelled = self.xy_distance(start_xy, self.get_latest_odom_xy())
            else:
                travelled = timed_travelled
            if travelled is None:
                travelled = 0.0

            remaining = target_distance - travelled
            if remaining <= finish_tolerance:
                self.stop_base()
                rospy.loginfo(
                    "%s extra-close advance complete: travelled=%.2f target=%.2f remaining=%.2f tolerance=%.2f",
                    label,
                    travelled,
                    target_distance,
                    remaining,
                    finish_tolerance,
                )
                return True

            twist = Twist()
            twist.linear.x = move_speed
            if front_distance is not None:
                clearance = front_distance - guard_distance
                twist.linear.x = min(twist.linear.x, max(0.0, clearance) * 0.35)
            slow_finish_cycles, actual_speed, slow_reason = self.update_approach_slow_finish_cycles(
                remaining,
                twist.linear.x,
                slow_finish_cycles,
            )
            if slow_finish_cycles >= self.approach_slow_finish_cycles:
                self.stop_base()
                rospy.loginfo(
                    "%s extra-close advance accepted slow/stop near target: travelled=%.2f target=%.2f remaining=%.2f cmd=%.3f odom_speed=%.3f reason=%s cycles=%d",
                    label,
                    travelled,
                    target_distance,
                    remaining,
                    twist.linear.x,
                    actual_speed if actual_speed is not None else -1.0,
                    slow_reason,
                    slow_finish_cycles,
                )
                return True
            self.cmd_pub.publish(twist)
            now = time.time()
            if start_xy is None:
                timed_travelled += twist.linear.x * max(0.0, now - last_loop_time)
            last_loop_time = now
            rospy.loginfo_throttle(
                1.0,
                "%s extra-close advance: travelled=%.2f target=%.2f remaining=%.2f cmd=%.2f lidar=%.2f",
                label,
                travelled,
                target_distance,
                remaining,
                twist.linear.x,
                front_distance if front_distance is not None else -1.0,
            )
            rate.sleep()

        self.stop_base()
        travelled = self.xy_distance(start_xy, self.get_latest_odom_xy()) if start_xy is not None else None
        if travelled is not None and travelled >= target_distance - 0.10:
            rospy.logwarn(
                "%s extra-close advance timed out near target; accepting stop: travelled=%.2f target=%.2f",
                label,
                travelled,
                target_distance,
            )
            return True
        self.last_approach_failure_reason = "%s extra-close advance timed out before target: travelled=%s target=%.2f" % (
            label,
            "%.2f" % travelled if travelled is not None else "unknown",
            target_distance,
        )
        rospy.logwarn(self.last_approach_failure_reason)
        return False

    def approach_waving_owner(self, owner_candidate=None):
        if not self.approach_on_waving_enabled:
            rospy.loginfo("Approach after waving is disabled")
            return True

        if owner_candidate is not None and "det" in owner_candidate:
            det = owner_candidate["det"]
            image_width = float(owner_candidate.get("image_width", 0.0))
            if image_width > 0:
                self.owner_track_center = float(det.center_x) / image_width

        target_distance = self.waving_approach_safety_radius
        lidar_guard_distance = max(0.30, self.approach_lidar_stop_distance + self.approach_lidar_margin)
        rospy.loginfo(
            "Approaching waving owner with move_base: safety_radius=%.2fm still_duration=%.1fs topic=%s",
            target_distance,
            self.waving_approach_still_duration,
            self.points_topic,
        )

        deadline = time.time() + max(1.0, self.approach_timeout)
        rate = rospy.Rate(10)
        position = None
        fallback_used = False
        self.last_approach_failure_reason = "no valid owner position before move_base approach"

        while not rospy.is_shutdown() and time.time() < deadline:
            det, width, height = self.select_tracking_detection()
            if (det is None or width is None or height is None) and owner_candidate is not None and not fallback_used:
                det = owner_candidate.get("det")
                width = owner_candidate.get("image_width")
                height = owner_candidate.get("image_height")
                fallback_used = det is not None

            if det is None or width is None or height is None or width <= 0 or height <= 0:
                self.last_approach_failure_reason = "no owner detection before waving-owner move_base approach"
                rate.sleep()
                continue

            self.owner_track_center = float(det.center_x) / float(width)
            position = self.estimate_owner_position_from_pointcloud(det, width, height)
            if position is not None:
                break

            self.last_approach_failure_reason = self.last_pointcloud_reason or "no valid owner point cloud position"
            rate.sleep()

        if position is None:
            self.stop_base()
            rospy.logwarn("Waving owner move_base approach cannot start: %s", self.last_approach_failure_reason)
            return False

        distance = position["distance"]
        bearing = position["bearing"]
        if distance <= self.waving_approach_safety_radius:
            self.stop_base()
            rospy.loginfo(
                "Waving owner already within safety circle: mode=%s distance=%.2f safety=%.2f bearing=%.3f samples=%d",
                position["mode"],
                distance,
                self.waving_approach_safety_radius,
                bearing,
                position["samples"],
            )
            return True

        if not self.approach_navigation_enabled:
            self.stop_base()
            self.last_approach_failure_reason = "move_base owner approach is disabled"
            rospy.logwarn("Waving owner approach aborted: %s", self.last_approach_failure_reason)
            return False

        nav_lidar_guard = lidar_guard_distance if self.approach_navigation_lidar_guard_enabled else None
        return self.navigate_to_waving_owner_candidates(
            position,
            target_distance,
            timeout=self.approach_navigation_timeout,
            label="waving owner move_base approach",
            lidar_guard_distance=nav_lidar_guard,
        )

    def run_owner_task_at_waypoint(self, waypoint_name):
        self.waypoint_name = str(waypoint_name).strip()
        waypoint_label = self.waypoint_display_name(self.waypoint_name)
        self.owner_track_center = None
        self.last_pointcloud_reason = ""
        self.last_approach_failure_reason = ""

        rospy.loginfo("Starting owner-search task at waypoint %s", self.waypoint_name)
        self.navigate_to_waypoint(self.waypoint_name)

        if self.speak_on_arrival:
            self.say("我已到达%s，开始寻找主人。" % waypoint_label)

        owner_candidate = self.scan_for_owner()
        if owner_candidate is None:
            rospy.loginfo("No owner confirmed at waypoint %s; continuing to next waypoint", self.waypoint_name)
            self.say("我在%s没有确认主人，继续前往下一个航点。" % waypoint_label)
            return False

        if self.speak_on_owner_found:
            self.say("我已经识别到主人%s。" % self.owner_name)

        centered = self.center_owner_in_camera(owner_candidate)
        if not centered:
            rospy.logwarn("Owner centering unstable; skipping centering and continuing action recognition")

        self.say("识别中。", hold=0.0)
        action_label, action_confidence, action_reason = self.recognize_owner_action(owner_candidate)
        rospy.loginfo(
            "Owner action summary at %s: label=%s confidence=%.2f reason=%s",
            self.waypoint_name,
            action_label,
            action_confidence,
            action_reason,
        )
        normalized_action = str(action_label).strip().lower()
        initial_approach_position = None
        self.say(
            self.action_to_speech(
                action_label,
                self.last_owner_action_place,
            ),
            hold=self.action_speech_hold,
        )
        if normalized_action in self.fall_approach_action_labels:
            needs_fall_assist_arm = normalized_action in self.fall_assist_arm_action_labels
            approached = self.approach_fallen_owner(owner_candidate, initial_position=initial_approach_position)
            if approached:
                extra_close_completed = True
                if self.fall_approach_extra_close_enabled:
                    extra_close_completed = self.advance_forward_by_distance(
                        self.fall_approach_extra_close_distance,
                        speed=self.fall_approach_extra_close_speed,
                        timeout=self.fall_approach_extra_close_timeout,
                        label="owner",
                    )
                    if not extra_close_completed:
                        rospy.logwarn(
                            "Owner extra-close advance did not complete for action=%s: %s",
                            normalized_action,
                            self.last_approach_failure_reason,
                        )
            else:
                rospy.logwarn("Owner blind approach did not complete for action=%s: %s", normalized_action, self.last_approach_failure_reason)
                self.stop_base()
            if needs_fall_assist_arm:
                self.stop_base()
                rospy.loginfo(
                    "Completing fall assist arm motion before leaving waypoint %s",
                    self.waypoint_name,
                )
                self.perform_fall_assist_arm_motion()
            elif normalized_action in ("lying", "sitting"):
                if not approached:
                    rospy.logwarn(
                        "Proceeding to voice interaction after incomplete %s approach",
                        normalized_action,
                    )
                self.wait_for_electrical_switch_instruction()
        elif normalized_action == "waving":
            approached = self.approach_waving_owner(owner_candidate)
            if approached:
                self.say(self.approach_help_prompt, hold=self.waving_help_pause_seconds)
            else:
                reason = self.last_approach_failure_reason
                rospy.logwarn("Waving owner move_base approach did not complete: %s", reason)
                if "lidar guard" in reason or "blocked by close obstacle" in reason:
                    self.say("我前方有障碍，暂时无法靠近您。")
                elif "point cloud" in reason or "ROI" in reason:
                    self.say("我看到您在挥手，但点云定位还不稳定，暂时无法靠近您。")
                else:
                    self.say("我看到您在挥手，但还没有成功靠近您。")
        return True

    def run(self):
        self.set_yolo_paused(False)
        self.rename_exit_waypoint_alias_if_needed()
        self.wait_for_hardware_inputs()
        self.enroll_owner()
        self.init_action_recognizer()
        self.ensure_qwen_action_warmup()

        if self.speak_on_start:
            waypoint_labels = "、".join(self.waypoint_display_name(name) for name in self.task_waypoint_names)
            self.say("我开始依次前往%s寻找主人。" % waypoint_labels)

        completed_waypoints = 0
        owner_found_count = 0
        for waypoint_name in self.task_waypoint_names:
            if rospy.is_shutdown():
                break
            found_owner = self.run_owner_task_at_waypoint(waypoint_name)
            completed_waypoints += 1
            if found_owner:
                owner_found_count += 1

        if self.return_to_exit_when_complete and not rospy.is_shutdown():
            self.navigate_to_waypoint(self.exit_waypoint_name)
            if self.speak_on_finish:
                self.say("所有航点任务已完成，我已到达出口。")
        elif self.speak_on_finish and not rospy.is_shutdown():
            self.say("所有航点任务已完成。")

        rospy.loginfo(
            "task1 owner-search route finished: completed_waypoints=%d owner_found=%d exit=%s",
            completed_waypoints,
            owner_found_count,
            self.exit_waypoint_name if self.return_to_exit_when_complete else "disabled",
        )
        return completed_waypoints == len(self.task_waypoint_names)


def main():
    rospy.init_node("task1_find_owner_real")
    node = None
    try:
        node = RealOwnerSearchBeforeAction()
        success = node.run()
        rospy.loginfo("task1_find_owner_real finished: success=%s", success)
    except MissingHardwareError as exc:
        rospy.logfatal("task1_find_owner_real failed: %s", exc)
        if node is not None:
            node.stop_base()
            try:
                node.say("没有收到机器人传感器数据，请检查Kinect、雷达和里程计。", hold=2.5)
            except Exception as say_exc:
                rospy.logwarn("Unable to speak hardware failure message: %s", say_exc)
    except Exception as exc:
        rospy.logfatal("task1_find_owner_real failed: %s", exc)
        if node is not None:
            node.stop_base()
        raise
    finally:
        if node is not None:
            node.stop_base()


if __name__ == "__main__":
    main()
