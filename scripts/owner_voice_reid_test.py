#!/usr/bin/env python3
# coding=utf-8

import json
import math
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import actionlib
import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetPlan
from sensor_msgs.msg import JointState, LaserScan, PointCloud2
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty

try:
    import tf
except ImportError:
    tf = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from person_reid_owner_test import PersonReidOwnerTest

DEFAULT_ACTION_DIAGNOSTICS_LOG_PATH = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "logs",
        "optimized_action_diagnostics.jsonl",
    )
)


class OwnerVoiceReidTest(PersonReidOwnerTest):
    def __init__(self):
        super().__init__()

        self.asr_topic = rospy.get_param("~asr_topic", "/voice/asr_text")
        self.answer_timeout = max(1.0, float(rospy.get_param("~answer_timeout", 30.0)))
        self.asr_wait_for_publishers = bool(rospy.get_param("~asr_wait_for_publishers", True))
        self.asr_wait_timeout = max(0.0, float(rospy.get_param("~asr_wait_timeout", 45.0)))
        self.asr_settle_seconds = max(
            0.0,
            float(rospy.get_param("~asr_settle_seconds", 0.5)),
        )
        self.help_result_topic = str(
            rospy.get_param(
                "~help_result_topic",
                "/owner_voice_reid_test/help_result",
            )
        ).strip()
        self.help_llm_url = str(
            rospy.get_param(
                "~help_llm_url",
                "http://127.0.0.1:11434/api/chat",
            )
        ).strip()
        self.help_llm_model = str(
            rospy.get_param("~help_llm_model", "qwen3.5:0.8b")
        ).strip() or "qwen3.5:0.8b"
        self.help_llm_timeout = max(
            1.0,
            float(rospy.get_param("~help_llm_timeout", 60.0)),
        )
        self.help_llm_keep_alive = str(
            rospy.get_param("~help_llm_keep_alive", "30m")
        ).strip()
        self.help_llm_max_tokens = max(
            16,
            int(rospy.get_param("~help_llm_max_tokens", 80)),
        )
        self.help_llm_num_ctx = max(
            512,
            int(rospy.get_param("~help_llm_num_ctx", 4096)),
        )
        self.help_llm_num_gpu = max(
            0,
            int(rospy.get_param("~help_llm_num_gpu", 0)),
        )
        self.help_llm_warmup_enabled = bool(
            rospy.get_param("~help_llm_warmup_enabled", True)
        )
        self.help_llm_warmup_retries = max(
            1,
            int(rospy.get_param("~help_llm_warmup_retries", 2)),
        )
        self.help_llm_warmup_retry_delay = max(
            0.0,
            float(rospy.get_param("~help_llm_warmup_retry_delay", 2.0)),
        )
        self.strict_rephrase = bool(
            rospy.get_param("~strict_rephrase", False)
        )
        self.max_name_retries = max(
            1,
            int(rospy.get_param("~max_name_retries", rospy.get_param("~max_retries", 3))),
        )
        self.max_name_length = max(1, int(rospy.get_param("~max_name_length", 12)))
        self.owner_count = max(1, int(rospy.get_param("~owner_count", 3)))
        self.reuse_owner_index = max(
            1,
            int(rospy.get_param("~reuse_owner_index", 1)),
        )
        self.owner_profiles = []
        self.skipped_owner_indices = []
        self.current_owner_index = 1
        self.last_match_owner_index = None
        self.base_profile_path = self.profile_path
        self.base_metadata_path = self.metadata_path

        self.face_verify_enabled = bool(rospy.get_param("~face_verify_enabled", True))
        self.face_auto_download = bool(rospy.get_param("~face_auto_download", False))
        self.face_model_name = rospy.get_param("~face_model_name", "buffalo_sc")
        self.face_model_root = os.path.expanduser(rospy.get_param("~face_model_root", "~/.insightface"))
        self.face_ctx_id = int(rospy.get_param("~face_ctx_id", -1))
        self.face_det_size = int(rospy.get_param("~face_det_size", 480))
        self.face_det_thresh = float(rospy.get_param("~face_det_thresh", 0.35))
        self.face_accept_threshold = float(rospy.get_param("~face_accept_threshold", 0.45))
        self.face_reject_threshold = float(rospy.get_param("~face_reject_threshold", 0.25))
        self.face_fast_reject = bool(rospy.get_param("~face_fast_reject", True))
        self.face_min_reid_score = float(rospy.get_param("~face_min_reid_score", 0.45))
        self.face_identity_weight = max(
            0.0,
            min(1.0, float(rospy.get_param("~face_identity_weight", 0.85))),
        )
        self.lying_face_identity_weight = max(
            0.0,
            min(1.0, float(rospy.get_param("~lying_face_identity_weight", 0.10))),
        )
        self.owner_score_margin_threshold = max(
            0.0,
            float(rospy.get_param("~owner_score_margin_threshold", 0.04)),
        )
        self.face_crop_padding = float(rospy.get_param("~face_crop_padding", 0.20))
        self.face_crop_top_ratio = float(rospy.get_param("~face_crop_top_ratio", 0.68))

        self.face_app = None
        self.face_model_ready = False
        self.face_ready = False
        self.owner_face_embedding = None
        self.owner_face_embedding_bank = None
        self.current_record_face_embeddings = []
        self.current_record_pose_id = ""
        self.current_record_pose_name = ""

        default_action_launch_file = os.path.abspath(
            os.path.join(
                SCRIPT_DIR,
                "..",
                "..",
                "offline_voice_bridge",
                "launch",
                "qwen_action_recognition.launch",
            )
        )
        self.action_recognition_enabled = bool(
            rospy.get_param("~action_recognition_enabled", True)
        )
        self.action_launch_file = os.path.abspath(
            os.path.expanduser(
                rospy.get_param("~action_launch_file", default_action_launch_file)
            )
        )
        self.action_node_name = str(
                rospy.get_param("~action_node_name", "owner_action_pose")
        ).strip()
        self.action_result_topic = str(
            rospy.get_param(
                "~action_result_topic",
                "/owner_voice_reid_test/action_result",
            )
        ).strip()
        self.action_diagnostics_log_path = os.path.abspath(
            os.path.expanduser(
                str(
                    rospy.get_param(
                        "~action_diagnostics_log_path",
                        DEFAULT_ACTION_DIAGNOSTICS_LOG_PATH,
                    )
                )
            )
        )
        self.action_diagnostics_session_id = str(
            rospy.get_param(
                "~action_diagnostics_session_id",
                "%s-%d" % (time.strftime("%Y%m%d-%H%M%S"), os.getpid()),
            )
        ).strip()
        self.action_timeout = max(
            15.0,
            float(rospy.get_param("~action_timeout", 150.0)),
        )
        self.action_speech_grace = max(
            0.2,
            float(rospy.get_param("~action_speech_grace", 1.0)),
        )
        self.action_show_window = bool(
            rospy.get_param("~action_show_window", False)
        )
        self.action_llm_url = str(
            rospy.get_param(
                "~action_llm_url",
                "http://127.0.0.1:11434/api/chat",
            )
        ).strip()
        self.action_llm_model = str(
            rospy.get_param("~action_llm_model", "qwen3.5:0.8b")
        ).strip() or "qwen3.5:0.8b"
        self.action_pointcloud_enabled = bool(
            rospy.get_param("~action_pointcloud_enabled", False)
        )
        self.action_startup_settle_seconds = max(
            0.0,
            float(rospy.get_param("~action_startup_settle_seconds", 0.6)),
        )
        self.action_capture_delay = max(
            0.0,
            float(rospy.get_param("~action_capture_delay", 0.3)),
        )
        self.action_capture_duration = max(
            0.0,
            float(rospy.get_param("~action_capture_duration", 5.0)),
        )
        self.action_capture_frame_count = max(
            1,
            int(rospy.get_param("~action_capture_frame_count", 9)),
        )
        self.action_llm_frame_count = max(
            1,
            int(rospy.get_param("~action_llm_frame_count", 5)),
        )
        self.action_pose_enabled = bool(
            rospy.get_param("~action_pose_enabled", True)
        )
        self.action_pose_model_path = str(
            rospy.get_param("~action_pose_model_path", "")
        ).strip()
        self.action_pose_device = str(
            rospy.get_param("~action_pose_device", "cpu")
        ).strip()
        self.action_pose_image_size = max(
            160,
            int(rospy.get_param("~action_pose_image_size", 416)),
        )
        self.action_pose_confidence = float(
            rospy.get_param("~action_pose_confidence", 0.25)
        )
        self.action_pose_iou = float(rospy.get_param("~action_pose_iou", 0.45))
        self.action_pose_max_detections = max(
            1,
            int(rospy.get_param("~action_pose_max_detections", 4)),
        )
        self.action_llm_timeout = max(
            1.0,
            float(rospy.get_param("~action_llm_timeout", 90.0)),
        )
        self.action_llm_keep_alive = str(
            rospy.get_param("~action_llm_keep_alive", "30m")
        ).strip()
        self.action_llm_max_tokens = max(
            1,
            int(rospy.get_param("~action_llm_max_tokens", 32)),
        )
        self.action_llm_num_ctx = max(
            512,
            int(rospy.get_param("~action_llm_num_ctx", 8192)),
        )
        self.action_jpeg_quality = max(
            1,
            min(100, int(rospy.get_param("~action_jpeg_quality", 65))),
        )
        self.action_image_max_width = max(
            0,
            int(rospy.get_param("~action_image_max_width", 384)),
        )
        self.action_model_warmup_retries = max(
            1,
            int(rospy.get_param("~action_model_warmup_retries", 3)),
        )
        self.action_model_warmup_retry_delay = max(
            0.0,
            float(rospy.get_param("~action_model_warmup_retry_delay", 2.0)),
        )
        self.action_use_owner_roi = bool(
            rospy.get_param("~action_use_owner_roi", True)
        )
        self.action_roi_padding = max(
            0.0,
            min(1.0, float(rospy.get_param("~action_roi_padding", 0.55))),
        )
        self.action_yolo_pause_settle_seconds = max(
            0.0,
            float(rospy.get_param("~action_yolo_pause_settle_seconds", 0.8)),
        )
        self.action_result_connect_timeout = max(
            0.0,
            float(rospy.get_param("~action_result_connect_timeout", 10.0)),
        )
        self.action_pause_yolo = bool(
            rospy.get_param("~action_pause_yolo", True)
        )
        self.yolo_pause_topic = str(
            rospy.get_param("~yolo_pause_topic", "/yoloworld/pause")
        ).strip()
        self.action_launch_parent = None
        self.action_process = None
        self.action_result_sub = None
        self.action_result_event = threading.Event()
        self.action_result = None
        self.action_result_time = 0.0
        self.action_launch_lock = threading.Lock()
        self.action_completed = False

        self.interaction_enabled = bool(
            rospy.get_param("~interaction_enabled", True)
        )
        self.points_topic = str(
            rospy.get_param("~points_topic", "/kinect2/qhd/points")
        ).strip()
        self.pointcloud_nodelet_manager = str(
            rospy.get_param("~pointcloud_nodelet_manager", "/kinect2_points_manager")
        ).rstrip("/")
        self.pointcloud_nodelet_name = str(
            rospy.get_param("~pointcloud_nodelet_name", "kinect2_points_xyzrgb_qhd")
        )
        self.pointcloud_nodelet_loaded_param = "%s/%s_loaded" % (
            self.pointcloud_nodelet_manager,
            self.pointcloud_nodelet_name,
        )
        self.pointcloud_nodelet_timeout = max(
            0.1, float(rospy.get_param("~pointcloud_nodelet_timeout", 3.0))
        )
        self.pointcloud_max_age = max(
            0.1,
            float(rospy.get_param("~pointcloud_max_age", 1.0)),
        )
        self.pointcloud_stride = max(
            1,
            int(rospy.get_param("~pointcloud_stride", 8)),
        )
        self.pointcloud_min_samples = max(
            5,
            int(rospy.get_param("~pointcloud_min_samples", 30)),
        )
        self.pointcloud_roi_x_margin = self.clamp_value(
            float(rospy.get_param("~pointcloud_roi_x_margin", 0.25)),
            0.0,
            0.45,
        )
        self.pointcloud_roi_y_min_ratio = self.clamp_value(
            float(rospy.get_param("~pointcloud_roi_y_min_ratio", 0.15)),
            0.0,
            0.95,
        )
        self.pointcloud_roi_y_max_ratio = self.clamp_value(
            float(rospy.get_param("~pointcloud_roi_y_max_ratio", 0.85)),
            self.pointcloud_roi_y_min_ratio + 0.01,
            1.0,
        )
        self.pointcloud_frame_mode = str(
            rospy.get_param("~pointcloud_frame_mode", "auto")
        ).strip().lower()
        self.scan_topic = str(rospy.get_param("~scan_topic", "/scan")).strip()
        self.approach_enabled = bool(
            rospy.get_param("~approach_enabled", True)
        )
        self.approach_timeout = max(
            1.0,
            float(rospy.get_param("~approach_timeout", 25.0)),
        )
        self.approach_standoff_distance = max(
            0.35,
            float(rospy.get_param("~approach_standoff_distance", 0.75)),
        )
        self.waving_standoff_distance = max(
            0.35,
            float(rospy.get_param("~waving_standoff_distance", 0.48)),
        )
        self.approach_distance_tolerance = max(
            0.03,
            float(rospy.get_param("~approach_distance_tolerance", 0.08)),
        )
        self.approach_bearing_tolerance = max(
            0.03,
            float(rospy.get_param("~approach_bearing_tolerance", 0.10)),
        )
        self.approach_angular_gain = float(
            rospy.get_param("~approach_angular_gain", 0.55)
        )
        self.approach_max_angular_speed = abs(
            float(rospy.get_param("~approach_max_angular_speed", 0.25))
        )
        self.approach_linear_gain = float(
            rospy.get_param("~approach_linear_gain", 0.28)
        )
        self.approach_max_linear_speed = abs(
            float(rospy.get_param("~approach_max_linear_speed", 0.18))
        )
        self.approach_min_linear_speed = abs(
            float(rospy.get_param("~approach_min_linear_speed", 0.05))
        )
        self.approach_navigation_enabled = bool(
            rospy.get_param("~approach_navigation_enabled", True)
        )
        self.approach_navigation_frame = str(
            rospy.get_param("~approach_navigation_frame", "map")
        ).strip()
        self.approach_navigation_base_frame = str(
            rospy.get_param("~approach_navigation_base_frame", "base_footprint")
        ).strip()
        self.approach_navigation_tf_timeout = max(
            0.1,
            float(rospy.get_param("~approach_navigation_tf_timeout", 0.6)),
        )
        self.approach_navigation_tf_max_age = max(
            0.2,
            float(rospy.get_param("~approach_navigation_tf_max_age", 1.0)),
        )
        self.approach_navigation_timeout = max(
            1.0,
            float(rospy.get_param("~approach_navigation_timeout", 18.0)),
        )
        self.approach_navigation_server_timeout = max(
            0.5,
            float(rospy.get_param("~approach_navigation_server_timeout", 3.0)),
        )
        self.approach_navigation_retry_after_clear = bool(
            rospy.get_param("~approach_navigation_retry_after_clear", False)
        )
        self.approach_navigation_clear_costmaps_before_goal = bool(
            rospy.get_param("~approach_navigation_clear_costmaps_before_goal", False)
        )
        self.approach_navigation_min_distance = max(
            0.02,
            float(rospy.get_param("~approach_navigation_min_distance", 0.05)),
        )
        self.approach_navigation_lidar_guard_enabled = bool(
            rospy.get_param("~approach_navigation_lidar_guard_enabled", True)
        )
        self.approach_navigation_stuck_timeout = max(
            0.0,
            float(rospy.get_param("~approach_navigation_stuck_timeout", 4.0)),
        )
        self.approach_navigation_stuck_min_progress = max(
            0.01,
            float(rospy.get_param("~approach_navigation_stuck_min_progress", 0.05)),
        )
        self.approach_navigation_stuck_linear_speed = max(
            0.0,
            float(rospy.get_param("~approach_navigation_stuck_linear_speed", 0.025)),
        )
        self.approach_direct_fallback_enabled = bool(
            rospy.get_param("~approach_direct_fallback_enabled", False)
        )
        self.approach_front_scan_degrees = max(
            1.0,
            float(rospy.get_param("~approach_front_scan_degrees", 25.0)),
        )
        self.approach_scan_max_age = max(
            0.1,
            float(rospy.get_param("~approach_scan_max_age", 0.8)),
        )
        self.approach_lidar_stop_distance = max(
            0.30,
            float(rospy.get_param("~approach_lidar_stop_distance", 0.55)),
        )
        self.approach_lidar_margin = max(
            0.0,
            float(rospy.get_param("~approach_lidar_margin", 0.08)),
        )
        self.approach_slow_finish_tolerance = max(
            self.approach_navigation_min_distance,
            float(rospy.get_param("~approach_slow_finish_tolerance", 0.05)),
        )
        self.approach_slow_finish_linear_speed = max(
            0.0,
            float(rospy.get_param("~approach_slow_finish_linear_speed", 0.03)),
        )
        self.approach_odom_speed_max_age = max(
            0.1,
            float(rospy.get_param("~approach_odom_speed_max_age", 0.6)),
        )
        self.approach_max_travel_distance = max(
            0.0,
            float(rospy.get_param("~approach_max_travel_distance", 1.50)),
        )
        self.waving_approach_candidate_enabled = bool(
            rospy.get_param("~waving_approach_candidate_enabled", True)
        )
        self.waving_approach_min_owner_distance = max(
            0.35,
            float(rospy.get_param("~waving_approach_min_owner_distance", 0.45)),
        )
        self.waving_approach_max_owner_distance = max(
            self.waving_approach_min_owner_distance,
            float(rospy.get_param("~waving_approach_max_owner_distance", 0.50)),
        )
        self.waving_approach_candidate_distances = self.parse_float_list(
            rospy.get_param("~waving_approach_candidate_distances", [0.45, 0.48, 0.50])
        )
        self.waving_approach_candidate_angles_deg = self.parse_float_list(
            rospy.get_param(
                "~waving_approach_candidate_angles_deg",
                [65, -65, 95, -95, 35, -35, 0, 125, -125],
            )
        )
        self.waving_approach_plan_check = bool(
            rospy.get_param("~waving_approach_plan_check", True)
        )
        self.waving_approach_plan_service = str(
            rospy.get_param("~waving_approach_plan_service", "/move_base/make_plan")
        ).strip()
        self.waving_approach_plan_tolerance = max(
            0.05,
            float(rospy.get_param("~waving_approach_plan_tolerance", 0.20)),
        )
        self.waving_approach_safety_radius = max(
            0.40,
            float(rospy.get_param("~waving_approach_safety_radius", 0.48)),
        )
        self.waving_approach_still_duration = max(
            0.0,
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
            0.0,
            float(rospy.get_param("~waving_approach_plan_turn_limit", 4.5)),
        )
        self.waving_approach_retry_after_clear = bool(
            rospy.get_param("~waving_approach_retry_after_clear", False)
        )
        self.approach_help_prompt = rospy.get_param(
            "~approach_help_prompt", "请问您需要什么帮助？"
        )
        self.approach_failed_prompt = rospy.get_param(
            "~approach_failed_prompt", "我暂时无法靠近您。"
        )
        self.fall_prompt = rospy.get_param(
            "~fall_prompt", "我马上过去帮您。"
        )
        self.fall_approach_standoff_distance = float(
            rospy.get_param(
                "~fall_approach_standoff_distance",
                self.approach_standoff_distance,
            )
        )
        self.fall_approach_lidar_stop_distance = float(
            rospy.get_param(
                "~fall_approach_lidar_stop_distance",
                0.22,
            )
        )
        self.fall_approach_lidar_margin = float(
            rospy.get_param("~fall_approach_lidar_margin", self.approach_lidar_margin)
        )
        self.fall_assist_arm_enabled = bool(
            rospy.get_param("~fall_assist_arm_enabled", True)
        )
        self.mani_ctrl_topic = str(
            rospy.get_param("~mani_ctrl_topic", "/wpb_home/mani_ctrl")
        ).strip()
        self.fall_assist_arm_extend_lift = float(
            rospy.get_param("~fall_assist_arm_extend_lift", 0.50)
        )
        self.fall_assist_arm_extend_gripper = float(
            rospy.get_param("~fall_assist_arm_extend_gripper", 0.12)
        )
        self.fall_assist_arm_retract_lift = float(
            rospy.get_param("~fall_assist_arm_retract_lift", 0.0)
        )
        self.fall_assist_arm_retract_gripper = float(
            rospy.get_param("~fall_assist_arm_retract_gripper", 0.12)
        )
        self.fall_assist_arm_extend_wait = max(
            0.0,
            float(rospy.get_param("~fall_assist_arm_extend_wait", 3.0)),
        )
        self.fall_assist_arm_hold_seconds = max(
            0.0,
            float(rospy.get_param("~fall_assist_arm_hold_seconds", 4.0)),
        )
        self.fall_assist_arm_retract_wait = max(
            0.0,
            float(rospy.get_param("~fall_assist_arm_retract_wait", 3.0)),
        )
        self.fall_assist_arm_completion_wait = max(
            0.0,
            float(rospy.get_param("~fall_assist_arm_completion_wait", 1.0)),
        )
        self.fall_assist_arm_command_rate = max(
            0.5,
            float(rospy.get_param("~fall_assist_arm_command_rate", 5.0)),
        )
        self.fall_assist_arm_lift_velocity = max(
            0.0,
            float(rospy.get_param("~fall_assist_arm_lift_velocity", 0.5)),
        )
        self.fall_assist_arm_gripper_velocity = max(
            0.0,
            float(rospy.get_param("~fall_assist_arm_gripper_velocity", 5.0)),
        )

        self.electrical_switch_instruction_enabled = bool(
            rospy.get_param("~electrical_switch_instruction_enabled", True)
        )
        self.electrical_switch_prompt = rospy.get_param(
            "~electrical_switch_prompt", "请指示。"
        )
        self.electrical_switch_reply_on = rospy.get_param(
            "~electrical_switch_reply_on", "好的，已开启电气开关。"
        )
        self.electrical_switch_reply_off = rospy.get_param(
            "~electrical_switch_reply_off", "好的，已关闭电气开关。"
        )
        self.electrical_switch_state_topic = str(
            rospy.get_param(
                "~electrical_switch_state_topic",
                "/electrical_switch/state",
            )
        ).strip()
        self.electrical_switch_state = "unknown"
        self.latest_switch_answer = None
        self.accepting_switch_answer = False
        self.pointcloud_time = None
        self.latest_pointcloud = None
        self.pointcloud_reason = ""
        self.last_approach_failure_reason = ""
        self.latest_scan = None
        self.latest_scan_time = None
        self.latest_odom_xy = None
        self.latest_odom_linear_speed = None
        self.latest_odom_time = None
        self.switch_state_pub = None
        self.mani_ctrl_pub = None

        self.show_yolo_window = bool(rospy.get_param("~show_yolo_window", True))
        self.yolo_window_name = rospy.get_param("~yolo_window_name", "Owner YOLO-Person Box")
        self.yolo_window_ready = False
        self.yolo_window_failed = False
        self.yolo_window_retry_seconds = max(
            0.5,
            float(rospy.get_param("~yolo_window_retry_seconds", 2.0)),
        )
        self.yolo_window_last_retry = 0.0
        self.yolo_window_last_check = 0.0
        self.yolo_window_lock = threading.RLock()

        self.navigate_enabled = bool(rospy.get_param("~navigate_enabled", True))
        self.waypoint_name = str(rospy.get_param("~waypoint_name", "living_room")).strip()
        self.waypoint_file = os.path.expanduser(
            rospy.get_param("~waypoint_file", "/home/ubuntu20/waypoints.xml")
        )
        self.move_base_server_timeout = max(
            1.0,
            float(rospy.get_param("~move_base_server_timeout", 25.0)),
        )
        self.navigate_timeout = max(1.0, float(rospy.get_param("~navigate_timeout", 120.0)))
        self.clear_costmaps_before_navigation = bool(
            rospy.get_param("~clear_costmaps_before_navigation", True)
        )
        self.clear_costmaps_service = rospy.get_param(
            "~clear_costmaps_service", "/move_base/clear_costmaps"
        )
        self.clear_costmaps_timeout = max(
            0.1,
            float(rospy.get_param("~clear_costmaps_timeout", 5.0)),
        )
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.scan_angular_speed = float(rospy.get_param("~scan_angular_speed", -0.25))
        self.scan_total_angle = max(0.1, float(rospy.get_param("~scan_total_angle", math.pi)))
        self.scan_timeout = max(1.0, float(rospy.get_param("~scan_timeout", 35.0)))
        self.scan_step_angle = max(0.02, float(rospy.get_param("~scan_step_angle", 0.18)))
        self.scan_pause = max(0.0, float(rospy.get_param("~scan_pause", 0.35)))
        self.scan_after_arrival_delay = max(
            0.0,
            float(rospy.get_param("~scan_after_arrival_delay", 0.2)),
        )
        self.scan_return_to_owner = bool(rospy.get_param("~scan_return_to_owner", True))
        self.center_owner_enabled = bool(rospy.get_param("~center_owner_enabled", True))
        self.center_owner_timeout = max(0.5, float(rospy.get_param("~center_owner_timeout", 5.0)))
        self.center_owner_tolerance = max(
            0.01,
            float(rospy.get_param("~center_owner_tolerance", 0.06)),
        )
        self.center_owner_angular_gain = float(
            rospy.get_param("~center_owner_angular_gain", 0.65)
        )
        self.center_owner_max_angular_speed = abs(
            float(rospy.get_param("~center_owner_max_angular_speed", 0.30))
        )
        self.center_owner_lost_turn_speed = abs(
            float(rospy.get_param("~center_owner_lost_turn_speed", 0.10))
        )
        self.owner_not_found_text = rospy.get_param(
            "~owner_not_found_text", "没有找到主人。"
        )
        self.navigation_failed_text = rospy.get_param(
            "~navigation_failed_text", "导航到客厅失败。"
        )
        self.latest_yaw = None
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.tf_listener = tf.TransformListener() if tf is not None else None
        self.waving_make_plan = rospy.ServiceProxy(self.waving_approach_plan_service, GetPlan)

        self.name_prompt_text = rospy.get_param(
            "~name_prompt_text",
            "请按照姓名加名字的格式说出姓名",
        )
        self.indexed_name_prompt_text = rospy.get_param(
            "~indexed_name_prompt_text",
            "请第%s位主人按照姓名加名字的格式说出姓名",
        )
        self.name_confirm_prompt_text = rospy.get_param(
            "~name_confirm_prompt_text",
            "姓名%s是否正确？正确请说正确，错误请再说一遍。",
        )
        self.name_retry_text = rospy.get_param(
            "~name_retry_text",
            "请按照姓名加名字的格式回答",
        )
        self.name_timeout_text = rospy.get_param("~name_timeout_text", "没有听到主人姓名。")
        self.name_invalid_text = rospy.get_param(
            "~name_invalid_text",
            "没有识别到姓名格式，请以姓名开头再说一次。",
        )
        self.name_failed_text = rospy.get_param(
            "~name_failed_text",
            "没有记录到主人姓名，测试结束。",
        )
        self.owner_skip_text = rospy.get_param(
            "~owner_skip_text",
            "已跳过第%s位主人的记录。",
        )
        self.first_owner_skip_text = rospy.get_param(
            "~first_owner_skip_text",
            "第一位主人不能跳过，请按照姓名加名字的格式说出姓名，例如姓名张三。",
        )
        self.asr_not_ready_text = rospy.get_param(
            "~asr_not_ready_text",
            "语音识别没有连接，请检查离线语音节点。",
        )
        self.name_recorded_text = rospy.get_param(
            "~name_recorded_text",
            "已记录主人姓名，%s。",
        )
        self.face_recording_text = rospy.get_param(
            "~face_recording_text",
            rospy.get_param(
                "~front_recording_text",
                "请靠近摄像头，保持正脸，然后慢慢转向侧脸，我会连续记录多张人脸特征。",
            ),
        )
        self.face_front_to_side_record_seconds = max(
            self.record_seconds,
            float(rospy.get_param("~face_front_to_side_record_seconds", 8.0)),
        )
        self.face_front_to_side_record_sample_count = max(
            self.record_min_samples,
            int(rospy.get_param("~face_front_to_side_record_sample_count", 36)),
        )
        self.face_front_to_side_record_sample_interval = max(
            0.05,
            float(rospy.get_param("~face_front_to_side_record_sample_interval", 0.25)),
        )
        self.full_body_recording_text = rospy.get_param(
            "~full_body_recording_text",
            "现在请后退一点，确保全身都在画面中，再从正面慢慢转向侧身，我会记录更多全身特征。",
        )
        self.full_body_front_to_side_record_seconds = max(
            self.record_seconds,
            float(rospy.get_param("~full_body_front_to_side_record_seconds", 8.0)),
        )
        self.full_body_front_to_side_record_sample_count = max(
            self.record_min_samples,
            int(rospy.get_param("~full_body_front_to_side_record_sample_count", 36)),
        )
        self.full_body_front_to_side_record_sample_interval = max(
            0.05,
            float(rospy.get_param("~full_body_front_to_side_record_sample_interval", 0.25)),
        )
        self.owner_record_poses = [
            ("face_front_to_side", "近距离正脸转侧脸", self.face_recording_text),
            ("full_body_front_to_side", "远距离正面转侧身", self.full_body_recording_text),
        ]
        self.all_profiles_recorded_text = rospy.get_param(
            "~all_profiles_recorded_text",
            "主人信息已全部记录完成",
        )
        self.named_owner_found_text = rospy.get_param(
            "~named_owner_found_text",
            "识别到主人，%s。",
        )

        self.owner_name = ""
        self.latest_name_answer = None
        self.accepting_name_answer = False
        self.latest_help_answer = None
        self.accepting_help_answer = False
        self.name_condition = threading.Condition()

        self.result_pub = rospy.Publisher(
            rospy.get_param("~result_topic", "/owner_voice_reid_test/result"),
            String,
            queue_size=5,
            latch=True,
        )
        self.help_result_pub = rospy.Publisher(
            self.help_result_topic,
            String,
            queue_size=1,
            latch=True,
        )
        self.yolo_pause_pub = rospy.Publisher(
            self.yolo_pause_topic,
            Bool,
            queue_size=1,
            latch=True,
        )
        self.switch_state_pub = rospy.Publisher(
            self.electrical_switch_state_topic,
            String,
            queue_size=1,
            latch=True,
        )
        self.mani_ctrl_pub = rospy.Publisher(
            self.mani_ctrl_topic,
            JointState,
            queue_size=5,
        )
        self.asr_sub = rospy.Subscriber(self.asr_topic, String, self.asr_callback, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.scan_sub = rospy.Subscriber(
            self.scan_topic,
            LaserScan,
            self.scan_callback,
            queue_size=1,
        )
        self.pointcloud_sub = rospy.Subscriber(
            self.points_topic,
            PointCloud2,
            self.pointcloud_callback,
            queue_size=1,
        )
        rospy.on_shutdown(self.shutdown_action_recognition)
        rospy.on_shutdown(self.close_yolo_window)
        rospy.on_shutdown(self.stop_base)

    def wait_for_asr(self):
        if not self.asr_wait_for_publishers:
            return True
        deadline = time.time() + self.asr_wait_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            try:
                if self.asr_sub.get_num_connections() > 0:
                    self.publish_status("asr_ready", topic=self.asr_topic)
                    return True
            except AttributeError:
                return True
            rospy.sleep(0.1)
        self.publish_status("asr_not_ready", topic=self.asr_topic, timeout=self.asr_wait_timeout)
        rospy.logerr("No ASR publisher connected on %s", self.asr_topic)
        self.speak(self.asr_not_ready_text, wait=True)
        return False

    def asr_callback(self, message):
        answer = str(message.data or "").strip()
        if not answer:
            return
        rospy.loginfo("ASR: %s", answer)
        with self.name_condition:
            if self.accepting_name_answer:
                self.latest_name_answer = answer
            if self.accepting_help_answer:
                self.latest_help_answer = answer
            if self.accepting_switch_answer:
                self.latest_switch_answer = answer
            self.name_condition.notify_all()

    def wait_for_owner_help(self):
        with self.name_condition:
            self.latest_help_answer = None
            self.accepting_help_answer = False

        self.stop_navigation_for_interaction()
        self.speak(self.approach_help_prompt, wait=True)
        if self.asr_settle_seconds > 0.0:
            rospy.sleep(self.asr_settle_seconds)

        deadline = time.time() + self.answer_timeout
        with self.name_condition:
            self.latest_help_answer = None
            self.accepting_help_answer = True
            try:
                while (
                    self.latest_help_answer is None
                    and not rospy.is_shutdown()
                    and time.time() < deadline
                ):
                    self.name_condition.wait(timeout=0.2)
                    self.update_yolo_window("等待主人说明需求")
                return (self.latest_help_answer or "").strip()
            finally:
                self.accepting_help_answer = False

    @staticmethod
    def normalize_for_compare(text):
        return re.sub(
            r"[\s，。！？、,.!?;；:：\"'“”‘’（）()【】\[\]]+",
            "",
            str(text or ""),
        )

    @staticmethod
    def content_characters(text):
        stop_chars = set(
            "我你他她它们请要想能可以需要帮把给的了着过和与及并且然后再又是有在对从到为让向用一这那您"
        )
        return {
            char
            for char in OwnerVoiceReidTest.normalize_for_compare(text)
            if "\u4e00" <= char <= "\u9fff" and char not in stop_chars
        }

    @classmethod
    def preserves_intent(cls, transcript, rephrase):
        source = cls.normalize_for_compare(transcript)
        candidate = cls.normalize_for_compare(rephrase)
        if not source or not candidate:
            return False
        if source == candidate:
            return True

        source_chars = cls.content_characters(transcript)
        candidate_chars = cls.content_characters(rephrase)
        if not source_chars:
            return True

        overlap = len(source_chars.intersection(candidate_chars)) / float(
            len(source_chars)
        )
        if overlap < 0.4:
            return False

        source_digits = set(
            re.findall(r"\d+|[零一二两三四五六七八九十百千万亿]+", source)
        )
        candidate_digits = set(
            re.findall(r"\d+|[零一二两三四五六七八九十百千万亿]+", candidate)
        )
        if source_digits and not source_digits.issubset(candidate_digits):
            return False

        if len(candidate) > max(40, len(source) * 3.5):
            return False
        return True

    @staticmethod
    def parse_rephrase(content):
        content = str(content or "").strip()
        if not content:
            return ""

        def extract_reply(parsed):
            if not isinstance(parsed, dict):
                return ""
            for key in ("reply", "rephrase", "复述", "text"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return extract_reply(parsed)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", content, re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return extract_reply(parsed)
            except json.JSONDecodeError:
                pass

        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.I).strip()
        if content.startswith("{") and content.endswith("}"):
            return ""
        content = re.sub(
            r"^(回复|复述|重复|重述|rephrase|reply)\s*[:：]\s*",
            "",
            content,
            flags=re.I,
        )
        return content.strip("\"' ")

    def call_help_qwen(self, transcript):
        prompt = (
            "/no_think\n"
            "你是机器人。主人告诉你需求后，请站在机器人的角度，用第一人称回复，表示你会去完成这些事情。\n"
            "回复要自然、简短，必须逐个保留原话中的所有动作、对象、数量、否定、时间和先后顺序。\n"
            "严禁把对象换成近义词或你猜测的对象，例如牛奶不能改成水，垃圾不能改成杯子。\n"
            "严禁合并、删除或新增动作。不要回答‘您是说’，不要只重复主人原话。\n"
            "示例：主人说‘帮我倒一杯牛奶然后把垃圾给倒了’，应回复‘好的，我会帮您倒一杯牛奶，然后把垃圾倒掉。’\n"
            "只输出一行 JSON，不要输出任何解释，格式必须是："
            '{"reply":"机器人确认会完成的内容"}\n'
            "主人原话："
            + transcript
        )
        payload = {
            "model": self.help_llm_model,
            "stream": False,
            "think": False,
            "format": "json",
            "keep_alive": self.help_llm_keep_alive,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": 0,
                "top_p": 0.7,
                "num_predict": self.help_llm_max_tokens,
                "num_ctx": self.help_llm_num_ctx,
                "num_gpu": self.help_llm_num_gpu,
            },
        }
        request = urllib.request.Request(
            self.help_llm_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.time()
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.help_llm_timeout,
            ) as response:
                raw = response.read().decode("utf-8")
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError("Qwen 请求超时：%s" % exc)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:
                detail = ""
            raise RuntimeError(
                "Qwen HTTP %s：%s" % (exc.code, detail or exc.reason)
            )
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "无法连接本地 Qwen，请确认 Ollama 和模型 %s 已启动：%s"
                % (self.help_llm_model, exc)
            )

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Qwen 返回内容不是有效 JSON：%s" % exc)
        content = result.get("message", {}).get("content", "")
        rephrase = self.parse_rephrase(content)
        if not rephrase:
            raise RuntimeError("Qwen 没有返回有效机器人回复")
        return rephrase, time.time() - started

    def warmup_help_qwen(self):
        if not self.help_llm_warmup_enabled:
            return True

        for attempt in range(1, self.help_llm_warmup_retries + 1):
            if rospy.is_shutdown():
                return False
            try:
                rephrase, elapsed = self.call_help_qwen(
                    "这是启动预热，不是用户指令。请回复：好的，我已准备好。"
                )
                rospy.loginfo(
                    "Qwen voice warm-up complete: model=%s seconds=%.2f reply=%s",
                    self.help_llm_model,
                    elapsed,
                    rephrase,
                )
                return True
            except Exception as exc:
                if attempt >= self.help_llm_warmup_retries:
                    rospy.logwarn(
                        "Qwen voice warm-up failed; continuing with runtime fallback: %s",
                        exc,
                    )
                    return False
                rospy.logwarn(
                    "Qwen voice warm-up attempt %d failed: %s; retrying in %.1fs",
                    attempt,
                    exc,
                    self.help_llm_warmup_retry_delay,
                )
                rospy.sleep(self.help_llm_warmup_retry_delay)
        return False

    @staticmethod
    def fallback_help_reply():
        return "抱歉，我没有完全理解您的需求，请您再说一遍。"

    def publish_help_result(
        self,
        transcript,
        rephrase,
        latency=None,
        error="",
        used_fallback=False,
    ):
        payload = {
            "transcript": transcript,
            "rephrase": rephrase,
            "reply": rephrase,
            "model": self.help_llm_model,
            "latency_sec": latency,
            "error": error,
            "used_fallback": used_fallback,
        }
        self.help_result_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False))
        )

    def handle_owner_help(self, owner_result=None):
        owner_name = (owner_result or {}).get("owner_name", self.owner_name)
        self.publish_status(
            "owner_help_started",
            owner_name=owner_name,
            owner_index=(owner_result or {}).get("owner_index"),
        )
        transcript = self.wait_for_owner_help()
        if not transcript:
            timeout_text = "没有听到您的需求。"
            self.speak(timeout_text, wait=True)
            self.publish_help_result("", "", error="asr_timeout")
            self.publish_status(
                "owner_help_finished",
                owner_name=owner_name,
                owner_index=(owner_result or {}).get("owner_index"),
                transcript="",
                reply="",
                error="asr_timeout",
                used_fallback=False,
            )
            return False

        transcript = self.clean_text(transcript)
        used_fallback = False
        latency = None
        try:
            model_rephrase, latency = self.call_help_qwen(transcript)
            rephrase = model_rephrase
            if self.strict_rephrase and not self.preserves_intent(
                transcript,
                model_rephrase,
            ):
                rospy.logwarn(
                    "Qwen reply may change intent; using fallback: model=%s source=%s",
                    model_rephrase,
                    transcript,
                )
                rephrase = self.fallback_help_reply()
                used_fallback = True
            error = ""
        except Exception as exc:
            rospy.logwarn("Qwen owner-help reply failed: %s", exc)
            rephrase = self.fallback_help_reply()
            error = str(exc)
            used_fallback = True

        spoken_text = rephrase.rstrip("。！？!? ") + "。"
        self.speak(spoken_text, wait=True)
        self.publish_help_result(
            transcript,
            rephrase,
            latency=latency,
            error=error,
            used_fallback=used_fallback,
        )
        self.publish_status(
            "owner_help_finished",
            owner_name=owner_name,
            owner_index=(owner_result or {}).get("owner_index"),
            transcript=transcript,
            reply=rephrase,
            latency_sec=latency,
            error=error,
            used_fallback=used_fallback,
        )
        rospy.loginfo(
            "Owner help finished: transcript=%s reply=%s fallback=%s error=%s",
            transcript,
            rephrase,
            used_fallback,
            error or "none",
        )
        return True

    def pointcloud_callback(self, message):
        with self.lock:
            self.latest_pointcloud = message
            self.pointcloud_time = time.time()

    def ensure_pointcloud_nodelet(self):
        if rospy.get_param(self.pointcloud_nodelet_loaded_param, False):
            return True

        from nodelet.srv import NodeletLoad

        service_name = self.pointcloud_nodelet_manager + "/load_nodelet"
        rospy.wait_for_service(service_name, timeout=self.pointcloud_nodelet_timeout)
        request = NodeletLoad._request_class()
        request.name = self.pointcloud_nodelet_name
        request.type = "depth_image_proc/point_cloud_xyzrgb"
        request.remap_source_args = [
            "rgb/camera_info",
            "rgb/image_rect_color",
            "depth_registered/image_rect",
            "depth_registered/points",
        ]
        request.remap_target_args = [
            "/kinect2/qhd/camera_info",
            "/kinect2/qhd/image_color_rect",
            "/kinect2/qhd/image_depth_rect",
            self.points_topic,
        ]
        response = rospy.ServiceProxy(service_name, NodeletLoad)(request)
        if not response.success:
            raise RuntimeError(response.error or "point-cloud nodelet load failed")
        rospy.set_param(self.pointcloud_nodelet_loaded_param, True)
        return True

    def wait_for_fresh_pointcloud(self):
        with self.lock:
            self.latest_pointcloud = None
            self.pointcloud_time = None
        deadline = time.time() + self.pointcloud_nodelet_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                pointcloud = self.latest_pointcloud
            if pointcloud is not None:
                return True
            rospy.sleep(0.05)
        return False

    def scan_callback(self, message):
        with self.lock:
            self.latest_scan = message
            self.latest_scan_time = time.time()

    def action_result_callback(self, message):
        try:
            result = json.loads(message.data)
        except (TypeError, ValueError) as exc:
            rospy.logwarn("Invalid YOLO Pose action result: %s", exc)
            return
        if not isinstance(result, dict):
            rospy.logwarn("Ignoring non-object YOLO Pose action result")
            return
        self.action_result = result
        self.action_result_time = time.time()
        self.action_result_event.set()

    @staticmethod
    def roslaunch_bool(value):
        return "true" if bool(value) else "false"

    @staticmethod
    def action_owner_roi(owner_result):
        candidate = owner_result.get("candidate", {}) if owner_result else {}
        bbox = candidate.get("bbox") if isinstance(candidate, dict) else None
        if not bbox or len(bbox) < 4:
            return ""
        try:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
        except (TypeError, ValueError):
            return ""
        if x2 <= x1 or y2 <= y1:
            return ""
        return "%d,%d,%d,%d" % (x1, y1, x2, y2)

    def start_action_recognition(self, owner_result=None):
        if not self.action_recognition_enabled:
            self.publish_status("owner_action_disabled")
            return False
        if not self.action_launch_file or not os.path.exists(self.action_launch_file):
            rospy.logwarn("YOLO Pose action launch file not found: %s", self.action_launch_file)
            self.publish_status(
                "owner_action_unavailable",
                message="action launch file not found",
            )
            return False

        self.shutdown_action_recognition()
        self.action_result = None
        self.action_result_event.clear()
        self.action_result_sub = rospy.Subscriber(
            self.action_result_topic,
            String,
            self.action_result_callback,
            queue_size=1,
        )

        try:
            import roslaunch

            launch_uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(launch_uuid)
            launch_args = [
                "start_camera:=false",
                "start_voice:=false",
                "start_sound_play:=false",
                "node_name:=%s" % self.action_node_name,
                "image_topic:=%s" % self.image_topic,
                "points_topic:=%s" % self.points_topic,
                "pointcloud_nodelet_manager:=%s" % self.pointcloud_nodelet_manager,
                "pointcloud_nodelet_name:=%s" % self.pointcloud_nodelet_name,
                "pointcloud_wait_timeout:=%.3f" % self.pointcloud_nodelet_timeout,
                "say_topic:=%s" % self.say_topic,
                "result_topic:=%s" % self.action_result_topic,
                "show_window:=%s" % self.roslaunch_bool(self.action_show_window),
                "auto_analyze:=true",
                "auto_repeat_seconds:=0.0",
                "startup_settle_seconds:=%.3f" % self.action_startup_settle_seconds,
                "capture_delay:=%.3f" % self.action_capture_delay,
                "capture_duration:=%.3f" % self.action_capture_duration,
                "capture_frame_count:=%d" % self.action_capture_frame_count,
                "llm_frame_count:=%d" % self.action_llm_frame_count,
                "pose_enabled:=%s" % self.roslaunch_bool(self.action_pose_enabled),
                "pose_model_path:=%s" % self.action_pose_model_path,
                "pose_device:=%s" % self.action_pose_device,
                "pose_image_size:=%d" % self.action_pose_image_size,
                "pose_confidence:=%.3f" % self.action_pose_confidence,
                "pose_iou:=%.3f" % self.action_pose_iou,
                "pose_max_detections:=%d" % self.action_pose_max_detections,
                "llm_url:=%s" % self.action_llm_url,
                "llm_model:=%s" % self.action_llm_model,
                "llm_timeout:=%.3f" % self.action_llm_timeout,
                "llm_keep_alive:=%s" % self.action_llm_keep_alive,
                "llm_max_tokens:=%d" % self.action_llm_max_tokens,
                "llm_num_ctx:=%d" % self.action_llm_num_ctx,
                "jpeg_quality:=%d" % self.action_jpeg_quality,
                "image_max_width:=%d" % self.action_image_max_width,
                "model_warmup_retries:=%d" % self.action_model_warmup_retries,
                "model_warmup_retry_delay:=%.3f" % self.action_model_warmup_retry_delay,
                "pointcloud_enabled:=%s"
                % self.roslaunch_bool(self.action_pointcloud_enabled),
                "use_owner_roi:=%s" % self.roslaunch_bool(self.action_use_owner_roi),
                "owner_roi:=%s" % self.action_owner_roi(owner_result),
                "roi_padding:=%.3f" % self.action_roi_padding,
                "diagnostics_log_path:=%s" % self.action_diagnostics_log_path,
                "diagnostics_session_id:=%s"
                % self.action_diagnostics_session_id,
            ]
            launch_parent = roslaunch.parent.ROSLaunchParent(
                launch_uuid,
                [(self.action_launch_file, launch_args)],
            )
            launch_parent.start()
            self.action_launch_parent = launch_parent
            connect_deadline = time.time() + self.action_result_connect_timeout
            while (
                not rospy.is_shutdown()
                and time.time() < connect_deadline
                and self.action_result_sub.get_num_connections() == 0
            ):
                rospy.sleep(0.05)
            if self.action_result_sub.get_num_connections() == 0:
                rospy.logwarn(
                    "YOLO Pose action result topic has no publisher yet: %s",
                    self.action_result_topic,
                )
            rospy.loginfo(
                "Started YOLO Pose action node: name=%s result_topic=%s owner_roi=%s",
                self.action_node_name,
                self.action_result_topic,
                self.action_owner_roi(owner_result) or "disabled",
            )
            self.publish_status(
                "owner_action_started",
                node=self.action_node_name,
                result_topic=self.action_result_topic,
            )
            return True
        except Exception as exc:
            rospy.logwarn("Failed to start YOLO Pose action node: %s", exc)
            self.publish_status("owner_action_unavailable", message=str(exc))
            self.shutdown_action_recognition()
            return False

    def set_yolo_paused(self, paused):
        if not self.action_pause_yolo or self.yolo_pause_pub is None:
            return False
        try:
            self.yolo_pause_pub.publish(Bool(data=bool(paused)))
            rospy.loginfo("YOLO-World %s for owner action recognition", "paused" if paused else "resumed")
            return True
        except Exception as exc:
            rospy.logwarn("Failed to set YOLO-World pause=%s: %s", paused, exc)
            return False

    def shutdown_action_recognition(self):
        with self.action_launch_lock:
            launch_parent = self.action_launch_parent
            self.action_launch_parent = None
            self.action_process = None
            result_sub = self.action_result_sub
            self.action_result_sub = None

        if result_sub is not None:
            try:
                result_sub.unregister()
            except Exception:
                pass
        if launch_parent is not None:
            try:
                launch_parent.shutdown()
                rospy.loginfo("Stopped YOLO Pose action node")
            except Exception as exc:
                rospy.logwarn("Failed to stop YOLO Pose action node: %s", exc)

    def run_owner_action_recognition(self, owner_result):
        if self.action_completed:
            return self.action_result
        self.action_completed = True
        owner_name = owner_result.get("owner_name", self.owner_name) if owner_result else self.owner_name
        rospy.loginfo(
            "Owner confirmed; starting YOLO Pose action recognition before interaction"
        )
        yolo_paused = False
        if self.action_pause_yolo:
            yolo_paused = self.set_yolo_paused(True)
            if yolo_paused and self.action_yolo_pause_settle_seconds > 0.0:
                rospy.loginfo(
                    "Waiting %.2fs for YOLO to settle before action recognition",
                    self.action_yolo_pause_settle_seconds,
                )
                rospy.sleep(self.action_yolo_pause_settle_seconds)
        if not self.start_action_recognition(owner_result):
            if yolo_paused:
                self.set_yolo_paused(False)
            return None

        deadline = time.time() + self.action_timeout
        rate = rospy.Rate(10)
        try:
            while not rospy.is_shutdown() and time.time() < deadline:
                if self.action_result_event.is_set():
                    break
                rate.sleep()
            result = self.action_result
            if result is None:
                rospy.logwarn(
                    "Timed out waiting for YOLO Pose action result after %.1fs",
                    self.action_timeout,
                )
                self.publish_status(
                    "owner_action_timeout",
                    owner_name=owner_name,
                    timeout=self.action_timeout,
                )
                return None

            rospy.loginfo(
                "Owner action recognized: owner=%s action=%s place=%s",
                owner_name or "unknown",
                result.get("action", "unknown"),
                result.get("place", "unknown"),
            )
            self.publish_status(
                "owner_action_finished",
                owner_index=owner_result.get("owner_index") if owner_result else None,
                owner_name=owner_name,
                action=result.get("action", "unknown"),
                place=result.get("place", "unknown"),
                speech=result.get("speech", ""),
                recognizer=result.get("recognizer", "yolo_pose"),
                latency_sec=result.get("total_sec"),
            )
            rospy.sleep(self.action_speech_grace)
            return result
        finally:
            self.shutdown_action_recognition()
            if yolo_paused:
                self.set_yolo_paused(False)

    def estimate_owner_position_from_pointcloud(self, owner_result):
        candidate = owner_result.get("candidate", {}) if owner_result else {}
        bbox = candidate.get("bbox") if isinstance(candidate, dict) else None
        if not bbox or len(bbox) < 4:
            self.pointcloud_reason = "owner bbox unavailable"
            return None

        with self.lock:
            cloud = self.latest_pointcloud
            cloud_time = self.pointcloud_time
            image = None if self.latest_image is None else self.latest_image.copy()

        if cloud is None:
            self.pointcloud_reason = "no point cloud"
            return None
        if cloud_time is not None and time.time() - cloud_time > self.pointcloud_max_age:
            self.pointcloud_reason = "point cloud stale"
            return None
        if image is None or cloud.width <= 1 or cloud.height <= 1:
            self.pointcloud_reason = "invalid image or unorganized point cloud"
            return None

        image_height, image_width = image.shape[:2]
        scale_x = float(cloud.width) / float(max(1, image_width))
        scale_y = float(cloud.height) / float(max(1, image_height))
        xmin = max(0, min(cloud.width - 1, int(float(bbox[0]) * scale_x)))
        xmax = max(xmin + 1, min(cloud.width, int(float(bbox[2]) * scale_x)))
        ymin = max(0, min(cloud.height - 1, int(float(bbox[1]) * scale_y)))
        ymax = max(ymin + 1, min(cloud.height, int(float(bbox[3]) * scale_y)))
        box_width = max(1, xmax - xmin)
        box_height = max(1, ymax - ymin)
        x_margin = int(box_width * self.pointcloud_roi_x_margin)
        roi_x1 = max(0, min(cloud.width - 1, xmin + x_margin))
        roi_x2 = max(roi_x1 + 1, min(cloud.width, xmax - x_margin))
        roi_y1 = max(
            0,
            min(
                cloud.height - 1,
                ymin + int(box_height * self.pointcloud_roi_y_min_ratio),
            ),
        )
        roi_y2 = max(
            roi_y1 + 1,
            min(
                cloud.height,
                ymin + int(box_height * self.pointcloud_roi_y_max_ratio),
            ),
        )
        uvs = [
            (u, v)
            for v in range(roi_y1, roi_y2, self.pointcloud_stride)
            for u in range(roi_x1, roi_x2, self.pointcloud_stride)
        ]
        points = []
        try:
            for x, y, z in pc2.read_points(
                cloud,
                field_names=("x", "y", "z"),
                skip_nans=True,
                uvs=uvs,
            ):
                values = (float(x), float(y), float(z))
                if all(math.isfinite(value) for value in values):
                    points.append(values)
        except Exception as exc:
            self.pointcloud_reason = "point cloud read failed: %s" % exc
            return None

        if len(points) < self.pointcloud_min_samples:
            self.pointcloud_reason = "too few valid point cloud samples: %d" % len(points)
            return None

        raw = np.asarray(points, dtype=np.float32)
        raw_x = float(np.median(raw[:, 0]))
        raw_y = float(np.median(raw[:, 1]))
        raw_z = float(np.median(raw[:, 2]))
        mode = self.pointcloud_frame_mode
        if mode not in ("auto", "optical", "base"):
            mode = "auto"
        frame = str(getattr(cloud.header, "frame_id", "") or "").lower()
        horizontal = math.hypot(raw_x, raw_y)
        use_optical = mode == "optical" or (
            mode == "auto"
            and ("optical" in frame or raw_z > max(0.8, horizontal * 1.35))
        )
        if use_optical:
            forward = raw[:, 2]
            lateral = -raw[:, 0]
            used_mode = "optical"
        else:
            forward = raw[:, 0]
            lateral = raw[:, 1]
            used_mode = "base"

        valid = []
        for forward_value, lateral_value in zip(forward, lateral):
            forward_value = float(forward_value)
            lateral_value = float(lateral_value)
            if (
                0.30 <= forward_value <= 5.0
                and abs(lateral_value) <= 5.0
                and math.isfinite(forward_value)
                and math.isfinite(lateral_value)
            ):
                valid.append((forward_value, lateral_value))
        if len(valid) < self.pointcloud_min_samples:
            self.pointcloud_reason = "too few in-range point cloud samples"
            return None

        valid = np.asarray(valid, dtype=np.float32)
        forward_value = float(np.median(valid[:, 0]))
        lateral_value = float(np.median(valid[:, 1]))
        self.pointcloud_reason = ""
        return {
            "x": forward_value,
            "y": lateral_value,
            "forward": forward_value,
            "lateral": lateral_value,
            "distance": math.hypot(forward_value, lateral_value),
            "bearing": math.atan2(lateral_value, max(0.05, forward_value)),
            "mode": used_mode,
            "samples": int(len(valid)),
            "raw": (raw_x, raw_y, raw_z),
        }

    def get_latest_odom_xy(self):
        with self.lock:
            return self.latest_odom_xy

    def get_latest_odom_linear_speed(self, max_age=None):
        with self.lock:
            speed = self.latest_odom_linear_speed
            stamp = self.latest_odom_time
        if speed is None:
            return None
        if max_age is not None and (stamp is None or time.time() - stamp > max_age):
            return None
        return float(speed)

    def wait_for_odom_xy(self, timeout=1.0):
        deadline = time.time() + max(0.0, float(timeout))
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
        return math.hypot(
            float(end_xy[0]) - float(start_xy[0]),
            float(end_xy[1]) - float(start_xy[1]),
        )

    def front_scan_distance(self):
        with self.lock:
            scan = self.latest_scan
            scan_time = self.latest_scan_time
        if scan is None or not scan.ranges:
            return None
        if scan_time is None or time.time() - scan_time > self.approach_scan_max_age:
            rospy.logwarn_throttle(
                2.0,
                "Front lidar data is stale during owner approach",
            )
            return None

        half_window = math.radians(self.approach_front_scan_degrees)
        values = []
        for index, distance in enumerate(scan.ranges):
            angle = scan.angle_min + index * scan.angle_increment
            if (
                abs(angle) <= half_window
                and math.isfinite(distance)
                and scan.range_min < distance < scan.range_max
            ):
                values.append(float(distance))
        return min(values) if values else None

    def ensure_move_base_for_approach(self):
        if not self.approach_navigation_enabled:
            self.last_approach_failure_reason = "move_base owner approach is disabled"
            rospy.logwarn(self.last_approach_failure_reason)
            return False
        if self.move_base.wait_for_server(
            rospy.Duration(self.approach_navigation_server_timeout)
        ):
            return True
        self.last_approach_failure_reason = (
            "move_base action server unavailable for owner approach"
        )
        rospy.logwarn(self.last_approach_failure_reason)
        return False

    def lookup_latest_transform(self, target_frame, source_frame):
        last_error = None
        for _ in range(2):
            try:
                self.tf_listener.waitForTransform(
                    target_frame,
                    source_frame,
                    rospy.Time(0),
                    rospy.Duration(self.approach_navigation_tf_timeout),
                )
                transform = self.tf_listener.lookupTransform(
                    target_frame,
                    source_frame,
                    rospy.Time(0),
                )
                try:
                    latest_common_time = self.tf_listener.getLatestCommonTime(
                        target_frame,
                        source_frame,
                    )
                    if isinstance(latest_common_time, tuple):
                        latest_common_time = latest_common_time[0]
                    if latest_common_time and latest_common_time != rospy.Time(0):
                        age = (rospy.Time.now() - latest_common_time).to_sec()
                        if age > self.approach_navigation_tf_max_age:
                            rospy.logwarn_throttle(
                                2.0,
                                "Latest %s<-%s TF is %.2fs old",
                                target_frame,
                                source_frame,
                                age,
                            )
                except Exception:
                    pass
                return transform
            except Exception as exc:
                last_error = exc
                rospy.sleep(0.05)
        raise last_error

    def wait_for_fresh_navigation_tf(self, label="owner approach", timeout=None):
        if self.tf_listener is None:
            self.last_approach_failure_reason = "tf is unavailable for owner approach"
            rospy.logwarn(self.last_approach_failure_reason)
            return False
        deadline = time.time() + max(
            0.1,
            float(timeout if timeout is not None else self.approach_navigation_tf_timeout),
        )
        last_error = ""
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() < deadline:
            try:
                latest_common_time = self.tf_listener.getLatestCommonTime(
                    self.approach_navigation_frame,
                    self.approach_navigation_base_frame,
                )
                if isinstance(latest_common_time, tuple):
                    latest_common_time = latest_common_time[0]
                if latest_common_time and latest_common_time != rospy.Time(0):
                    age = (rospy.Time.now() - latest_common_time).to_sec()
                    if age <= self.approach_navigation_tf_max_age:
                        return True
                    last_error = "latest tf age %.2fs > %.2fs" % (
                        age,
                        self.approach_navigation_tf_max_age,
                    )
            except Exception as exc:
                last_error = str(exc)
            rate.sleep()
        self.last_approach_failure_reason = (
            "%s waiting for fresh %s<-%s tf timed out: %s"
            % (
                label,
                self.approach_navigation_frame,
                self.approach_navigation_base_frame,
                last_error or "no common tf time",
            )
        )
        rospy.logwarn(self.last_approach_failure_reason)
        return False

    def lookup_robot_navigation_pose(self):
        if self.tf_listener is None:
            self.last_approach_failure_reason = "tf is unavailable for owner approach"
            rospy.logwarn(self.last_approach_failure_reason)
            return None
        try:
            translation, rotation = self.lookup_latest_transform(
                self.approach_navigation_frame,
                self.approach_navigation_base_frame,
            )
            yaw = tf.transformations.euler_from_quaternion(rotation)[2]
            return float(translation[0]), float(translation[1]), float(yaw)
        except Exception as exc:
            self.last_approach_failure_reason = (
                "cannot transform robot pose for owner approach: %s" % exc
            )
            rospy.logwarn_throttle(2.0, self.last_approach_failure_reason)
            return None

    @staticmethod
    def relative_navigation_xy(robot_x, robot_y, robot_yaw, forward, lateral):
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        return (
            robot_x + cos_yaw * float(forward) - sin_yaw * float(lateral),
            robot_y + sin_yaw * float(forward) + cos_yaw * float(lateral),
        )

    def relative_navigation_goal(self, forward, lateral=0.0, yaw=0.0):
        robot_pose = self.lookup_robot_navigation_pose()
        if robot_pose is None:
            return None
        robot_x, robot_y, robot_yaw = robot_pose
        map_x, map_y = self.relative_navigation_xy(
            robot_x,
            robot_y,
            robot_yaw,
            forward,
            lateral,
        )
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

    def navigation_goal_remaining_distance(self, goal):
        robot_pose = self.lookup_robot_navigation_pose()
        if robot_pose is None:
            return None
        return math.hypot(
            float(goal.target_pose.pose.position.x) - robot_pose[0],
            float(goal.target_pose.pose.position.y) - robot_pose[1],
        )

    def approach_debug_event(self, event, **fields):
        return None

    def send_approach_navigation_goal(
        self,
        goal,
        move_distance,
        timeout=None,
        label="owner approach",
        lidar_guard_distance=None,
        stop_still_duration=None,
        early_stop_owner_xy=None,
        early_stop_owner_distance=None,
    ):
        self.move_base.send_goal(goal)
        self.approach_debug_event(
            "navigation_goal_sent",
            label=label,
            move_distance=float(move_distance),
            goal_frame=goal.target_pose.header.frame_id,
            goal_x=float(goal.target_pose.pose.position.x),
            goal_y=float(goal.target_pose.pose.position.y),
            goal_orientation_z=float(goal.target_pose.pose.orientation.z),
            goal_orientation_w=float(goal.target_pose.pose.orientation.w),
        )
        start_xy = self.wait_for_odom_xy(timeout=0.5)
        deadline = time.time() + max(
            0.1,
            float(timeout if timeout is not None else self.approach_navigation_timeout),
        )
        best_remaining = float(move_distance)
        last_progress_time = time.time()
        movement_seen = False
        still_since = None
        rate = rospy.Rate(10)

        while not rospy.is_shutdown() and time.time() < deadline:
            state = self.move_base.get_state()
            travelled = self.xy_distance(start_xy, self.get_latest_odom_xy())
            remaining = self.navigation_goal_remaining_distance(goal)
            if remaining is None and travelled is not None:
                remaining = max(0.0, float(move_distance) - travelled)

            current_speed = self.get_latest_odom_linear_speed(
                self.approach_odom_speed_max_age
            )
            if travelled is not None and travelled >= self.approach_navigation_min_distance:
                movement_seen = True
            elif current_speed is not None and current_speed > self.approach_navigation_stuck_linear_speed:
                movement_seen = True

            if remaining is not None:
                if remaining < best_remaining - self.approach_navigation_stuck_min_progress:
                    best_remaining = remaining
                    last_progress_time = time.time()
            elif travelled is not None and travelled > self.approach_navigation_stuck_min_progress:
                last_progress_time = time.time()

            owner_distance = None
            if early_stop_owner_xy is not None:
                robot_pose = self.lookup_robot_navigation_pose()
                if robot_pose is not None:
                    owner_distance = math.hypot(
                        float(early_stop_owner_xy[0]) - robot_pose[0],
                        float(early_stop_owner_xy[1]) - robot_pose[1],
                    )
                    if (
                        early_stop_owner_xy is not None
                        and early_stop_owner_distance is not None
                        and owner_distance <= float(early_stop_owner_distance)
                    ):
                        self.move_base.cancel_goal()
                        self.stop_base()
                        rospy.loginfo(
                            "%s stopped in owner safety circle: %.2fm",
                            label,
                            owner_distance,
                        )
                        return True, ""

            if stop_still_duration is not None and movement_seen:
                near_target = (
                    remaining is not None
                    and remaining <= max(0.20, self.approach_slow_finish_tolerance)
                )
                if (
                    current_speed is not None
                    and current_speed <= self.approach_slow_finish_linear_speed
                    and near_target
                ):
                    if still_since is None:
                        still_since = time.time()
                    elif time.time() - still_since >= float(stop_still_duration):
                        self.move_base.cancel_goal()
                        self.stop_base()
                        self.approach_debug_event(
                            "navigation_goal_finished",
                            label=label,
                            success=True,
                            reason="stationary_near_target",
                            remaining=remaining,
                        )
                        return True, ""
                else:
                    still_since = None

            if remaining is not None and remaining <= self.approach_navigation_min_distance:
                self.move_base.cancel_goal()
                self.stop_base()
                self.approach_debug_event(
                    "navigation_goal_finished",
                    label=label,
                    success=True,
                    reason="minimum_goal_distance",
                    remaining=remaining,
                )
                return True, ""

            if state == GoalStatus.SUCCEEDED:
                tolerance = max(
                    self.approach_navigation_min_distance,
                    self.approach_slow_finish_tolerance,
                )
                if remaining is None or remaining <= tolerance:
                    self.stop_base()
                    self.approach_debug_event(
                        "navigation_goal_finished",
                        label=label,
                        success=True,
                        reason="move_base_succeeded",
                        remaining=remaining,
                        tolerance=tolerance,
                    )
                    return True, ""
                self.stop_base()
                failure_reason = (
                    "%s reported success before reaching the approach goal: remaining=%.2f"
                    % (label, remaining)
                )
                self.approach_debug_event(
                    "navigation_goal_finished",
                    label=label,
                    success=False,
                    reason=failure_reason,
                    remaining=remaining,
                )
                return False, failure_reason

            if state in (
                GoalStatus.PREEMPTED,
                GoalStatus.ABORTED,
                GoalStatus.REJECTED,
                GoalStatus.RECALLED,
                GoalStatus.LOST,
            ):
                if (
                    owner_distance is not None
                    and early_stop_owner_distance is not None
                    and owner_distance <= float(early_stop_owner_distance)
                ):
                    self.stop_base()
                    rospy.logwarn(
                        "%s reported navigation failure after reaching the owner "
                        "safety circle (%.2fm); continuing interaction",
                        label,
                        owner_distance,
                    )
                    self.approach_debug_event(
                        "navigation_goal_finished",
                        label=label,
                        success=True,
                        reason="move_base_failed_inside_owner_safety_circle",
                        state=int(state),
                        owner_distance=owner_distance,
                    )
                    return True, ""
                self.stop_base()
                failure_reason = "%s failed with move_base state %s" % (label, state)
                self.approach_debug_event(
                    "navigation_goal_finished",
                    label=label,
                    success=False,
                    reason=failure_reason,
                    state=int(state),
                    owner_distance=owner_distance,
                )
                return False, failure_reason

            if lidar_guard_distance is not None:
                front_distance = self.front_scan_distance()
                if front_distance is None:
                    self.move_base.cancel_goal()
                    self.stop_base()
                    failure_reason = "%s stopped because lidar data is unavailable or stale" % label
                    self.approach_debug_event(
                        "navigation_goal_finished",
                        label=label,
                        success=False,
                        reason=failure_reason,
                    )
                    return False, failure_reason
                if front_distance <= lidar_guard_distance:
                    self.move_base.cancel_goal()
                    self.stop_base()
                    if label.startswith("waving owner approach"):
                        rospy.loginfo(
                            "%s stopped by obstacle protection at %.2fm; "
                            "treating waving approach as successful",
                            label,
                            front_distance,
                        )
                        self.approach_debug_event(
                            "navigation_goal_finished",
                            label=label,
                            success=True,
                            reason="lidar_obstacle_protection_for_waving",
                            front_distance=front_distance,
                            lidar_guard_distance=lidar_guard_distance,
                        )
                        return True, ""
                    if owner_distance is None:
                        failure_reason = "%s blocked by close obstacle: lidar=%.2f guard=%.2f" % (
                            label,
                            front_distance,
                            lidar_guard_distance,
                        )
                        self.approach_debug_event(
                            "navigation_goal_finished",
                            label=label,
                            success=False,
                            reason=failure_reason,
                            front_distance=front_distance,
                            lidar_guard_distance=lidar_guard_distance,
                        )
                        return False, failure_reason
                    failure_reason = (
                        "%s blocked by close obstacle: lidar=%.2f guard=%.2f "
                        "owner_distance=%.2f"
                        % (label, front_distance, lidar_guard_distance, owner_distance)
                    )
                    self.approach_debug_event(
                        "navigation_goal_finished",
                        label=label,
                        success=False,
                        reason=failure_reason,
                        front_distance=front_distance,
                        lidar_guard_distance=lidar_guard_distance,
                        owner_distance=owner_distance,
                    )
                    return False, failure_reason

            if (
                self.approach_navigation_stuck_timeout > 0.0
                and time.time() - last_progress_time >= self.approach_navigation_stuck_timeout
                and (current_speed is None or current_speed <= self.approach_navigation_stuck_linear_speed)
            ):
                self.move_base.cancel_goal()
                self.stop_base()
                failure_reason = (
                    "%s stuck or blocked before target: remaining=%s"
                    % (
                        label,
                        "%.2f" % remaining if remaining is not None else "unknown",
                    )
                )
                self.approach_debug_event(
                    "navigation_goal_finished",
                    label=label,
                    success=False,
                    reason=failure_reason,
                    remaining=remaining,
                    current_speed=current_speed,
                )
                return False, failure_reason

            self.approach_debug_event(
                "navigation_tick",
                label=label,
                state=int(state),
                travelled=travelled,
                remaining=remaining,
                current_speed=current_speed,
                owner_distance=owner_distance,
            )

            self.update_yolo_window(
                "%s %.2fm" % (label, max(0.0, remaining if remaining is not None else 0.0))
            )
            rate.sleep()

        self.move_base.cancel_goal()
        self.stop_base()
        if early_stop_owner_xy is not None and early_stop_owner_distance is not None:
            robot_pose = self.lookup_robot_navigation_pose()
            if robot_pose is not None:
                owner_distance = math.hypot(
                    float(early_stop_owner_xy[0]) - robot_pose[0],
                    float(early_stop_owner_xy[1]) - robot_pose[1],
                )
                if owner_distance <= float(early_stop_owner_distance):
                    rospy.logwarn(
                        "%s timed out after reaching the owner safety circle "
                        "(%.2fm); continuing interaction",
                        label,
                        owner_distance,
                    )
                    self.approach_debug_event(
                        "navigation_goal_finished",
                        label=label,
                        success=True,
                        reason="timeout_inside_owner_safety_circle",
                        owner_distance=owner_distance,
                    )
                    return True, ""
        failure_reason = "move_base timed out while navigating to %s" % label
        self.approach_debug_event(
            "navigation_goal_finished",
            label=label,
            success=False,
            reason=failure_reason,
        )
        return False, failure_reason

    def navigate_relative_for_approach(
        self,
        forward,
        lateral=0.0,
        yaw=0.0,
        timeout=None,
        label="owner approach",
        lidar_guard_distance=None,
        stop_still_duration=None,
        early_stop_owner_xy=None,
        early_stop_owner_distance=None,
    ):
        move_distance = math.hypot(float(forward), float(lateral))
        if move_distance <= self.approach_navigation_min_distance:
            self.stop_base()
            return True
        if not self.ensure_move_base_for_approach():
            return False
        if (
            self.clear_costmaps_before_navigation
            and self.approach_navigation_clear_costmaps_before_goal
        ):
            self.clear_move_base_costmaps("before %s" % label)

        goal = self.relative_navigation_goal(forward, lateral, yaw)
        if goal is None:
            return False
        wait_timeout = timeout or self.approach_navigation_timeout
        success, error_message = self.send_approach_navigation_goal(
            goal,
            move_distance,
            timeout=wait_timeout,
            label=label,
            lidar_guard_distance=lidar_guard_distance,
            stop_still_duration=stop_still_duration,
            early_stop_owner_xy=early_stop_owner_xy,
            early_stop_owner_distance=early_stop_owner_distance,
        )
        if (
            not success
            and self.approach_navigation_retry_after_clear
            and "blocked by close obstacle" not in error_message
            and "stuck or blocked" not in error_message
        ):
            self.clear_move_base_costmaps("after %s failure" % label)
            goal.target_pose.header.stamp = rospy.Time.now()
            success, error_message = self.send_approach_navigation_goal(
                goal,
                move_distance,
                timeout=wait_timeout,
                label=label,
                lidar_guard_distance=lidar_guard_distance,
                stop_still_duration=stop_still_duration,
                early_stop_owner_xy=early_stop_owner_xy,
                early_stop_owner_distance=early_stop_owner_distance,
            )
        self.stop_base()
        if not success:
            self.last_approach_failure_reason = error_message
            rospy.logwarn("%s failed: %s", label, error_message)
        return success

    def current_pose_for_plan(self, target_frame):
        if self.tf_listener is None:
            return None
        try:
            translation, rotation = self.lookup_latest_transform(
                target_frame,
                self.approach_navigation_base_frame,
            )
        except Exception as exc:
            rospy.logwarn_throttle(
                3.0,
                "Cannot get robot pose for approach plan check: %s",
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
            self.approach_debug_event(
                "global_plan_check",
                checked=False,
                accepted=True,
                reason="plan_check_disabled",
                goal_frame=goal.target_pose.header.frame_id,
                goal_x=float(goal.target_pose.pose.position.x),
                goal_y=float(goal.target_pose.pose.position.y),
            )
            return True
        start = self.current_pose_for_plan(goal.target_pose.header.frame_id)
        if start is None:
            self.approach_debug_event(
                "global_plan_check",
                checked=False,
                accepted=True,
                reason="robot_pose_unavailable",
                goal_frame=goal.target_pose.header.frame_id,
                goal_x=float(goal.target_pose.pose.position.x),
                goal_y=float(goal.target_pose.pose.position.y),
            )
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
                "Cannot check waving owner plan: %s",
                exc,
            )
            self.approach_debug_event(
                "global_plan_check",
                checked=False,
                accepted=True,
                reason="plan_service_unavailable",
                exception=str(exc),
            )
            return True

        poses = response.plan.poses
        if len(poses) <= 1:
            self.approach_debug_event(
                "global_plan_check",
                checked=True,
                accepted=False,
                reason="plan_empty_or_single_pose",
                plan_pose_count=len(poses),
            )
            return False
        plan_length = 0.0
        total_turn = 0.0
        previous_pose = poses[0].pose.position
        previous_heading = None
        for plan_pose in poses[1:]:
            current_pose = plan_pose.pose.position
            delta_x = float(current_pose.x) - float(previous_pose.x)
            delta_y = float(current_pose.y) - float(previous_pose.y)
            segment_length = math.hypot(delta_x, delta_y)
            plan_length += segment_length
            if segment_length > 1e-3:
                heading = math.atan2(delta_y, delta_x)
                if previous_heading is not None:
                    total_turn += abs(
                        math.atan2(
                            math.sin(heading - previous_heading),
                            math.cos(heading - previous_heading),
                        )
                    )
                previous_heading = heading
            previous_pose = current_pose
        start_position = start.pose.position
        direct_distance = math.hypot(
            float(goal.target_pose.pose.position.x) - float(start_position.x),
            float(goal.target_pose.pose.position.y) - float(start_position.y),
        )
        detour_rejected = (
            plan_length > max(0.5, direct_distance) * self.waving_approach_plan_detour_ratio
            and plan_length > direct_distance + self.waving_approach_plan_detour_margin
        )
        turn_rejected = total_turn > self.waving_approach_plan_turn_limit
        accepted = not detour_rejected and not turn_rejected
        self.approach_debug_event(
            "global_plan_check",
            checked=True,
            accepted=accepted,
            reason=(
                "detour_too_long"
                if detour_rejected
                else "turn_limit_exceeded"
                if turn_rejected
                else "plan_accepted"
            ),
            plan_pose_count=len(poses),
            plan_length=plan_length,
            direct_distance=direct_distance,
            total_turn=total_turn,
            detour_ratio_limit=self.waving_approach_plan_detour_ratio,
            detour_margin_limit=self.waving_approach_plan_detour_margin,
            turn_limit=self.waving_approach_plan_turn_limit,
        )
        return accepted

    def waving_owner_standoff_candidates(
        self,
        position,
        standoff_distance,
        front_only=False,
    ):
        owner_x = float(position.get("x", position.get("forward", 0.0)))
        owner_y = float(position.get("y", position.get("lateral", 0.0)))
        distance = math.hypot(owner_x, owner_y)
        if distance <= 1e-3 or not all(math.isfinite(value) for value in (owner_x, owner_y)):
            return []

        min_distance = self.clamp_value(
            abs(self.waving_approach_min_owner_distance),
            0.35,
            0.50,
        )
        max_distance = self.clamp_value(
            abs(self.waving_approach_max_owner_distance),
            min_distance,
            0.50,
        )
        if front_only:
            max_distance = max(max_distance, abs(float(standoff_distance)))
        requested = self.clamp_value(abs(float(standoff_distance)), min_distance, max_distance)
        raw_distances = (
            [requested]
            if front_only
            else [requested] + list(self.waving_approach_candidate_distances)
        )
        candidate_distances = []
        for raw_distance in raw_distances:
            try:
                value = float(raw_distance)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                value = self.clamp_value(
                    max(abs(value), self.waving_approach_safety_radius),
                    min_distance,
                    max_distance,
                )
                if round(value, 2) not in [round(item, 2) for item in candidate_distances]:
                    candidate_distances.append(value)

        if front_only:
            angles = [0.0]
        else:
            angles = []
            for raw_angle in self.waving_approach_candidate_angles_deg:
                try:
                    value = float(raw_angle)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    angles.append(value)
            if not angles:
                angles = [65.0, -65.0, 95.0, -95.0, 35.0, -35.0, 0.0]
            if all(abs(value) > 1e-3 for value in angles):
                angles.append(0.0)

        near_side_angle = math.atan2(-owner_y, -owner_x)
        candidates = []
        seen = set()
        for owner_clearance in candidate_distances:
            for angle_deg in angles:
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
                candidates.append(
                    {
                        "forward": goal_x,
                        "lateral": goal_y,
                        "yaw": yaw,
                        "owner_clearance": owner_clearance,
                        "goal_distance": goal_distance,
                    }
                )
        return candidates

    def navigate_to_waving_owner_candidates(
        self,
        position,
        standoff_distance,
        timeout=None,
        label="waving owner approach",
        lidar_guard_distance=None,
        front_only=False,
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

        candidates = self.waving_owner_standoff_candidates(
            position,
            standoff_distance,
            front_only=front_only,
        )
        self.approach_debug_event(
            "waving_candidates_generated",
            candidate_count=len(candidates),
            requested_standoff=float(standoff_distance),
            front_only=bool(front_only),
            owner_local_x=float(position.get("x", position.get("forward", 0.0))),
            owner_local_y=float(position.get("y", position.get("lateral", 0.0))),
            candidates=[
                {
                    "index": index,
                    "forward": candidate["forward"],
                    "lateral": candidate["lateral"],
                    "yaw": candidate["yaw"],
                    "owner_clearance": candidate["owner_clearance"],
                    "goal_distance": candidate["goal_distance"],
                }
                for index, candidate in enumerate(candidates, start=1)
            ],
        )
        if not candidates:
            self.last_approach_failure_reason = "no valid waving owner standoff candidates"
            self.approach_debug_event(
                "waving_candidates_finished",
                success=False,
                reason=self.last_approach_failure_reason,
            )
            return False

        robot_pose = self.lookup_robot_navigation_pose()
        if robot_pose is None:
            self.approach_debug_event(
                "waving_candidates_finished",
                success=False,
                reason="robot_navigation_pose_unavailable",
            )
            return False
        owner_map_xy = self.relative_navigation_xy(
            robot_pose[0],
            robot_pose[1],
            robot_pose[2],
            float(position.get("x", position.get("forward", 0.0))),
            float(position.get("y", position.get("lateral", 0.0))),
        )
        current_owner_distance = math.hypot(
            float(owner_map_xy[0]) - robot_pose[0],
            float(owner_map_xy[1]) - robot_pose[1],
        )
        if current_owner_distance <= self.waving_approach_safety_radius:
            self.stop_base()
            rospy.loginfo(
                "%s already inside owner safety circle (%.2fm); "
                "continuing interaction",
                label,
                current_owner_distance,
            )
            self.approach_debug_event(
                "waving_candidates_finished",
                success=True,
                reason="already_inside_owner_safety_circle",
                owner_distance=current_owner_distance,
                safety_radius=self.waving_approach_safety_radius,
            )
            return True
        last_error = "no reachable waving owner candidate"
        for index, candidate in enumerate(candidates, start=1):
            goal = self.relative_navigation_goal(
                candidate["forward"],
                candidate["lateral"],
                candidate["yaw"],
            )
            if goal is None:
                self.approach_debug_event(
                    "waving_candidate_rejected",
                    candidate_index=index,
                    reason="relative_goal_unavailable",
                    candidate=candidate,
                )
                last_error = "waving owner candidate %d has no relative goal" % index
                continue
            plan_accepted = self.waving_goal_has_global_plan(goal)
            if not plan_accepted:
                last_error = "waving owner candidate %d has no safe global plan" % index
                self.approach_debug_event(
                    "waving_candidate_rejected",
                    candidate_index=index,
                    reason="global_plan_rejected",
                    candidate=candidate,
                    goal_x=float(goal.target_pose.pose.position.x),
                    goal_y=float(goal.target_pose.pose.position.y),
                )
                continue
            candidate_label = "%s candidate %d" % (label, index)
            self.approach_debug_event(
                "waving_candidate_started",
                candidate_index=index,
                label=candidate_label,
                candidate=candidate,
                goal_x=float(goal.target_pose.pose.position.x),
                goal_y=float(goal.target_pose.pose.position.y),
                lidar_guard_distance=lidar_guard_distance,
            )
            success, error_message = self.send_approach_navigation_goal(
                goal,
                candidate["goal_distance"],
                timeout=timeout or self.approach_navigation_timeout,
                label=candidate_label,
                lidar_guard_distance=lidar_guard_distance,
                stop_still_duration=self.waving_approach_still_duration,
                early_stop_owner_xy=owner_map_xy,
                early_stop_owner_distance=self.waving_approach_safety_radius,
            )
            self.stop_base()
            self.approach_debug_event(
                "waving_candidate_finished",
                candidate_index=index,
                label=candidate_label,
                success=bool(success),
                error_message=error_message,
            )
            if success:
                return True
            last_error = error_message or last_error
            if self.waving_approach_retry_after_clear and "obstacle" not in last_error:
                self.approach_debug_event(
                    "waving_candidate_costmap_clear",
                    candidate_index=index,
                    label=candidate_label,
                    reason=last_error,
                )
                self.clear_move_base_costmaps("after %s failure" % candidate_label)

        self.last_approach_failure_reason = last_error
        self.approach_debug_event(
            "waving_candidates_finished",
            success=False,
            reason=last_error,
        )
        return False

    def navigate_to_owner_standoff(
        self,
        position,
        standoff_distance,
        max_travel=None,
        timeout=None,
        label="owner approach",
        lidar_guard_distance=None,
    ):
        distance = max(0.0, float(position.get("distance", 0.0)))
        if distance <= 1e-3:
            self.stop_base()
            self.last_approach_failure_reason = "owner distance unavailable"
            return False

        travel_distance = max(0.0, distance - max(0.0, float(standoff_distance)))
        travel_limit = self.approach_max_travel_distance if max_travel is None else max_travel
        if travel_limit > 0.0:
            travel_distance = min(travel_distance, travel_limit)
        if travel_distance <= self.approach_distance_tolerance:
            self.stop_base()
            return True

        scale = travel_distance / distance
        forward = float(position.get("x", position.get("forward", 0.0))) * scale
        lateral = float(position.get("y", position.get("lateral", 0.0))) * scale
        yaw = math.atan2(
            float(position.get("y", position.get("lateral", 0.0))) - lateral,
            max(0.05, float(position.get("x", position.get("forward", 0.0))) - forward),
        )
        return self.navigate_relative_for_approach(
            forward,
            lateral,
            yaw=yaw,
            timeout=timeout,
            label=label,
            lidar_guard_distance=lidar_guard_distance,
        )

    def approach_owner(
        self,
        owner_result,
        standoff_distance,
        waving=False,
        lidar_guard_distance_override=None,
        waving_front_only=False,
    ):
        if not self.interaction_enabled or not self.approach_enabled:
            rospy.loginfo("Owner approach disabled")
            return True

        self.last_approach_failure_reason = ""
        try:
            self.ensure_pointcloud_nodelet()
            if not self.wait_for_fresh_pointcloud():
                self.pointcloud_reason = "timed out waiting for fresh point cloud"
                self.last_approach_failure_reason = self.pointcloud_reason
                self.approach_debug_event(
                    "approach_precondition_failed",
                    reason=self.pointcloud_reason,
                    stage="pointcloud",
                )
                return False
        except Exception as exc:
            self.pointcloud_reason = "could not start point-cloud nodelet: %s" % exc
            self.last_approach_failure_reason = self.pointcloud_reason
            rospy.logwarn(self.last_approach_failure_reason)
            self.approach_debug_event(
                "approach_precondition_failed",
                reason=self.pointcloud_reason,
                stage="pointcloud_start",
            )
            return False

        position = self.estimate_owner_position_from_pointcloud(owner_result)
        if position is None:
            self.stop_base()
            rospy.logwarn(
                "Owner approach cannot start because point-cloud position is invalid: %s",
                self.pointcloud_reason,
            )
            self.approach_debug_event(
                "approach_precondition_failed",
                reason=self.pointcloud_reason,
                stage="owner_position",
            )
            return False

        lidar_guard_distance = None
        if self.approach_navigation_lidar_guard_enabled:
            if lidar_guard_distance_override is None:
                lidar_guard_distance = (
                    self.approach_lidar_stop_distance + self.approach_lidar_margin
                )
            else:
                lidar_guard_distance = max(
                    0.05,
                    float(lidar_guard_distance_override),
                )
            if self.front_scan_distance() is None:
                self.stop_base()
                self.last_approach_failure_reason = (
                    "fresh lidar data is required before owner approach"
                )
                rospy.logwarn(self.last_approach_failure_reason)
                self.approach_debug_event(
                    "approach_precondition_failed",
                    reason=self.last_approach_failure_reason,
                    stage="lidar",
                )
                return False
        if waving:
            approached = self.navigate_to_waving_owner_candidates(
                position,
                standoff_distance,
                timeout=self.approach_navigation_timeout,
                label="waving owner approach",
                lidar_guard_distance=lidar_guard_distance,
                front_only=waving_front_only,
            )
        else:
            approached = self.navigate_to_owner_standoff(
                position,
                standoff_distance,
                max_travel=self.approach_max_travel_distance,
                timeout=self.approach_navigation_timeout,
                label="owner approach",
                lidar_guard_distance=lidar_guard_distance,
            )
        self.stop_base()
        self.update_yolo_window("已靠近主人" if approached else "靠近主人失败")
        return approached

    def approach_fallen_owner(self, owner_result):
        return self.approach_owner(
            owner_result,
            self.fall_approach_standoff_distance,
            lidar_guard_distance_override=max(
                0.05,
                self.fall_approach_lidar_stop_distance
                + self.fall_approach_lidar_margin,
            ),
        )

    def publish_manipulator_command(self, lift, gripper):
        if self.mani_ctrl_pub is None:
            rospy.logwarn("Manipulator publisher is unavailable")
            return
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = ["lift", "gripper"]
        message.position = [float(lift), float(gripper)]
        message.velocity = [
            float(self.fall_assist_arm_lift_velocity),
            float(self.fall_assist_arm_gripper_velocity),
        ]
        self.mani_ctrl_pub.publish(message)

    def hold_manipulator_command(self, lift, gripper, seconds):
        deadline = time.time() + max(0.0, float(seconds))
        rate = rospy.Rate(self.fall_assist_arm_command_rate)
        self.publish_manipulator_command(lift, gripper)
        while not rospy.is_shutdown() and time.time() < deadline:
            self.publish_manipulator_command(lift, gripper)
            rate.sleep()

    def perform_fall_assist_arm_motion(self):
        if not self.fall_assist_arm_enabled:
            rospy.logwarn("Fall assist arm motion is disabled")
            return True
        if self.mani_ctrl_pub is None:
            rospy.logerr("Cannot perform fall assist arm motion: publisher is unavailable")
            return False
        if self.mani_ctrl_pub.get_num_connections() == 0:
            rospy.logwarn(
                "No subscriber connected to %s; publishing fall assist arm commands anyway",
                self.mani_ctrl_topic,
            )
        rospy.loginfo(
            "Fall assist arm motion: extend lift=%.2f gripper=%.2f velocity=(%.2f, %.2f), "
            "hold=%.1fs then retract lift=%.2f gripper=%.2f",
            self.fall_assist_arm_extend_lift,
            self.fall_assist_arm_extend_gripper,
            self.fall_assist_arm_lift_velocity,
            self.fall_assist_arm_gripper_velocity,
            self.fall_assist_arm_extend_wait + self.fall_assist_arm_hold_seconds,
            self.fall_assist_arm_retract_lift,
            self.fall_assist_arm_retract_gripper,
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
        if self.fall_assist_arm_completion_wait > 0.0:
            rospy.sleep(self.fall_assist_arm_completion_wait)
        return True

    def parse_electrical_switch_command(self, text):
        normalized = self.clean_text(text)
        normalized = (
            normalized.replace("開啟", "开启")
            .replace("關閉", "关闭")
            .replace("開着", "开着")
            .replace("關着", "关着")
            .replace("開", "开")
            .replace("關", "关")
        )
        if any(token in normalized for token in ("关闭", "关掉", "关上", "关了")):
            return "off"
        if any(token in normalized for token in ("开启", "打开", "开起", "开了")):
            return "on"
        return "unknown"

    def wait_for_electrical_switch_instruction(self):
        if not self.electrical_switch_instruction_enabled:
            return "unknown"

        self.stop_base()
        self.speak(self.electrical_switch_prompt, wait=True)
        action = "unknown"
        with self.name_condition:
            self.latest_switch_answer = None
            self.accepting_switch_answer = True
            try:
                while not rospy.is_shutdown():
                    while self.latest_switch_answer is None and not rospy.is_shutdown():
                        self.name_condition.wait(timeout=0.2)
                        self.update_yolo_window("等待主人开关指令")
                    answer = self.latest_switch_answer or ""
                    self.latest_switch_answer = None
                    action = self.parse_electrical_switch_command(answer)
                    rospy.loginfo(
                        "Electrical switch ASR: text=%s action=%s",
                        answer,
                        action,
                    )
                    if action in ("on", "off"):
                        break
            finally:
                self.accepting_switch_answer = False

        self.electrical_switch_state = action
        if self.switch_state_pub is not None:
            self.switch_state_pub.publish(String(data=action))
        if action == "on":
            self.speak(self.electrical_switch_reply_on, wait=True)
        elif action == "off":
            self.speak(self.electrical_switch_reply_off, wait=True)
        return action

    def handle_owner_action_interaction(self, action_result, owner_result):
        if not self.interaction_enabled or not action_result:
            if not action_result:
                rospy.logwarn(
                    "Skipping owner interaction because no YOLO Pose action result is available"
                )
            return

        recognizer = str(action_result.get("recognizer", "") or "").strip().lower()
        if recognizer != "yolo_pose":
            rospy.logwarn(
                "Skipping owner interaction because the action result is not from YOLO Pose: %s",
                recognizer or "unknown",
            )
            return

        action = str(action_result.get("action", "unknown") or "unknown").strip().lower()
        place = str(action_result.get("place", "unknown") or "unknown").strip().lower()
        if action == "lying" and place == "floor":
            action = "fallen"
        if action not in {
            "waving",
            "sitting",
            "lying",
            "fallen",
            "sudden_fall",
            "falling",
            "lying_ground",
        }:
            rospy.logwarn(
                "Skipping owner interaction because YOLO Pose returned no actionable action: %s",
                action or "unknown",
            )
            return
        self.publish_status(
            "owner_interaction_started",
            owner_name=(owner_result or {}).get("owner_name", self.owner_name),
            action=action,
            place=place,
        )

        if action == "waving":
            approached = self.approach_owner(
                owner_result,
                self.waving_standoff_distance,
                waving=True,
            )
            if approached:
                self.handle_owner_help(owner_result)
            else:
                self.speak(self.approach_failed_prompt, wait=True)
        elif action in ("sitting", "lying"):
            self.speak("我马上靠近您。", wait=True)
            self.approach_owner(owner_result, self.approach_standoff_distance)
            self.wait_for_electrical_switch_instruction()
        elif action in ("fallen", "sudden_fall", "falling", "lying_ground"):
            self.speak(self.fall_prompt, wait=True)
            approached = False
            try:
                approached = self.approach_fallen_owner(owner_result)
            except Exception as exc:
                self.last_approach_failure_reason = "fall approach exception: %s" % exc
                rospy.logerr(self.last_approach_failure_reason)
            finally:
                if not approached:
                    rospy.logwarn(
                        "Proceeding to fall assist after incomplete fall approach: %s",
                        self.last_approach_failure_reason,
                    )
                self.perform_fall_assist_arm_motion()

        self.publish_status(
            "owner_interaction_finished",
            owner_name=(owner_result or {}).get("owner_name", self.owner_name),
            action=action,
            place=place,
            electrical_switch_state=self.electrical_switch_state,
        )

    def odom_callback(self, message):
        orientation = message.pose.pose.orientation
        siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        position = message.pose.pose.position
        linear = message.twist.twist.linear
        linear_speed = math.hypot(float(linear.x), float(linear.y))
        with self.lock:
            self.latest_yaw = yaw
            self.latest_odom_xy = (float(position.x), float(position.y))
            self.latest_odom_linear_speed = linear_speed
            self.latest_odom_time = time.time()

    def init_yolo_window(self):
        if not self.show_yolo_window:
            return
        if not os.environ.get("DISPLAY"):
            if not self.yolo_window_failed:
                rospy.logwarn("DISPLAY is not set; YOLO debug window will not be shown")
            self.yolo_window_failed = True
            return

        now = time.time()
        if now - self.yolo_window_last_retry < self.yolo_window_retry_seconds:
            return
        self.yolo_window_last_retry = now

        with self.yolo_window_lock:
            if self.yolo_window_ready:
                return
            self.yolo_window_failed = False
            try:
                if hasattr(cv2, "startWindowThread"):
                    cv2.startWindowThread()
                cv2.namedWindow(self.yolo_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.yolo_window_name, 960, 540)
                self.yolo_window_ready = True
                self.yolo_window_last_check = now
                rospy.loginfo("YOLO debug window enabled: %s", self.yolo_window_name)
            except Exception as exc:
                self.yolo_window_ready = False
                self.yolo_window_failed = True
                rospy.logwarn_throttle(2.0, "Unable to create YOLO debug window: %s", exc)

    def window_is_valid(self):
        if not self.yolo_window_ready:
            return False
        try:
            return cv2.getWindowProperty(self.yolo_window_name, cv2.WND_PROP_VISIBLE) >= 0.0
        except Exception:
            return False

    def close_yolo_window(self):
        with self.yolo_window_lock:
            if not self.yolo_window_ready:
                return
            try:
                cv2.destroyWindow(self.yolo_window_name)
            except Exception:
                pass
            self.yolo_window_ready = False

    def update_yolo_window(self, status_text=""):
        if not self.yolo_window_ready:
            self.init_yolo_window()
        if not self.yolo_window_ready:
            return
        with self.lock:
            if self.latest_image is None:
                return
            image = self.latest_image.copy()
            detections = list(self.latest_detections)

        height, width = image.shape[:2]
        for detection in detections:
            label = str(getattr(detection, "class_name", "person") or "person")
            score = float(getattr(detection, "score", 0.0) or 0.0)
            x1 = max(0, min(width - 1, int(getattr(detection, "xmin", 0))))
            y1 = max(0, min(height - 1, int(getattr(detection, "ymin", 0))))
            x2 = max(0, min(width - 1, int(getattr(detection, "xmax", 0))))
            y2 = max(0, min(height - 1, int(getattr(detection, "ymax", 0))))
            if x2 <= x1 or y2 <= y1:
                continue
            color = (0, 220, 0) if label.lower() == "person" else (0, 180, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = "%s %.2f" % (label, score)
            cv2.putText(
                image,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        header = "YOLO: %d persons" % sum(
            1
            for detection in detections
            if str(getattr(detection, "class_name", "person") or "person").lower() == "person"
        )
        if status_text:
            header += " | " + status_text
        cv2.putText(
            image,
            header,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        with self.yolo_window_lock:
            try:
                now = time.time()
                if now - self.yolo_window_last_check >= 1.0:
                    if not self.window_is_valid():
                        self.yolo_window_ready = False
                        self.yolo_window_failed = True
                        return
                    self.yolo_window_last_check = now
                cv2.imshow(self.yolo_window_name, image)
                key = (
                    cv2.pollKey()
                    if hasattr(cv2, "pollKey")
                    else cv2.waitKey(1)
                ) & 0xFF
                if key in (27, ord("q")):
                    rospy.signal_shutdown("YOLO debug window closed by user")
            except Exception as exc:
                rospy.logwarn_throttle(2.0, "YOLO debug window update failed: %s", exc)
                self.yolo_window_ready = False
                self.yolo_window_failed = True
        if not self.yolo_window_ready:
            self.init_yolo_window()

    def get_latest_yaw(self):
        with self.lock:
            return self.latest_yaw

    def stop_base(self):
        self.cmd_pub.publish(Twist())

    def stop_navigation_for_interaction(self):
        terminal_states = (
            GoalStatus.PREEMPTED,
            GoalStatus.SUCCEEDED,
            GoalStatus.ABORTED,
            GoalStatus.REJECTED,
            GoalStatus.RECALLED,
            GoalStatus.LOST,
        )
        started = time.time()
        minimum_stop_time = 0.6
        cancel_timeout = 2.0
        try:
            self.move_base.cancel_all_goals()
        except Exception as exc:
            rospy.logwarn("Could not cancel move_base goals before owner help: %s", exc)

        while not rospy.is_shutdown() and time.time() - started < cancel_timeout:
            self.stop_base()
            try:
                state = self.move_base.get_state()
            except Exception:
                state = GoalStatus.LOST
            elapsed = time.time() - started
            if elapsed >= minimum_stop_time and state in terminal_states:
                break
            rospy.sleep(0.05)
        self.stop_base()

    def load_waypoint_pose(self, waypoint_name=None):
        target_name = str(waypoint_name or self.waypoint_name).strip()
        if not os.path.exists(self.waypoint_file):
            raise RuntimeError("waypoint file not found: %s" % self.waypoint_file)
        root = ET.parse(self.waypoint_file).getroot()
        for waypoint in root.findall("Waypoint"):
            if waypoint.findtext("Name", "").strip() != target_name:
                continue
            return {
                "x": float(waypoint.findtext("Pos_x", "0")),
                "y": float(waypoint.findtext("Pos_y", "0")),
                "z": float(waypoint.findtext("Ori_z", "0")),
                "w": float(waypoint.findtext("Ori_w", "1")),
            }
        raise RuntimeError("waypoint not found: %s in %s" % (target_name, self.waypoint_file))

    def clear_move_base_costmaps(self, reason):
        if not self.clear_costmaps_before_navigation:
            return
        try:
            rospy.wait_for_service(self.clear_costmaps_service, timeout=self.clear_costmaps_timeout)
            clear_costmaps = rospy.ServiceProxy(self.clear_costmaps_service, Empty)
            clear_costmaps()
            rospy.loginfo("Cleared move_base costmaps: %s", reason)
        except Exception as exc:
            rospy.logwarn("Unable to clear move_base costmaps %s: %s", reason, exc)

    def navigate_to_waypoint(self, waypoint_name=None):
        target_name = str(waypoint_name or self.waypoint_name).strip()
        if not self.navigate_enabled:
            rospy.loginfo("Navigation disabled; assuming robot is already at %s", target_name)
            return True

        rospy.loginfo("Waiting for move_base action server")
        if not self.move_base.wait_for_server(rospy.Duration(self.move_base_server_timeout)):
            raise RuntimeError("move_base action server is not available")

        pose = self.load_waypoint_pose(target_name)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = pose["x"]
        goal.target_pose.pose.position.y = pose["y"]
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation.z = pose["z"]
        goal.target_pose.pose.orientation.w = pose["w"]

        self.clear_move_base_costmaps("before navigating to %s" % target_name)
        self.move_base.send_goal(goal)
        deadline = time.time() + self.navigate_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            self.update_yolo_window("导航到%s" % target_name)
            if self.move_base.wait_for_result(rospy.Duration(0.2)):
                break

        if rospy.is_shutdown():
            self.move_base.cancel_goal()
            self.stop_base()
            return False
        if not self.move_base.wait_for_result(rospy.Duration(0.0)):
            self.move_base.cancel_goal()
            self.stop_base()
            raise RuntimeError("move_base timed out while navigating to %s" % target_name)

        state = self.move_base.get_state()
        if state != GoalStatus.SUCCEEDED:
            self.stop_base()
            raise RuntimeError("move_base failed with state %s" % state)

        self.stop_base()
        rospy.loginfo("Arrived at waypoint %s", target_name)
        return True

    @staticmethod
    def candidate_center_norm(candidate, image_width):
        bbox = candidate.get("bbox", [0, 0, image_width, 1])
        return ((float(bbox[0]) + float(bbox[2])) * 0.5) / max(1.0, float(image_width))

    def center_owner_in_camera(self, owner_result):
        if not self.center_owner_enabled:
            return True

        initial_candidate = owner_result.get("candidate", {}) if owner_result else {}
        image, _detections = self.snapshot()
        if image is None:
            return False
        image_width = image.shape[1]
        last_center = self.candidate_center_norm(initial_candidate, image_width)
        latest_candidate = dict(initial_candidate)
        deadline = time.time() + self.center_owner_timeout
        centered_count = 0
        rate = rospy.Rate(12)

        while not rospy.is_shutdown() and time.time() < deadline:
            image, detections = self.snapshot()
            if image is None:
                rate.sleep()
                continue
            candidates = self.person_candidates(image, detections)
            if candidates:
                candidate = min(
                    candidates,
                    key=lambda item: abs(self.candidate_center_norm(item, image.shape[1]) - last_center),
                )
                latest_candidate = dict(candidate)
                last_center = self.candidate_center_norm(candidate, image.shape[1])
                x_error = last_center - 0.5
                if abs(x_error) <= self.center_owner_tolerance:
                    self.stop_base()
                    centered_count += 1
                    self.update_yolo_window("主人已居中")
                    if centered_count >= 3:
                        if owner_result is not None:
                            updated_candidate = dict(owner_result.get("candidate", {}))
                            updated_candidate.update(latest_candidate)
                            owner_result["candidate"] = updated_candidate
                        rospy.loginfo(
                            "Updated action ROI after centering: bbox=%s",
                            latest_candidate.get("bbox"),
                        )
                        rospy.loginfo("Owner centered in camera: center=%.3f", last_center)
                        return True
                else:
                    centered_count = 0
                    twist = Twist()
                    twist.angular.z = max(
                        -self.center_owner_max_angular_speed,
                        min(
                            self.center_owner_max_angular_speed,
                            -self.center_owner_angular_gain * x_error,
                        ),
                    )
                    self.cmd_pub.publish(twist)
                    self.update_yolo_window("主人居中中")
            else:
                centered_count = 0
                direction = 1.0 if last_center > 0.5 else -1.0
                twist = Twist()
                twist.angular.z = -direction * self.center_owner_lost_turn_speed
                self.cmd_pub.publish(twist)
                self.update_yolo_window("重新寻找主人")
            rate.sleep()

        self.stop_base()
        if owner_result is not None and latest_candidate:
            updated_candidate = dict(owner_result.get("candidate", {}))
            updated_candidate.update(latest_candidate)
            owner_result["candidate"] = updated_candidate
        rospy.logwarn("Owner centering timed out")
        return False

    def scan_for_owner(self):
        rospy.sleep(self.scan_after_arrival_delay)
        start_yaw = self.get_latest_yaw()
        deadline = time.time() + self.scan_timeout
        last_yaw = start_yaw
        rotated_angle = 0.0
        last_eval_time = 0.0
        last_match_owner_index = None
        match_consecutive_count = 0
        step_duration = self.scan_step_angle / max(abs(self.scan_angular_speed), 1e-3)
        moving = True
        phase_deadline = time.time() + step_duration
        rate = rospy.Rate(10)

        rospy.loginfo(
            "Scanning for owner: total_angle=%.2f rad step_angle=%.2f rad "
            "angular_speed=%.2f rad/s pause=%.2fs",
            self.scan_total_angle,
            self.scan_step_angle,
            self.scan_angular_speed,
            self.scan_pause,
        )
        while not rospy.is_shutdown() and time.time() < deadline:
            now = time.time()
            if moving:
                twist = Twist()
                twist.angular.z = self.scan_angular_speed
                self.cmd_pub.publish(twist)
                self.update_yolo_window("小步旋转寻找主人")
            else:
                self.stop_base()
                self.update_yolo_window("停顿观察画面")

            current_yaw = self.get_latest_yaw()
            if start_yaw is not None and current_yaw is not None and last_yaw is not None:
                delta = math.atan2(
                    math.sin(current_yaw - last_yaw),
                    math.cos(current_yaw - last_yaw),
                )
                if abs(delta) < 0.7:
                    rotated_angle += abs(delta)
                last_yaw = current_yaw

            now = time.time()
            if now - last_eval_time >= self.match_check_interval:
                result = self.evaluate_current_frame()
                last_eval_time = now
                matched = result is not None and self.result_is_match(result, self.match_threshold)
                if matched:
                    owner_index = result.get("owner_index")
                    required_consecutive = result.get(
                        "required_consecutive",
                        self.match_required_consecutive,
                    )
                    if last_match_owner_index == owner_index:
                        match_consecutive_count += 1
                    else:
                        match_consecutive_count = 1
                    last_match_owner_index = owner_index
                    self.publish_status(
                        "scan_match_score",
                        owner_index=owner_index,
                        name=result.get("owner_name", ""),
                        score=result.get("score"),
                        identity_score=result.get("identity_score"),
                        reid_score=result.get("reid_score"),
                        face_score=result.get("face_score"),
                        owner_score_margin=result.get("owner_score_margin"),
                        consecutive=match_consecutive_count,
                        required_consecutive=required_consecutive,
                    )
                    if match_consecutive_count >= required_consecutive:
                        self.stop_base()
                        rospy.sleep(0.2)
                        self.center_owner_in_camera(result)
                        self.announce_owner_result(result)
                        action_result = self.run_owner_action_recognition(result)
                        self.handle_owner_action_interaction(action_result, result)
                        return result
                else:
                    match_consecutive_count = 0
                    last_match_owner_index = None

            if start_yaw is not None and rotated_angle >= self.scan_total_angle:
                break

            now = time.time()
            if now >= phase_deadline:
                if moving:
                    self.stop_base()
                    moving = False
                    phase_deadline = now + self.scan_pause
                else:
                    moving = True
                    phase_deadline = now + step_duration
            rate.sleep()

        self.stop_base()
        if start_yaw is not None:
            rospy.loginfo("Owner scan completed: rotated %.2f rad", rotated_angle)
        else:
            rospy.logwarn("No odom yaw received on %s; scan ended by timeout", self.odom_topic)
        self.speak(self.owner_not_found_text, wait=True)
        self.publish_status("owner_not_found", rotated_angle=rotated_angle)
        return None

    @staticmethod
    def clean_text(text):
        return re.sub(r"\s+", "", text.strip().strip("，。！？、,.!?;；:： "))

    @staticmethod
    def parse_float_list(value):
        if isinstance(value, str):
            value = value.strip().strip("[]")
            raw_values = value.replace(";", ",").split(",") if value else []
        elif isinstance(value, (list, tuple)):
            raw_values = value
        else:
            raw_values = [value]
        parsed = []
        for item in raw_values:
            try:
                number = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                parsed.append(number)
        return parsed

    @staticmethod
    def clamp_value(value, low, high):
        return max(low, min(high, value))

    @staticmethod
    def normalize_embedding(embedding):
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6 or not np.isfinite(norm):
            return None
        return vector / norm

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

    def init_face_recognizer(self):
        if not self.face_verify_enabled:
            rospy.loginfo("InsightFace fusion disabled")
            return

        model_dir = os.path.join(self.face_model_root, "models", self.face_model_name)
        cached_models = []
        if os.path.isdir(model_dir):
            cached_models = [name for name in os.listdir(model_dir) if name.endswith(".onnx")]
        if not cached_models and not self.face_auto_download:
            rospy.logwarn(
                "InsightFace model %s is not cached under %s. Face fusion will be disabled.",
                self.face_model_name,
                model_dir,
            )
            return

        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            rospy.logwarn("InsightFace is not available: %s. Face fusion will be disabled.", exc)
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
            rospy.loginfo(
                "InsightFace fusion ready: model=%s ctx_id=%d det_size=%d threshold=%.2f",
                self.face_model_name,
                self.face_ctx_id,
                self.face_det_size,
                self.face_accept_threshold,
            )
        except Exception as exc:
            self.face_app = None
            self.face_model_ready = False
            rospy.logwarn("Failed to initialize InsightFace fusion: %s", exc)

    def candidate_face_regions(self, crop, candidate=None):
        if crop is None or crop.size == 0:
            return []
        height, width = crop.shape[:2]
        top_ratio = self.clamp_value(float(self.face_crop_top_ratio), 0.35, 1.0)
        padding = self.clamp_value(float(self.face_crop_padding), 0.0, 0.40)

        upper_y2 = max(1, int(height * top_ratio))
        pad_x = int(width * padding)
        x1 = int(self.clamp_value(0 - pad_x, 0, width - 1))
        x2 = int(self.clamp_value(width + pad_x, x1 + 1, width))
        regions = [
            ("upper", crop[0:upper_y2, x1:x2]),
            ("full", crop),
        ]
        return [(name, image) for name, image in regions if image is not None and image.size > 0]

    def extract_face_embedding(self, crop, candidate=None):
        if not self.face_model_ready or self.face_app is None:
            return None, 0.0, "face unavailable"

        best_embedding = None
        best_area = 0.0
        best_region = ""
        face_count = 0
        elapsed_ms = 0.0
        for region_name, face_image in self.candidate_face_regions(crop, candidate=candidate):
            try:
                start = time.time()
                faces = self.face_app.get(face_image)
                elapsed_ms += (time.time() - start) * 1000.0
            except Exception as exc:
                rospy.logwarn("InsightFace failed on %s crop: %s", region_name, exc)
                continue

            face = self.select_largest_face(faces)
            if face is None:
                continue
            face_count += len(faces) if faces is not None else 0
            embedding = self.normalize_embedding(face.embedding)
            if embedding is None:
                continue
            bbox = getattr(face, "bbox", None)
            if bbox is not None and len(bbox) >= 4:
                area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(
                    0.0,
                    float(bbox[3]) - float(bbox[1]),
                )
            else:
                area = 1.0
            if area > best_area:
                best_embedding = embedding
                best_area = area
                best_region = region_name
            if best_embedding is not None:
                break

        if best_embedding is None:
            return None, elapsed_ms, "no face"
        return {
            "embedding": best_embedding,
            "region": best_region,
            "face_count": face_count,
        }, elapsed_ms, "face ready"

    def parse_owner_name(self, answer):
        text = self.clean_text(answer)
        match = re.match(r"^(?:主人)?姓名(?:是|叫|为|:|：)?(.+)$", text)
        if not match:
            return ""
        name = match.group(1).strip("，。！？、,.!?;；:： ")
        name = re.split(r"[，。！？、,.!?;；:：]", name, maxsplit=1)[0]
        name = re.sub(r"^(?:是|叫|为)", "", name).strip()
        name = self.clean_text(name)
        if not name:
            return ""
        return name[: self.max_name_length]

    def is_skip_owner_answer(self, answer):
        return self.clean_text(answer) in ("跳过", "跳過")

    @staticmethod
    def format_with_name(template, name):
        if "%s" in template:
            return template % name
        if "{name}" in template:
            return template.format(name=name)
        return "%s%s" % (template, name)

    @staticmethod
    def owner_index_label(owner_index):
        labels = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        if 1 <= owner_index <= len(labels):
            return labels[owner_index - 1]
        return str(owner_index)

    def format_owner_index_text(self, template, owner_index):
        label = self.owner_index_label(owner_index)
        if "%s" in template:
            return template % label
        if "%d" in template:
            return template % owner_index
        if "{index}" in template or "{label}" in template:
            return template.format(index=owner_index, label=label)
        return template

    def owner_profile_paths(self, owner_index):
        profile_root, profile_ext = os.path.splitext(self.base_profile_path)
        metadata_root, metadata_ext = os.path.splitext(self.base_metadata_path)
        if self.owner_count == 1:
            return self.base_profile_path, self.base_metadata_path
        return (
            "%s_%d%s" % (profile_root, owner_index, profile_ext or ".npz"),
            "%s_%d%s" % (metadata_root, owner_index, metadata_ext or ".json"),
        )

    def select_owner_profile_path(self, owner_index):
        self.current_owner_index = owner_index
        self.profile_path, self.metadata_path = self.owner_profile_paths(owner_index)

    def wait_for_owner_name(self, owner_index=None):
        if owner_index is None:
            prompt_text = self.name_prompt_text
        else:
            prompt_text = self.format_owner_index_text(self.indexed_name_prompt_text, owner_index)
        candidate_name = ""
        candidate_raw = ""
        self.speak(prompt_text, wait=True)

        while not rospy.is_shutdown():
            with self.name_condition:
                self.latest_name_answer = None
                self.accepting_name_answer = True
                while self.latest_name_answer is None and not rospy.is_shutdown():
                    self.name_condition.wait(timeout=0.2)
                    self.update_yolo_window(
                        "等待第%s位主人%s"
                        % (
                            self.owner_index_label(self.current_owner_index),
                            "确认姓名" if candidate_name else "姓名",
                        )
                    )
                raw_answer = self.latest_name_answer or ""
                self.latest_name_answer = None
                self.accepting_name_answer = False

            if not raw_answer:
                continue

            if not candidate_name and self.is_skip_owner_answer(raw_answer):
                if owner_index is None or owner_index <= 1:
                    self.publish_status(
                        "owner_skip_ignored",
                        owner_index=self.current_owner_index,
                        raw=raw_answer,
                        reason="first_owner_required",
                    )
                    self.speak(self.first_owner_skip_text, wait=True)
                    continue

                self.skipped_owner_indices.append(self.current_owner_index)
                payload = {
                    "event": "owner_skipped",
                    "owner_index": self.current_owner_index,
                    "raw": raw_answer,
                }
                self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
                self.publish_status(
                    "owner_skipped",
                    owner_index=self.current_owner_index,
                    raw=raw_answer,
                )
                self.speak(
                    self.format_owner_index_text(self.owner_skip_text, self.current_owner_index),
                    wait=True,
                )
                rospy.loginfo("Owner %d enrollment skipped by voice command", self.current_owner_index)
                return None

            if candidate_name:
                normalized_answer = self.clean_text(raw_answer)
                if normalized_answer in (
                    "正确",
                    "正确的",
                    "是正确的",
                    "正確",
                    "正確的",
                    "是正確的",
                ):
                    self.owner_name = candidate_name
                    payload = {
                        "event": "owner_name_recorded",
                        "owner_index": self.current_owner_index,
                        "name": candidate_name,
                        "raw": candidate_raw,
                        "confirmation": raw_answer,
                    }
                    self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
                    self.publish_status(
                        "owner_name_recorded",
                        owner_index=self.current_owner_index,
                        name=candidate_name,
                        raw=candidate_raw,
                        confirmation=raw_answer,
                    )
                    self.speak(
                        self.format_with_name(self.name_recorded_text, candidate_name),
                        wait=True,
                    )
                    rospy.loginfo(
                        "Owner %d name confirmed: %s",
                        self.current_owner_index,
                        candidate_name,
                    )
                    return candidate_name

                corrected_name = self.parse_owner_name(raw_answer)
                if corrected_name:
                    candidate_name = corrected_name
                    candidate_raw = raw_answer
                else:
                    self.publish_status(
                        "owner_name_confirmation_ignored",
                        owner_index=self.current_owner_index,
                        raw=raw_answer,
                        pending_name=candidate_name,
                    )
                    self.speak(
                        self.format_with_name(self.name_confirm_prompt_text, candidate_name),
                        wait=True,
                    )
                    continue

            else:
                candidate_name = self.parse_owner_name(raw_answer)
                candidate_raw = raw_answer if candidate_name else ""
                if not candidate_name:
                    self.publish_status("owner_name_ignored", raw=raw_answer)
                    self.speak(self.name_invalid_text, wait=True)
                    continue

            self.speak(
                self.format_with_name(self.name_confirm_prompt_text, candidate_name),
                wait=True,
            )

        self.speak(self.name_failed_text, wait=True)
        return ""

    def save_crop(self, crop, directory, prefix, index):
        path = super().save_crop(crop, directory, prefix, index)
        if (
            prefix.startswith("owner")
            and self.face_model_ready
            and self.current_record_pose_id in ("face_front_to_side", "front_to_side")
        ):
            face_data, elapsed_ms, reason = self.extract_face_embedding(crop)
            if face_data is None:
                rospy.logwarn(
                    "No usable face embedding for owner %d %s sample %d: %s elapsed=%.1fms",
                    self.current_owner_index,
                    self.current_record_pose_name or prefix,
                    index,
                    reason,
                    elapsed_ms,
                )
                return path
            self.current_record_face_embeddings.append(
                {
                    "sample": index,
                    "pose": self.current_record_pose_id,
                    "pose_name": self.current_record_pose_name,
                    "image": path,
                    "region": face_data.get("region", ""),
                    "face_count": face_data.get("face_count", 0),
                    "embedding": face_data["embedding"],
                }
            )
            self.publish_status(
                "face_recording_sample",
                owner_index=self.current_owner_index,
                sample=index,
                pose=self.current_record_pose_id,
                pose_name=self.current_record_pose_name,
                region=face_data.get("region", ""),
                faces=face_data.get("face_count", 0),
                elapsed_ms=elapsed_ms,
            )
        return path

    def record_owner_pose(
        self,
        pose_id,
        pose_name,
        prompt_text,
        sample_offset,
        duration=None,
        sample_count=None,
        sample_interval=None,
    ):
        self.current_record_pose_id = pose_id
        self.current_record_pose_name = pose_name
        self.publish_status(
            "recording_pose_started",
            owner_index=self.current_owner_index,
            name=self.owner_name,
            pose=pose_id,
            pose_name=pose_name,
        )
        self.play_ding()
        self.speak(prompt_text, wait=True)

        embeddings = []
        color_embeddings = []
        sample_meta = []
        sample_dir = self.profile_sample_dir()
        capture_duration = self.record_seconds if duration is None else max(0.5, float(duration))
        target_sample_count = self.record_sample_count if sample_count is None else max(1, int(sample_count))
        capture_interval = (
            self.record_sample_interval
            if sample_interval is None
            else max(0.05, float(sample_interval))
        )
        deadline = time.time() + capture_duration
        next_sample_time = 0.0
        rate = rospy.Rate(30)
        while not rospy.is_shutdown() and time.time() < deadline and len(embeddings) < target_sample_count:
            now = time.time()
            if now < next_sample_time:
                self.update_yolo_window("登记%s" % pose_name)
                rate.sleep()
                continue
            next_sample_time = now + capture_interval
            image, detections = self.snapshot()
            if image is None:
                rospy.logwarn_throttle(1.0, "Waiting for fresh camera image on %s", self.image_topic)
                self.update_yolo_window("等待相机")
                rate.sleep()
                continue
            candidates = self.person_candidates(image, detections)
            if not candidates:
                rospy.logwarn_throttle(1.0, "Waiting for person detection on %s", self.detections_topic)
                self.update_yolo_window("等待YOLO人体框")
                rate.sleep()
                continue
            crop, crop_bbox = self.crop_candidate(image, candidates[0])
            if crop is None:
                rate.sleep()
                continue
            extracted = self.extract_embeddings([crop])
            if not extracted:
                rate.sleep()
                continue
            pose_sample_index = len(embeddings) + 1
            sample_index = sample_offset + pose_sample_index
            path = self.save_crop(crop, sample_dir, "owner_%s" % pose_id, sample_index)
            color_embedding = self.color_hist_embedding(crop)
            if color_embedding is not None:
                color_embeddings.append(color_embedding)
            embeddings.append(extracted[0])
            sample_meta.append(
                {
                    "index": sample_index,
                    "pose": pose_id,
                    "pose_name": pose_name,
                    "pose_sample": pose_sample_index,
                    "image": path,
                    "bbox": candidates[0]["bbox"],
                    "crop_bbox": crop_bbox,
                    "det_score": candidates[0]["score"],
                    "area_ratio": candidates[0]["area_ratio"],
                }
            )
            self.publish_status(
                "recording_sample",
                owner_index=self.current_owner_index,
                sample=sample_index,
                pose=pose_id,
                pose_name=pose_name,
                pose_sample=pose_sample_index,
                required=self.record_min_samples,
                target=target_sample_count,
            )
            rospy.loginfo(
                "Owner %d %s Re-ID sample %d captured: %s",
                self.current_owner_index,
                pose_name,
                pose_sample_index,
                path or "not saved",
            )
            self.update_yolo_window("登记%s" % pose_name)
            rate.sleep()

        self.current_record_pose_id = ""
        self.current_record_pose_name = ""
        self.publish_status(
            "recording_pose_done",
            owner_index=self.current_owner_index,
            pose=pose_id,
            pose_name=pose_name,
            samples=len(embeddings),
            target=target_sample_count,
        )
        return embeddings, color_embeddings, sample_meta

    def record_owner(self):
        self.publish_status(
            "recording_started",
            owner_index=self.current_owner_index,
            name=self.owner_name,
            poses=[pose_name for _, pose_name, _ in self.owner_record_poses],
        )
        all_embeddings = []
        all_color_embeddings = []
        all_sample_meta = []
        pose_sample_counts = []
        for pose_id, pose_name, prompt_text in self.owner_record_poses:
            capture_settings = {
                "face_front_to_side": (
                    self.face_front_to_side_record_seconds,
                    self.face_front_to_side_record_sample_count,
                    self.face_front_to_side_record_sample_interval,
                ),
                "full_body_front_to_side": (
                    self.full_body_front_to_side_record_seconds,
                    self.full_body_front_to_side_record_sample_count,
                    self.full_body_front_to_side_record_sample_interval,
                ),
            }.get(pose_id)
            duration, sample_count, sample_interval = (
                capture_settings if capture_settings is not None else (None, None, None)
            )
            embeddings, color_embeddings, sample_meta = self.record_owner_pose(
                pose_id,
                pose_name,
                prompt_text,
                len(all_embeddings),
                duration=duration,
                sample_count=sample_count,
                sample_interval=sample_interval,
            )
            all_embeddings.extend(embeddings)
            all_color_embeddings.extend(color_embeddings)
            all_sample_meta.extend(sample_meta)
            pose_sample_counts.append(
                {
                    "pose": pose_id,
                    "pose_name": pose_name,
                    "samples": len(embeddings),
                }
            )

        if len(all_embeddings) < self.record_min_samples:
            self.publish_status(
                "recording_failed",
                owner_index=self.current_owner_index,
                samples=len(all_embeddings),
                required=self.record_min_samples,
                pose_samples=pose_sample_counts,
            )
            self.speak(self.record_failed_text, wait=True)
            raise RuntimeError(
                "only captured %d/%d usable owner Re-ID samples"
                % (len(all_embeddings), self.record_min_samples)
            )

        self.save_owner_profile(
            all_embeddings,
            all_sample_meta,
            color_embeddings=all_color_embeddings,
        )
        self.speak(self.record_done_text, wait=True)
        self.publish_status(
            "recording_done",
            owner_index=self.current_owner_index,
            samples=len(all_embeddings),
            pose_samples=pose_sample_counts,
            poses=[pose_id for pose_id, _, _ in self.owner_record_poses],
        )

    def remember_current_owner_profile(self):
        profile = {
            "index": self.current_owner_index,
            "name": self.owner_name,
            "embedding": self.owner_embedding,
            "embedding_bank": self.owner_embedding_bank,
            "color_embedding": self.owner_color_embedding,
            "face_embedding": self.owner_face_embedding,
            "face_embedding_bank": self.owner_face_embedding_bank,
            "metadata": dict(self.owner_profile_meta),
            "profile_path": self.profile_path,
            "metadata_path": self.metadata_path,
        }
        existing_index = next(
            (index for index, item in enumerate(self.owner_profiles) if item["index"] == self.current_owner_index),
            None,
        )
        if existing_index is None:
            self.owner_profiles.append(profile)
        else:
            self.owner_profiles[existing_index] = profile
        self.owner_profiles.sort(key=lambda item: item["index"])

    def record_all_owners(self):
        self.owner_profiles = []
        self.skipped_owner_indices = []
        for owner_index in range(1, self.owner_count + 1):
            if rospy.is_shutdown():
                return
            self.select_owner_profile_path(owner_index)
            self.owner_name = ""
            self.owner_embedding = None
            self.owner_embedding_bank = None
            self.owner_color_embedding = None
            self.owner_face_embedding = None
            self.owner_face_embedding_bank = None
            self.current_record_face_embeddings = []
            self.owner_profile_meta = {}
            owner_name = self.wait_for_owner_name(owner_index)
            if owner_name is None:
                continue
            if not owner_name:
                return
            self.record_owner()
            self.remember_current_owner_profile()
        if not rospy.is_shutdown():
            self.face_ready = any(profile.get("face_embedding_bank") is not None for profile in self.owner_profiles)
            self.speak(self.all_profiles_recorded_text, wait=True)
            self.publish_status(
                "all_profiles_recorded",
                count=len(self.owner_profiles),
                registered_count=len(self.owner_profiles),
                skipped_count=len(self.skipped_owner_indices),
                skipped_owner_indices=list(self.skipped_owner_indices),
                face_ready=self.face_ready,
            )

    def load_all_owner_profiles(self):
        loaded_profiles = []
        self.owner_profiles = []
        original_owner_name = self.owner_name
        owner_indices = (
            [self.reuse_owner_index]
            if self.reuse_existing_profile
            else list(range(1, self.owner_count + 1))
        )
        for owner_index in owner_indices:
            self.select_owner_profile_path(owner_index)
            missing_paths = [
                path
                for path in (self.profile_path, self.metadata_path)
                if not os.path.isfile(path)
            ]
            if missing_paths:
                rospy.logerr(
                    "Owner profile %d cannot be reused; missing files: %s",
                    owner_index,
                    ", ".join(missing_paths),
                )
                self.publish_status(
                    "profile_load_failed",
                    owner_index=owner_index,
                    profile_path=self.profile_path,
                    metadata_path=self.metadata_path,
                    reason="missing profile files",
                )
                self.owner_name = original_owner_name
                self.owner_profiles = []
                return False
            self.owner_name = ""
            self.owner_embedding = None
            self.owner_embedding_bank = None
            self.owner_color_embedding = None
            self.owner_face_embedding = None
            self.owner_face_embedding_bank = None
            self.owner_profile_meta = {}
            try:
                loaded = self.load_owner_profile()
            except Exception as exc:
                rospy.logerr(
                    "Owner profile %d cannot be reused: profile=%s metadata=%s error=%s",
                    owner_index,
                    self.profile_path,
                    self.metadata_path,
                    exc,
                )
                self.publish_status(
                    "profile_load_failed",
                    owner_index=owner_index,
                    profile_path=self.profile_path,
                    metadata_path=self.metadata_path,
                    reason=str(exc),
                )
                self.owner_name = original_owner_name
                self.owner_profiles = []
                return False
            if not loaded:
                rospy.logerr(
                    "Owner profile %d cannot be reused: %s",
                    owner_index,
                    self.profile_path,
                )
                self.owner_name = original_owner_name
                self.owner_profiles = []
                return False
            if self.face_verify_enabled and self.face_model_ready:
                if self.owner_face_embedding_bank is None:
                    rospy.logerr(
                        "Owner profile %d has no face embedding while face verification is enabled: %s",
                        owner_index,
                        self.profile_path,
                    )
                    self.owner_name = original_owner_name
                    self.owner_profiles = []
                    return False
            stored_index = self.owner_profile_meta.get("owner_index")
            try:
                stored_index = None if stored_index is None else int(stored_index)
            except (TypeError, ValueError):
                stored_index = None
            if stored_index is not None and stored_index != owner_index:
                rospy.logerr(
                    "Owner profile index mismatch: expected=%d stored=%s path=%s",
                    owner_index,
                    stored_index,
                    self.metadata_path,
                )
                self.owner_name = original_owner_name
                self.owner_profiles = []
                return False
            self.owner_name = str(
                self.owner_profile_meta.get("owner_name", "") or ""
            ).strip()
            if not self.owner_name:
                rospy.logerr(
                    "Owner profile %d has no stored owner name: %s",
                    owner_index,
                    self.metadata_path,
                )
                self.owner_profiles = []
                return False
            self.remember_current_owner_profile()
            loaded_profiles.append(self.owner_name or "主人%d" % owner_index)
        self.owner_profiles = sorted(self.owner_profiles, key=lambda item: item["index"])
        self.face_ready = any(profile.get("face_embedding_bank") is not None for profile in self.owner_profiles)
        self.publish_status(
            "all_profiles_loaded",
            count=len(self.owner_profiles),
            names=loaded_profiles,
            face_ready=self.face_ready,
            owner_indices=owner_indices,
        )
        rospy.loginfo("Loaded %d owner profiles: %s", len(self.owner_profiles), ", ".join(loaded_profiles))
        return len(self.owner_profiles) == len(owner_indices)

    def save_owner_profile(self, embeddings, sample_meta, color_embeddings=None):
        super().save_owner_profile(embeddings, sample_meta, color_embeddings=color_embeddings)
        face_embeddings = [item["embedding"] for item in self.current_record_face_embeddings]
        if face_embeddings:
            mean_face_embedding = self.normalize_embedding(np.mean(np.vstack(face_embeddings), axis=0))
            if mean_face_embedding is not None:
                face_bank = [mean_face_embedding]
                face_bank.extend(face_embeddings)
                self.owner_face_embedding = mean_face_embedding
                self.owner_face_embedding_bank = np.vstack(face_bank).astype(np.float32)
                with np.load(self.profile_path, allow_pickle=False) as profile_file:
                    npz_payload = {name: profile_file[name] for name in profile_file.files}
                npz_payload["face_embedding"] = self.owner_face_embedding.astype(np.float32)
                npz_payload["face_embedding_bank"] = self.owner_face_embedding_bank
                self.atomic_save_npz(self.profile_path, npz_payload)
                self.owner_profile_meta["has_face_embedding"] = True
                self.owner_profile_meta["face_samples"] = [
                    {
                        "sample": item["sample"],
                        "pose": item["pose"],
                        "pose_name": item["pose_name"],
                        "image": item["image"],
                        "region": item["region"],
                        "face_count": item["face_count"],
                    }
                    for item in self.current_record_face_embeddings
                ]
        else:
            self.owner_face_embedding = None
            self.owner_face_embedding_bank = None
            self.owner_profile_meta["has_face_embedding"] = False
            self.owner_profile_meta["face_samples"] = []
            if self.face_model_ready:
                rospy.logwarn(
                    "Owner %d profile saved without face embedding; Re-ID fallback remains enabled.",
                    self.current_owner_index,
                )
        self.owner_profile_meta["record_poses"] = [
            {"pose": pose_id, "pose_name": pose_name}
            for pose_id, pose_name, _ in self.owner_record_poses
        ]
        if self.owner_name:
            self.owner_profile_meta["owner_index"] = self.current_owner_index
            self.owner_profile_meta["owner_name"] = self.owner_name
            self.atomic_save_json(self.metadata_path, self.owner_profile_meta)
            self.publish_status(
                "profile_named",
                owner_index=self.current_owner_index,
                name=self.owner_name,
                profile_path=self.profile_path,
            )

    def load_owner_profile(self):
        loaded = super().load_owner_profile()
        self.owner_face_embedding = None
        self.owner_face_embedding_bank = None
        if loaded:
            try:
                with np.load(self.profile_path, allow_pickle=False) as profile_file:
                    if "face_embedding" in profile_file.files:
                        self.owner_face_embedding = self.normalize_embedding(profile_file["face_embedding"])
                    if "face_embedding_bank" in profile_file.files:
                        bank = []
                        for vector in np.asarray(profile_file["face_embedding_bank"]):
                            normalized = self.normalize_embedding(vector)
                            if normalized is not None:
                                bank.append(normalized)
                        if bank:
                            self.owner_face_embedding_bank = np.vstack(bank).astype(np.float32)
            except Exception as exc:
                rospy.logwarn("Failed to load owner face embedding from %s: %s", self.profile_path, exc)
            self.face_ready = self.owner_face_embedding_bank is not None
            if not self.owner_name:
                self.owner_name = self.owner_profile_meta.get("owner_name", "")
        return loaded

    def owner_found_message(self):
        name = self.owner_name or self.owner_profile_meta.get("owner_name", "主人")
        return self.format_with_name(self.named_owner_found_text, name)

    def owner_reid_similarity(self, owner_profile, embedding):
        embedding_bank = owner_profile.get("embedding_bank")
        if embedding_bank is not None:
            scores = np.dot(embedding_bank, embedding)
            return float(np.max(scores))
        owner_embedding = owner_profile.get("embedding")
        if owner_embedding is None:
            return -1.0
        return float(np.dot(owner_embedding, embedding))

    def owner_color_similarity(self, owner_profile, crop):
        color_embedding = owner_profile.get("color_embedding")
        if color_embedding is None:
            return None
        query_color_embedding = self.color_hist_embedding(crop)
        if query_color_embedding is None:
            return None
        return float(np.dot(color_embedding, query_color_embedding))

    def owner_face_similarity(self, owner_profile, face_embedding):
        if face_embedding is None:
            return None
        face_bank = owner_profile.get("face_embedding_bank")
        if face_bank is None:
            face_embedding_single = owner_profile.get("face_embedding")
            if face_embedding_single is None:
                return None
            return float(np.dot(face_embedding_single, face_embedding))
        return float(np.max(np.dot(face_bank, face_embedding)))

    def score_owner_match(self, owner_profile, embedding, crop, lying_pose, face_embedding=None):
        reid_score = self.owner_reid_similarity(owner_profile, embedding)
        color_score = self.owner_color_similarity(owner_profile, crop)
        if lying_pose:
            score = self.fused_lie_score(reid_score, color_score)
            identity_score = score
        else:
            score = reid_score
        face_score = self.owner_face_similarity(owner_profile, face_embedding)
        if not lying_pose:
            identity_score = face_score if face_score is not None else -1.0
        return score, reid_score, color_score, face_score, identity_score

    def result_is_match(self, result, default_threshold):
        if result is None:
            return False
        score_margin = result.get("owner_score_margin")
        if (
            len(self.owner_profiles) > 1
            and score_margin is not None
            and float(score_margin) < self.owner_score_margin_threshold
        ):
            return False

        face_score = result.get("face_score")
        lying_pose = bool(result.get("candidate", {}).get("lying_pose", False))
        if lying_pose:
            return PersonReidOwnerTest.result_is_match(result, default_threshold)

        if face_score is None:
            return False
        return float(face_score) >= self.face_accept_threshold

    def evaluate_current_frame(self):
        if not self.owner_profiles:
            return super().evaluate_current_frame()

        image, detections = self.snapshot()
        if image is None:
            return None
        candidates = self.person_candidates(image, detections)
        if not candidates:
            return None
        query_crops = []
        records = []
        for candidate in candidates:
            crop, crop_bbox = self.crop_candidate(
                image,
                candidate,
                padding=self.crop_padding,
            )
            if crop is None:
                continue
            lying_pose = self.is_lying_candidate(candidate, crop=crop)
            if lying_pose and self.lying_crop_padding > self.crop_padding:
                crop, crop_bbox = self.crop_candidate(
                    image,
                    candidate,
                    padding=self.lying_crop_padding,
                )
                if crop is None:
                    continue
            meta = dict(candidate)
            meta["crop_bbox"] = crop_bbox
            meta["lying_pose"] = lying_pose
            variants = self.reid_query_variants(crop, lying_pose)
            query_indexes = []
            variant_names = []
            for variant_name, variant_crop in variants:
                query_indexes.append(len(query_crops))
                variant_names.append(variant_name)
                query_crops.append(variant_crop)
            records.append(
                {
                    "crop": crop,
                    "meta": meta,
                    "query_indexes": query_indexes,
                    "variant_names": variant_names,
                }
            )
        if not query_crops:
            return None
        embeddings = self.extract_embeddings(query_crops)
        if len(embeddings) != len(query_crops):
            return None

        results = []
        face_profiles_ready = any(profile.get("face_embedding_bank") is not None for profile in self.owner_profiles)
        for record in records:
            lying_pose = record["meta"].get("lying_pose", False)
            face_embedding = None
            face_region = ""
            face_count = 0
            face_elapsed_ms = 0.0
            face_reason = "face disabled"
            if self.face_model_ready and face_profiles_ready:
                face_data, face_elapsed_ms, face_reason = self.extract_face_embedding(
                    record["crop"],
                    candidate=record["meta"],
                )
                if face_data is not None:
                    face_embedding = face_data["embedding"]
                    face_region = face_data.get("region", "")
                    face_count = face_data.get("face_count", 0)
            for owner_profile in self.owner_profiles:
                variant_scores = []
                for query_index, variant_name in zip(record["query_indexes"], record["variant_names"]):
                    score, reid_score, color_score, face_score, identity_score = self.score_owner_match(
                        owner_profile,
                        embeddings[query_index],
                        record["crop"],
                        lying_pose,
                        face_embedding=face_embedding,
                    )
                    variant_scores.append(
                        (identity_score, score, reid_score, color_score, face_score, variant_name)
                    )
                identity_score, score, reid_score, color_score, face_score, best_variant = max(
                    variant_scores,
                    key=lambda item: item[0],
                )
                if lying_pose:
                    match_threshold = self.lying_match_threshold
                    min_reid_score = self.lying_min_reid_score
                    required_consecutive = self.lying_required_consecutive
                else:
                    match_threshold = self.match_threshold
                    min_reid_score = -1.0
                    required_consecutive = self.match_required_consecutive
                primary_score = identity_score if not lying_pose else score
                result = {
                    "score": float(primary_score),
                    "identity_score": float(identity_score),
                    "reid_score": float(reid_score),
                    "color_score": None if color_score is None else float(color_score),
                    "face_score": None if face_score is None else float(face_score),
                    "face_region": face_region,
                    "face_count": face_count,
                    "face_elapsed_ms": face_elapsed_ms,
                    "face_reason": face_reason,
                    "match_threshold": match_threshold,
                    "min_reid_score": min_reid_score,
                    "required_consecutive": required_consecutive,
                    "best_variant": best_variant,
                    "candidate": record["meta"],
                    "crop": record["crop"],
                    "num_candidates": len(records),
                    "owner_index": owner_profile["index"],
                    "owner_name": owner_profile.get("name") or "主人%d" % owner_profile["index"],
                }
                results.append(result)
        if not results:
            return None

        owner_best_scores = {}
        for result in results:
            owner_index = result["owner_index"]
            score = result["identity_score"]
            if owner_index not in owner_best_scores or score > owner_best_scores[owner_index]:
                owner_best_scores[owner_index] = score
        ranked_owner_scores = sorted(owner_best_scores.values(), reverse=True)
        score_margin = None
        if len(ranked_owner_scores) >= 2:
            score_margin = ranked_owner_scores[0] - ranked_owner_scores[1]

        best_result = max(results, key=lambda item: item["identity_score"])
        best_result["owner_score_margin"] = score_margin
        return best_result

    def announce_owner_result(self, result):
        score = result["score"]
        owner_index = result.get("owner_index", self.current_owner_index)
        owner_name = result.get("owner_name", self.owner_name)
        crop_path = self.maybe_save_match_crop(result["crop"], score)
        self.owner_name = owner_name
        message = self.format_with_name(self.named_owner_found_text, owner_name or "主人")
        rospy.loginfo(
            "Owner recognized: index=%s name=%s score=%.3f crop=%s",
            owner_index,
            owner_name or "unknown",
            score,
            crop_path or "not saved",
        )
        self.speak(message, wait=False)
        payload = {
            "event": "owner_recognized",
            "owner_index": owner_index,
            "name": owner_name,
            "score": score,
            "identity_score": result.get("identity_score", score),
            "reid_score": result.get("reid_score", score),
            "color_score": result.get("color_score"),
            "face_score": result.get("face_score"),
            "face_region": result.get("face_region", ""),
            "face_count": result.get("face_count", 0),
            "owner_score_margin": result.get("owner_score_margin"),
            "lying_pose": result.get("candidate", {}).get("lying_pose", False),
            "best_variant": result.get("best_variant", "raw"),
            "crop": crop_path,
        }
        self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.publish_status(
            "owner_recognized",
            owner_index=owner_index,
            name=owner_name,
            score=score,
            identity_score=result.get("identity_score", score),
            reid_score=result.get("reid_score", score),
            color_score=result.get("color_score"),
            face_score=result.get("face_score"),
            face_region=result.get("face_region", ""),
            face_count=result.get("face_count", 0),
            owner_score_margin=result.get("owner_score_margin"),
            lying_pose=result.get("candidate", {}).get("lying_pose", False),
            best_variant=result.get("best_variant", "raw"),
            crop=crop_path,
        )
        self.last_announce_time = time.time()

    def recognition_loop(self):
        self.publish_status(
            "recognition_started",
            threshold=self.match_threshold,
            owner_count=len(self.owner_profiles) or 1,
            names=[profile.get("name", "") for profile in self.owner_profiles],
            face_model_ready=self.face_model_ready,
            face_ready=self.face_ready,
            face_weight=self.face_identity_weight,
            owner_margin_threshold=self.owner_score_margin_threshold,
        )
        rate = rospy.Rate(max(1.0, 1.0 / self.match_check_interval))
        while not rospy.is_shutdown():
            self.update_yolo_window("识别中")
            result = self.evaluate_current_frame()
            if result is None:
                self.match_consecutive_count = 0
                self.last_match_owner_index = None
                rate.sleep()
                continue

            score = result["score"]
            owner_index = result.get("owner_index", self.current_owner_index)
            owner_name = result.get("owner_name", self.owner_name)
            match_threshold = result.get("match_threshold", self.match_threshold)
            required_consecutive = result.get("required_consecutive", self.match_required_consecutive)
            matched = self.result_is_match(result, self.match_threshold)
            if matched:
                if self.last_match_owner_index == owner_index:
                    self.match_consecutive_count += 1
                else:
                    self.match_consecutive_count = 1
                self.last_match_owner_index = owner_index
            else:
                self.match_consecutive_count = 0
                self.last_match_owner_index = None
            self.publish_status(
                "match_score",
                score=score,
                identity_score=result.get("identity_score", score),
                reid_score=result.get("reid_score", score),
                color_score=result.get("color_score"),
                face_score=result.get("face_score"),
                face_region=result.get("face_region", ""),
                face_count=result.get("face_count", 0),
                face_elapsed_ms=result.get("face_elapsed_ms", 0.0),
                owner_score_margin=result.get("owner_score_margin"),
                matched=matched,
                consecutive=self.match_consecutive_count,
                threshold=match_threshold,
                required_consecutive=required_consecutive,
                lying_pose=result.get("candidate", {}).get("lying_pose", False),
                best_variant=result.get("best_variant", "raw"),
                owner_index=owner_index,
                name=owner_name,
            )

            if matched and self.match_consecutive_count >= required_consecutive:
                now = time.time()
                if now - self.last_announce_time >= self.announce_cooldown:
                    crop_path = self.maybe_save_match_crop(result["crop"], score)
                    self.owner_name = owner_name
                    message = self.format_with_name(self.named_owner_found_text, owner_name or "主人")
                    rospy.loginfo(
                        "Owner recognized: index=%s name=%s score=%.3f crop=%s",
                        owner_index,
                        owner_name or "unknown",
                        score,
                        crop_path or "not saved",
                    )
                    self.speak(message, wait=False)
                    payload = {
                        "event": "owner_recognized",
                        "owner_index": owner_index,
                        "name": owner_name,
                        "score": score,
                        "identity_score": result.get("identity_score", score),
                        "reid_score": result.get("reid_score", score),
                        "color_score": result.get("color_score"),
                        "face_score": result.get("face_score"),
                        "face_region": result.get("face_region", ""),
                        "face_count": result.get("face_count", 0),
                        "owner_score_margin": result.get("owner_score_margin"),
                        "lying_pose": result.get("candidate", {}).get("lying_pose", False),
                        "best_variant": result.get("best_variant", "raw"),
                        "crop": crop_path,
                    }
                    self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
                    self.publish_status(
                        "owner_recognized",
                        owner_index=owner_index,
                        name=owner_name,
                        score=score,
                        identity_score=result.get("identity_score", score),
                        reid_score=result.get("reid_score", score),
                        color_score=result.get("color_score"),
                        face_score=result.get("face_score"),
                        face_region=result.get("face_region", ""),
                        face_count=result.get("face_count", 0),
                        owner_score_margin=result.get("owner_score_margin"),
                        lying_pose=result.get("candidate", {}).get("lying_pose", False),
                        best_variant=result.get("best_variant", "raw"),
                        crop=crop_path,
                    )
                    self.last_announce_time = now
                    if not self.action_completed:
                        action_result = self.run_owner_action_recognition(result)
                        self.handle_owner_action_interaction(action_result, result)
                    if self.stop_after_first_match:
                        return
            rate.sleep()

    def run(self):
        self.wait_for_tts()
        if not self.wait_for_asr():
            return
        self.warmup_help_qwen()
        self.wait_for_camera_inputs()
        self.init_yolo_window()
        self.update_yolo_window("相机已连接")
        self.init_reid_backend()
        self.init_face_recognizer()
        if self.reuse_existing_profile:
            rospy.loginfo(
                "Reusing owner profile: directory=%s owner_index=%d",
                self.profile_dir,
                self.reuse_owner_index,
            )
            if not self.load_all_owner_profiles():
                raise RuntimeError(
                    "reuse_existing_profile requested, but owner profiles could not be loaded "
                    "from %s; no new profiles were recorded"
                    % self.profile_dir
                )
            rospy.loginfo("Loaded existing owner Re-ID profiles from %s", self.profile_dir)
        else:
            self.record_all_owners()
        if not self.owner_profiles:
            if self.owner_embedding is not None:
                self.remember_current_owner_profile()
            else:
                raise RuntimeError("owner profiles are not ready")
        if self.navigate_enabled:
            try:
                self.navigate_to_waypoint(self.waypoint_name)
            except Exception as exc:
                self.stop_base()
                self.speak(self.navigation_failed_text, wait=True)
                self.publish_status("navigation_failed", message=str(exc), waypoint=self.waypoint_name)
                raise
            self.scan_for_owner()
        else:
            self.recognition_loop()


def main():
    rospy.init_node("owner_voice_reid_test")
    node = OwnerVoiceReidTest()
    try:
        node.run()
    except Exception as exc:
        rospy.logerr("Owner voice Re-ID test failed: %s", exc)
        node.publish_status(
            "error",
            message=str(exc),
            owner_index=getattr(node, "current_owner_index", 0),
            name=getattr(node, "owner_name", ""),
        )
        raise


if __name__ == "__main__":
    main()
