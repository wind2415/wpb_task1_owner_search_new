#!/usr/bin/env python3
# coding=utf-8

import json
import math
import os
import sys
import threading
import time
from enum import Enum

import rospy
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from owner_voice_reid_test import OwnerVoiceReidTest


class WorkflowState(Enum):
    INITIALIZE = "initialize"
    WAIT_FOR_ASR = "wait_for_asr"
    WAIT_FOR_CAMERA = "wait_for_camera"
    PREPARE_MODELS = "prepare_models"
    REGISTER_OWNERS = "register_owners"
    NAVIGATE = "navigate"
    SEARCH = "search"
    OWNER_CONFIRMED = "owner_confirmed"
    RECOGNIZE_ACTION = "recognize_action"
    INTERACT = "interact"
    COMPLETED = "completed"
    FAILED = "failed"


class OptimizedOwnerVoiceReid(OwnerVoiceReidTest):
    """Workflow wrapper that keeps the original owner-search implementation intact."""

    def __init__(self):
        super().__init__()

        self.optimized_waving_lidar_guard_distance = max(
            0.05,
            float(
                rospy.get_param(
                    "~optimized_waving_lidar_guard_distance",
                    0.20,
                )
            ),
        )
        self.approach_debug_logging_enabled = bool(
            rospy.get_param("~approach_debug_logging_enabled", True)
        )
        self.approach_debug_tick_log_period = max(
            0.2,
            float(rospy.get_param("~approach_debug_tick_log_period", 1.0)),
        )
        self.approach_debug_sequence = 0
        self.approach_debug_started_at = None

        self.workflow_state_topic = str(
            rospy.get_param(
                "~optimized_state_topic",
                "/owner_voice_reid_optimized/state",
            )
        ).strip()
        self.workflow_status_topic = str(
            rospy.get_param(
                "~optimized_status_topic",
                "/owner_voice_reid_optimized/status",
            )
        ).strip()
        self.patrol_enabled = bool(rospy.get_param("~patrol_enabled", True))
        self.patrol_repeat = bool(rospy.get_param("~patrol_repeat", False))
        self.patrol_max_rounds = max(1, int(rospy.get_param("~patrol_max_rounds", 1)))
        self.patrol_waypoint_names = self.parse_waypoint_names(
            rospy.get_param(
                "~patrol_waypoint_names",
                ["living_room", "kitchen", "bedroom", "canteen"],
            )
        )
        if not self.patrol_waypoint_names:
            self.patrol_waypoint_names = [self.waypoint_name]
        self.exit_waypoint_name = str(
            rospy.get_param("~exit_waypoint_name", "exit")
        ).strip()
        self.return_to_exit_when_complete = bool(
            rospy.get_param("~return_to_exit_when_complete", True)
        )
        self.optimized_seated_waving_aspect_ratio = max(
            0.35,
            float(
                rospy.get_param(
                    "~optimized_seated_waving_aspect_ratio",
                    0.70,
                )
            ),
        )
        self.seated_waving_two_stage_enabled = bool(
            rospy.get_param("~seated_waving_two_stage_enabled", True)
        )
        self.seated_waving_coarse_standoff_distance = max(
            self.waving_approach_safety_radius + 0.20,
            float(
                rospy.get_param(
                    "~seated_waving_coarse_standoff_distance",
                    1.00,
                )
            ),
        )
        self.seated_waving_refine_timeout = max(
            0.5,
            float(rospy.get_param("~seated_waving_refine_timeout", 3.0)),
        )
        self.seated_waving_refine_required_matches = max(
            1,
            int(rospy.get_param("~seated_waving_refine_required_matches", 2)),
        )
        if not self.exit_waypoint_name:
            self.exit_waypoint_name = "exit"

        self.workflow_timeouts = {
            WorkflowState.INITIALIZE: 30.0,
            WorkflowState.WAIT_FOR_ASR: max(10.0, self.asr_wait_timeout + 5.0),
            WorkflowState.WAIT_FOR_CAMERA: max(10.0, self.startup_timeout + 5.0),
            WorkflowState.PREPARE_MODELS: 120.0,
            WorkflowState.REGISTER_OWNERS: max(120.0, self.owner_count * 180.0),
            WorkflowState.NAVIGATE: max(30.0, self.navigate_timeout + 10.0),
            WorkflowState.SEARCH: max(20.0, self.scan_timeout + 10.0),
            WorkflowState.OWNER_CONFIRMED: 20.0,
            WorkflowState.RECOGNIZE_ACTION: max(30.0, self.action_timeout + 10.0),
            WorkflowState.INTERACT: max(30.0, self.answer_timeout + 30.0),
            WorkflowState.COMPLETED: 30.0,
            WorkflowState.FAILED: 30.0,
        }
        configured_timeout = rospy.get_param("~optimized_state_timeout", 0.0)
        if float(configured_timeout) > 0.0:
            for state in self.workflow_timeouts:
                self.workflow_timeouts[state] = float(configured_timeout)

        self.workflow_state = WorkflowState.INITIALIZE
        self.workflow_state_started_at = time.time()
        self.workflow_state_timeout_reported = False
        self.workflow_lock = threading.RLock()
        self.workflow_state_pub = rospy.Publisher(
            self.workflow_status_topic,
            String,
            queue_size=5,
            latch=True,
        )
        self.workflow_event_pub = rospy.Publisher(
            self.workflow_state_topic,
            String,
            queue_size=10,
            latch=True,
        )
        self.workflow_watchdog = rospy.Timer(
            rospy.Duration(1.0),
            self.workflow_watchdog_callback,
        )
        self.current_patrol_waypoint = ""
        self.current_action_result = None
        self.patrol_results = []
        rospy.on_shutdown(self.stop_workflow_watchdog)

    @staticmethod
    def _approach_debug_value(value):
        if isinstance(value, dict):
            return {
                str(key): OptimizedOwnerVoiceReid._approach_debug_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                OptimizedOwnerVoiceReid._approach_debug_value(item)
                for item in value
            ]
        if isinstance(value, float):
            if not math.isfinite(value):
                return str(value)
            return round(value, 4)
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        return str(value)

    def approach_debug_event(self, event, **fields):
        if not getattr(self, "approach_debug_logging_enabled", False):
            return

        try:
            payload = {
                "event": str(event),
                "sequence": int(getattr(self, "approach_debug_sequence", 0)),
            }
            safe_fields = {
                str(key): self._approach_debug_value(value)
                for key, value in fields.items()
            }
            payload.update(safe_fields)
            message = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            is_failure = fields.get("success") is False
            if event == "navigation_tick":
                rospy.loginfo_throttle(
                    self.approach_debug_tick_log_period,
                    "[optimized waving approach] %s",
                    message,
                )
            elif is_failure:
                rospy.logwarn("[optimized waving approach] %s", message)
            else:
                rospy.loginfo("[optimized waving approach] %s", message)

            if event != "navigation_tick" and hasattr(self, "workflow_state_pub"):
                self.publish_workflow_state(
                    "approach_debug",
                    approach_event=str(event),
                    **safe_fields
                )
        except Exception as exc:
            rospy.logwarn_throttle(
                5.0,
                "Optimized approach diagnostic logging failed: %s",
                exc,
            )

    def approach_owner(
        self,
        owner_result,
        standoff_distance,
        waving=False,
        lidar_guard_distance_override=None,
    ):
        self.approach_debug_sequence += 1
        self.approach_debug_started_at = time.time()
        candidate = (
            owner_result.get("candidate", {})
            if isinstance(owner_result, dict)
            else {}
        )
        if not isinstance(candidate, dict):
            candidate = {}
        effective_lidar_guard_distance = (
            self.optimized_waving_lidar_guard_distance
            if waving
            else lidar_guard_distance_override
        )
        self.approach_debug_event(
            "approach_started",
            waving=bool(waving),
            standoff_distance=float(standoff_distance),
            owner_index=(owner_result or {}).get("owner_index")
            if isinstance(owner_result, dict)
            else None,
            owner_name=(owner_result or {}).get("owner_name", "")
            if isinstance(owner_result, dict)
            else "",
            owner_score=(owner_result or {}).get("score")
            if isinstance(owner_result, dict)
            else None,
            candidate_bbox=candidate.get("bbox"),
            candidate_aspect_ratio=candidate.get("aspect_ratio"),
            lidar_guard_distance_override=effective_lidar_guard_distance,
        )
        try:
            approached = super().approach_owner(
                owner_result,
                standoff_distance,
                waving=waving,
                lidar_guard_distance_override=effective_lidar_guard_distance,
            )
        except Exception as exc:
            self.approach_debug_event(
                "approach_exception",
                waving=bool(waving),
                elapsed_sec=time.time() - self.approach_debug_started_at,
                exception=str(exc),
                failure_reason=getattr(self, "last_approach_failure_reason", ""),
            )
            raise

        self.approach_debug_event(
            "approach_finished",
            waving=bool(waving),
            success=bool(approached),
            elapsed_sec=time.time() - self.approach_debug_started_at,
            failure_reason=getattr(self, "last_approach_failure_reason", ""),
        )
        return approached

    def estimate_owner_position_from_pointcloud(self, owner_result):
        position = super().estimate_owner_position_from_pointcloud(owner_result)
        if position is None:
            self.approach_debug_event(
                "owner_position_estimated",
                success=False,
                reason=getattr(self, "pointcloud_reason", ""),
            )
        else:
            self.approach_debug_event(
                "owner_position_estimated",
                success=True,
                position={
                    key: position.get(key)
                    for key in ("x", "y", "z", "distance", "forward", "lateral")
                    if key in position
                },
            )
        return position

    @staticmethod
    def parse_waypoint_names(value):
        if isinstance(value, str):
            raw_names = value.replace(";", ",").split(",")
        elif isinstance(value, (list, tuple)):
            raw_names = value
        else:
            raw_names = [value]
        names = []
        for raw_name in raw_names:
            name = str(raw_name or "").strip()
            if name and name not in names:
                names.append(name)
        return names

    def stop_workflow_watchdog(self):
        timer = getattr(self, "workflow_watchdog", None)
        if timer is not None:
            timer.shutdown()

    def workflow_watchdog_callback(self, _event):
        with self.workflow_lock:
            state = self.workflow_state
            elapsed = time.time() - self.workflow_state_started_at
            timeout = self.workflow_timeouts.get(state, 0.0)
            if timeout <= 0.0 or elapsed <= timeout or self.workflow_state_timeout_reported:
                return
            self.workflow_state_timeout_reported = True
        rospy.logwarn(
            "Optimized owner workflow state timed out: state=%s elapsed=%.1fs timeout=%.1fs",
            state.value,
            elapsed,
            timeout,
        )
        self.publish_workflow_state(
            "state_timeout",
            state=state.value,
            elapsed_sec=round(elapsed, 3),
            timeout_sec=timeout,
        )

    def publish_workflow_state(self, event, **fields):
        with self.workflow_lock:
            state = self.workflow_state.value
            started_at = self.workflow_state_started_at
        payload = {
            "event": event,
            "state": state,
            "time": time.time(),
            "state_elapsed_sec": round(max(0.0, time.time() - started_at), 3),
        }
        if self.current_patrol_waypoint and "waypoint" not in fields:
            fields["waypoint"] = self.current_patrol_waypoint
        payload.update(fields)
        message = String(data=json.dumps(payload, ensure_ascii=False))
        self.workflow_event_pub.publish(message)
        self.workflow_state_pub.publish(message)

    def set_workflow_state(self, state, reason="", **fields):
        if not isinstance(state, WorkflowState):
            state = WorkflowState(str(state))
        with self.workflow_lock:
            previous = self.workflow_state.value
            self.workflow_state = state
            self.workflow_state_started_at = time.time()
            self.workflow_state_timeout_reported = False
        rospy.loginfo(
            "Optimized owner workflow: %s -> %s%s",
            previous,
            state.value,
            " (%s)" % reason if reason else "",
        )
        self.publish_workflow_state(
            "state_changed",
            previous_state=previous,
            reason=reason,
            **fields
        )
        super().publish_status(
            "workflow_state_changed",
            state=state.value,
            previous_state=previous,
            reason=reason,
            **fields
        )

    def navigate_to_waypoint(self, waypoint_name=None):
        self.set_workflow_state(
            WorkflowState.NAVIGATE,
            reason="navigate_to_waypoint",
            waypoint=waypoint_name or self.waypoint_name,
        )
        return super().navigate_to_waypoint(waypoint_name)

    def scan_for_owner(self):
        self.action_completed = False
        self.action_result = None
        self.action_result_event.clear()
        self.current_action_result = None
        self.set_workflow_state(WorkflowState.SEARCH, reason="scan_for_owner")
        result = super().scan_for_owner()
        return result

    def announce_owner_result(self, result):
        self.set_workflow_state(
            WorkflowState.OWNER_CONFIRMED,
            reason="owner_match_confirmed",
            owner_index=(result or {}).get("owner_index"),
            owner_name=(result or {}).get("owner_name", ""),
        )
        return super().announce_owner_result(result)

    def run_owner_action_recognition(self, owner_result):
        self.set_workflow_state(
            WorkflowState.RECOGNIZE_ACTION,
            reason="owner_confirmed",
            owner_index=(owner_result or {}).get("owner_index"),
        )
        result = super().run_owner_action_recognition(owner_result)
        self.current_action_result = self.normalize_action_result(result)
        return self.current_action_result

    @staticmethod
    def normalize_action_result(action_result):
        if not isinstance(action_result, dict):
            return action_result
        normalized = dict(action_result)
        action = str(normalized.get("action", "unknown") or "unknown").strip().lower()
        place = str(normalized.get("place", "unknown") or "unknown").strip().lower()
        pose_action = str(
            normalized.get("pose_action", "") or ""
        ).strip().lower()
        pose_reason = str(
            normalized.get("pose_reason", "") or ""
        ).strip().lower()
        ground_relation = normalized.get("ground_relation") or {}
        ground_fallen = bool(normalized.get("ground_fallen")) or bool(
            isinstance(ground_relation, dict) and ground_relation.get("ground_fallen")
        )
        pose_fall_transition = bool(normalized.get("pose_fall_transition")) or (
            pose_action in {"fallen", "falling", "sudden_fall"}
        ) or "fall transition" in pose_reason
        if action == "lying" and place == "floor":
            action = "fallen"
        if pose_fall_transition:
            action = "sudden_fall"
            place = "floor"
        elif ground_fallen and action in {
            "unknown",
            "lying",
            "falling",
        }:
            action = "fallen"
            place = "floor"
        normalized["action"] = action
        normalized["place"] = place
        normalized["action_priority"] = (
            "fall" if action in {"fallen", "falling", "sudden_fall"} else action
        )
        return normalized

    @classmethod
    def action_result_is_certain(cls, action_result):
        if not isinstance(action_result, dict):
            return False
        recognizer = str(action_result.get("recognizer", "") or "").strip().lower()
        action = str(action_result.get("action", "unknown") or "unknown").strip().lower()
        return recognizer == "yolo_pose" and action in {
            "waving",
            "sitting",
            "lying",
            "fallen",
            "sudden_fall",
            "falling",
            "lying_ground",
        }

    def is_seated_waving(self, action_result, owner_result):
        if not isinstance(action_result, dict):
            return False
        if str(action_result.get("action", "") or "").strip().lower() != "waving":
            return False

        place = str(action_result.get("place", "") or "").strip().lower()
        if place in {"chair", "sofa", "bed"}:
            return True

        ground_relation = action_result.get("ground_relation")
        if (
            isinstance(ground_relation, dict)
            and str(ground_relation.get("status", "") or "").strip().lower()
            == "elevated"
        ):
            return True

        speech = str(action_result.get("speech", "") or "")
        if any(marker in speech for marker in ("坐", "椅子", "沙发", "床")):
            return True

        candidate = owner_result.get("candidate", {}) if isinstance(owner_result, dict) else {}
        if not isinstance(candidate, dict):
            return False
        aspect_ratio = candidate.get("aspect_ratio")
        if aspect_ratio is None:
            bbox = candidate.get("bbox")
            if not bbox or len(bbox) < 4:
                return False
            try:
                width = float(bbox[2]) - float(bbox[0])
                height = float(bbox[3]) - float(bbox[1])
                aspect_ratio = width / max(1.0, height)
            except (TypeError, ValueError):
                return False
        try:
            aspect_ratio = float(aspect_ratio)
        except (TypeError, ValueError):
            return False
        is_not_lying = not bool(candidate.get("lying_pose", False))
        seated = (
            aspect_ratio >= self.optimized_seated_waving_aspect_ratio
            and aspect_ratio < self.lying_aspect_ratio_threshold
            and is_not_lying
        )
        rospy.loginfo(
            "Optimized waving posture check: aspect=%.3f threshold=%.3f "
            "lying_pose=%s seated=%s",
            aspect_ratio,
            self.optimized_seated_waving_aspect_ratio,
            bool(candidate.get("lying_pose", False)),
            seated,
        )
        return seated

    def is_probably_seated_waving(self, action_result, owner_result):
        if self.is_seated_waving(action_result, owner_result):
            return True

        if not isinstance(action_result, dict):
            return False
        ground_relation = action_result.get("ground_relation")
        if (
            isinstance(ground_relation, dict)
            and str(ground_relation.get("status", "") or "").strip().lower()
            == "elevated"
        ):
            return True

        candidate = owner_result.get("candidate", {}) if isinstance(owner_result, dict) else {}
        if not isinstance(candidate, dict) or bool(candidate.get("lying_pose", False)):
            return False
        bbox = candidate.get("bbox")
        if not bbox or len(bbox) < 4:
            return False
        try:
            width = float(bbox[2]) - float(bbox[0])
            height = float(bbox[3]) - float(bbox[1])
            aspect_ratio = width / max(1.0, height)
        except (TypeError, ValueError):
            return False
        return 0.35 <= aspect_ratio < self.lying_aspect_ratio_threshold

    def handle_waving_two_stage_interaction(self, action_result, owner_result):
        if not self.seated_waving_two_stage_enabled:
            return self.handle_waving_interaction(action_result, owner_result)

        self.approach_debug_event(
            "waving_interaction_started",
            mode="waving_two_stage",
            place=(action_result or {}).get("place", "unknown"),
            action=(action_result or {}).get("action", "waving"),
        )
        self.publish_status(
            "owner_interaction_started",
            owner_name=(owner_result or {}).get("owner_name", self.owner_name),
            action="waving",
            place=(action_result or {}).get("place", "unknown"),
            approach_mode="waving_two_stage",
            approach_stage="coarse",
        )
        rospy.loginfo(
            "Waving approach stage 1/2: coarse standoff=%.2fm",
            self.seated_waving_coarse_standoff_distance,
        )
        coarse_approached = self.approach_owner(
            owner_result,
            self.seated_waving_coarse_standoff_distance,
            waving=False,
        )
        if not coarse_approached:
            self.speak(self.approach_failed_prompt, wait=True)
            self.approach_debug_event(
                "waving_interaction_finished",
                success=False,
                stage="coarse_failed",
                failure_reason=getattr(self, "last_approach_failure_reason", ""),
            )
            self.publish_status(
                "owner_interaction_finished",
                owner_name=(owner_result or {}).get("owner_name", self.owner_name),
                action="waving",
                place=(action_result or {}).get("place", "unknown"),
                approach_mode="waving_two_stage",
                approach_stage="coarse_failed",
                electrical_switch_state=self.electrical_switch_state,
            )
            return False

        refined_owner_result = self.refresh_owner_result_for_precise_approach(
            owner_result
        )
        self.publish_status(
            "owner_interaction_stage",
            owner_name=(owner_result or {}).get("owner_name", self.owner_name),
            action="waving",
            approach_mode="waving_two_stage",
            approach_stage="precise",
        )
        rospy.loginfo(
            "Waving approach stage 2/2: refreshing YOLO detection "
            "and estimating point-cloud position"
        )
        approached = self.approach_owner(
            refined_owner_result,
            self.waving_standoff_distance,
            waving=True,
        )
        if approached:
            self.handle_owner_help(refined_owner_result)
        else:
            self.speak(self.approach_failed_prompt, wait=True)
        self.approach_debug_event(
            "waving_interaction_finished",
            success=bool(approached),
            stage="precise_finished" if approached else "precise_failed",
            failure_reason=getattr(self, "last_approach_failure_reason", ""),
        )
        self.publish_status(
            "owner_interaction_finished",
            owner_name=(owner_result or {}).get("owner_name", self.owner_name),
            action="waving",
            place=(action_result or {}).get("place", "unknown"),
            approach_mode="waving_two_stage",
            approach_stage="precise_finished" if approached else "precise_failed",
            electrical_switch_state=self.electrical_switch_state,
        )
        return approached

    def refresh_owner_result_for_precise_approach(self, owner_result):
        if not isinstance(owner_result, dict):
            return owner_result

        expected_owner_index = owner_result.get("owner_index")
        last_result = None
        consecutive_matches = 0
        deadline = time.time() + self.seated_waving_refine_timeout
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            result = self.evaluate_current_frame()
            if result is not None and self.result_is_match(
                result,
                self.match_threshold,
            ):
                result_owner_index = result.get("owner_index")
                if (
                    expected_owner_index is not None
                    and result_owner_index is not None
                    and result_owner_index != expected_owner_index
                ):
                    rate.sleep()
                    continue
                last_result = result
                consecutive_matches += 1
                if consecutive_matches >= self.seated_waving_refine_required_matches:
                    break
            else:
                consecutive_matches = 0
            rate.sleep()

        if last_result is not None:
            updated_result = dict(owner_result)
            updated_candidate = dict(owner_result.get("candidate", {}))
            updated_candidate.update(last_result.get("candidate", {}))
            updated_result["candidate"] = updated_candidate
            for key in (
                "score",
                "identity_score",
                "reid_score",
                "face_score",
                "owner_score_margin",
            ):
                if key in last_result:
                    updated_result[key] = last_result[key]
            rospy.loginfo(
                "Refreshed waving owner detection for precise approach: "
                "bbox=%s score=%s",
                updated_candidate.get("bbox"),
                updated_result.get("score"),
            )
            return updated_result

        self.center_owner_in_camera(owner_result)
        rospy.logwarn(
            "Could not obtain a consecutive refreshed owner Re-ID result; "
            "using the latest YOLO tracking box for point-cloud refinement"
        )
        return owner_result

    def handle_waving_interaction(self, action_result, owner_result):
        self.approach_debug_event(
            "waving_interaction_started",
            mode="waving",
            place=(action_result or {}).get("place", "unknown"),
            action=(action_result or {}).get("action", "waving"),
        )
        self.publish_status(
            "owner_interaction_started",
            owner_name=(owner_result or {}).get("owner_name", self.owner_name),
            action="waving",
            place=(action_result or {}).get("place", "unknown"),
            approach_mode="waving",
        )
        approached = self.approach_owner(
            owner_result,
            self.waving_standoff_distance,
            waving=True,
        )
        if (
            not approached
            and "plan" in str(self.last_approach_failure_reason or "").lower()
            and self.is_probably_seated_waving(action_result, owner_result)
        ):
            rospy.logwarn(
                "Waving approach plan failed for a likely seated owner; "
                "retrying with sitting approach logic"
            )
            approached = self.approach_owner(
                owner_result,
                self.approach_standoff_distance,
                waving=False,
            )
        if approached:
            self.handle_owner_help(owner_result)
        else:
            self.speak(self.approach_failed_prompt, wait=True)
        self.approach_debug_event(
            "waving_interaction_finished",
            success=bool(approached),
            stage="finished" if approached else "failed",
            failure_reason=getattr(self, "last_approach_failure_reason", ""),
        )
        self.publish_status(
            "owner_interaction_finished",
            owner_name=(owner_result or {}).get("owner_name", self.owner_name),
            action="waving",
            place=(action_result or {}).get("place", "unknown"),
            approach_mode="waving",
            electrical_switch_state=self.electrical_switch_state,
        )

    def handle_owner_action_interaction(self, action_result, owner_result):
        normalized = self.normalize_action_result(action_result)
        if not self.action_result_is_certain(normalized):
            self.publish_workflow_state(
                "interaction_skipped",
                reason="action_uncertain",
                action=(normalized or {}).get("action", "unknown"),
                place=(normalized or {}).get("place", "unknown"),
                recognizer=(normalized or {}).get("recognizer", "unknown"),
            )
            super().publish_status(
                "owner_interaction_skipped",
                reason="action_uncertain",
                owner_index=(owner_result or {}).get("owner_index"),
                owner_name=(owner_result or {}).get("owner_name", self.owner_name),
                action=(normalized or {}).get("action", "unknown"),
                place=(normalized or {}).get("place", "unknown"),
                recognizer=(normalized or {}).get("recognizer", "unknown"),
            )
            return None
        action = normalized.get("action")
        if action in {"sitting", "lying"}:
            rospy.loginfo(
                "Optimized owner action: skipping interaction for non-interactive action %s",
                action,
            )
            self.publish_workflow_state(
                "interaction_skipped",
                reason="optimized_non_interactive_action",
                action=action,
                place=normalized.get("place", "unknown"),
            )
            super().publish_status(
                "owner_interaction_skipped",
                reason="optimized_non_interactive_action",
                owner_index=(owner_result or {}).get("owner_index"),
                owner_name=(owner_result or {}).get("owner_name", self.owner_name),
                action=action,
                place=normalized.get("place", "unknown"),
                recognizer=normalized.get("recognizer", "unknown"),
            )
            return None
        self.set_workflow_state(
            WorkflowState.INTERACT,
            reason="action_result_received",
            action=action or "unknown",
            place=normalized.get("place", "unknown"),
        )
        if action == "waving":
            return self.handle_waving_two_stage_interaction(normalized, owner_result)
        return super().handle_owner_action_interaction(normalized, owner_result)

    def reset_patrol_waypoint_state(self, waypoint_name):
        self.current_patrol_waypoint = waypoint_name
        self.current_action_result = None
        self.action_completed = False
        self.action_result = None
        self.action_result_event.clear()

    def run_patrol(self):
        rounds = self.patrol_max_rounds if self.patrol_repeat else 1
        self.patrol_results = []
        for round_index in range(rounds):
            for waypoint_index, waypoint_name in enumerate(self.patrol_waypoint_names, start=1):
                if rospy.is_shutdown():
                    return {
                        "owner_found": any(
                            item["owner_found"] for item in self.patrol_results
                        ),
                        "rooms_completed": len(self.patrol_results),
                        "interrupted": True,
                    }
                self.reset_patrol_waypoint_state(waypoint_name)
                self.publish_workflow_state(
                    "patrol_waypoint_started",
                    round=round_index + 1,
                    waypoint_index=waypoint_index,
                    waypoint=waypoint_name,
                )
                self.navigate_to_waypoint(waypoint_name)
                result = self.scan_for_owner()
                action_result = self.current_action_result
                owner_found = result is not None
                action_certain = self.action_result_is_certain(action_result)
                if owner_found and action_certain:
                    finish_reason = "interaction_completed"
                elif owner_found:
                    finish_reason = "action_uncertain"
                else:
                    finish_reason = "owner_not_found"
                room_result = {
                    "round": round_index + 1,
                    "waypoint_index": waypoint_index,
                    "waypoint": waypoint_name,
                    "owner_found": owner_found,
                    "action_certain": action_certain,
                    "action": (action_result or {}).get("action", "unknown"),
                    "place": (action_result or {}).get("place", "unknown"),
                    "finish_reason": finish_reason,
                }
                self.patrol_results.append(room_result)
                self.publish_workflow_state(
                    "patrol_waypoint_finished",
                    round=round_index + 1,
                    waypoint_index=waypoint_index,
                    waypoint=waypoint_name,
                    owner_found=owner_found,
                    action_certain=action_certain,
                    action=room_result["action"],
                    place=room_result["place"],
                    finish_reason=finish_reason,
                )
        return {
            "owner_found": any(item["owner_found"] for item in self.patrol_results),
            "rooms_completed": len(self.patrol_results),
            "interrupted": False,
        }

    def run_single_waypoint(self):
        self.reset_patrol_waypoint_state(self.waypoint_name)
        self.navigate_to_waypoint(self.waypoint_name)
        result = self.scan_for_owner()
        action_result = self.current_action_result
        self.patrol_results = [
            {
                "round": 1,
                "waypoint_index": 1,
                "waypoint": self.waypoint_name,
                "owner_found": result is not None,
                "action_certain": self.action_result_is_certain(action_result),
                "action": (action_result or {}).get("action", "unknown"),
                "place": (action_result or {}).get("place", "unknown"),
            }
        ]
        return {
            "owner_found": result is not None,
            "rooms_completed": 1,
            "interrupted": False,
        }

    def run(self):
        self.set_workflow_state(WorkflowState.INITIALIZE, reason="startup")
        self.wait_for_tts()

        self.set_workflow_state(WorkflowState.WAIT_FOR_ASR, reason="wait_for_asr")
        if not self.wait_for_asr():
            self.set_workflow_state(WorkflowState.FAILED, reason="asr_not_ready")
            return

        self.set_workflow_state(WorkflowState.WAIT_FOR_CAMERA, reason="wait_for_camera")
        self.wait_for_camera_inputs()
        self.init_yolo_window()
        self.update_yolo_window("相机已连接")

        self.set_workflow_state(WorkflowState.PREPARE_MODELS, reason="initialize_reid_and_face")
        self.init_reid_backend()
        self.init_face_recognizer()

        self.set_workflow_state(WorkflowState.REGISTER_OWNERS, reason="load_or_record_profiles")
        if self.reuse_existing_profile and self.load_all_owner_profiles():
            rospy.loginfo("Loaded existing owner Re-ID profiles from %s", self.profile_dir)
        else:
            self.record_all_owners()
        if not self.owner_profiles:
            if self.owner_embedding is not None:
                self.remember_current_owner_profile()
            else:
                raise RuntimeError("owner profiles are not ready")

        if self.navigate_enabled and self.patrol_enabled:
            patrol_summary = self.run_patrol()
        elif self.navigate_enabled:
            patrol_summary = self.run_single_waypoint()
        else:
            self.reset_patrol_waypoint_state(self.waypoint_name)
            result = self.scan_for_owner()
            action_result = self.current_action_result
            self.patrol_results = [
                {
                    "round": 1,
                    "waypoint_index": 1,
                    "waypoint": self.waypoint_name,
                    "owner_found": result is not None,
                    "action_certain": self.action_result_is_certain(action_result),
                    "action": (action_result or {}).get("action", "unknown"),
                    "place": (action_result or {}).get("place", "unknown"),
                }
            ]
            patrol_summary = {
                "owner_found": result is not None,
                "rooms_completed": 1,
                "interrupted": False,
            }

        exit_reached = False
        if (
            self.navigate_enabled
            and self.return_to_exit_when_complete
            and not rospy.is_shutdown()
        ):
            self.publish_workflow_state(
                "exit_navigation_started",
                exit_waypoint=self.exit_waypoint_name,
            )
            exit_reached = self.navigate_to_waypoint(self.exit_waypoint_name)

        self.set_workflow_state(
            WorkflowState.COMPLETED,
            reason="exit_reached" if exit_reached else "patrol_finished",
            owner_found=patrol_summary["owner_found"],
            rooms_completed=patrol_summary["rooms_completed"],
            exit_waypoint=self.exit_waypoint_name,
            exit_reached=exit_reached,
        )


def main():
    rospy.init_node("owner_voice_reid_optimized")
    node = OptimizedOwnerVoiceReid()
    try:
        node.run()
    except Exception as exc:
        rospy.logerr("Optimized owner voice Re-ID failed: %s", exc)
        node.set_workflow_state(WorkflowState.FAILED, reason=str(exc))
        node.publish_status("error", message=str(exc))
        raise


if __name__ == "__main__":
    main()
