#!/usr/bin/env python3
import os
import threading
import time

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from perception_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image


class StableYoloWorldDebugViewer:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.gui_lock = threading.Lock()
        self.latest_raw = None
        self.latest_detections = (0, [])
        self.raw_sequence = 0
        self.detection_sequence = 0
        self.cached_raw_sequence = 0
        self.cached_detection_sequence = 0
        self.cached_image = None
        self.cached_debug_image = None
        self.last_display_time = 0.0
        self.last_gui_error_time = 0.0
        self.last_recreate_time = 0.0
        self.last_window_check = 0.0
        self.windows_ready = False
        self.gui_error_restart_seconds = max(
            2.0,
            float(rospy.get_param("~gui_error_restart_seconds", 5.0)),
        )
        self.window_retry_seconds = max(
            0.5,
            float(rospy.get_param("~window_retry_seconds", 2.0)),
        )

        self.image_topic = rospy.get_param("~image_topic", "/kinect2/hd/image_color_rect")
        self.detections_topic = rospy.get_param(
            "~detections_topic",
            "/perception/person_detections_2d",
        )
        self.show_raw = bool(rospy.get_param("~show_raw", True))
        self.show_debug = bool(rospy.get_param("~show_debug", True))
        self.display_fps = max(10.0, float(rospy.get_param("~display_fps", 30.0)))
        self.raw_window = rospy.get_param("~raw_window", "WPR Camera")
        self.debug_window = rospy.get_param(
            "~debug_window",
            "YOLO-World Person Detection",
        )
        self.gui_enabled = bool(os.environ.get("DISPLAY"))

        if not self.gui_enabled:
            rospy.logwarn("DISPLAY is not set; OpenCV windows will not be shown")
        else:
            if not self.create_windows():
                self.gui_enabled = False

        if self.show_raw or self.show_debug:
            self.raw_sub = rospy.Subscriber(
                self.image_topic,
                Image,
                self.raw_callback,
                queue_size=1,
                buff_size=2**24,
            )
            rospy.loginfo("Stable viewer raw image: %s", self.image_topic)
        if self.show_debug:
            self.detections_sub = rospy.Subscriber(
                self.detections_topic,
                Detection2DArray,
                self.detections_callback,
                queue_size=1,
            )
            rospy.loginfo("Stable viewer detections: %s", self.detections_topic)

    def raw_callback(self, msg):
        with self.lock:
            self.raw_sequence += 1
            self.latest_raw = (self.raw_sequence, msg)

    def detections_callback(self, msg):
        detections = []
        for det in msg.detections:
            detections.append(
                (
                    int(det.xmin),
                    int(det.ymin),
                    int(det.xmax),
                    int(det.ymax),
                    str(det.class_name),
                    float(det.score),
                )
            )
        with self.lock:
            self.detection_sequence += 1
            self.latest_detections = (self.detection_sequence, detections)

    def take_latest(self):
        with self.lock:
            return self.latest_raw, self.latest_detections

    def decode_image(self, item):
        if item is None or item[0] == self.cached_raw_sequence:
            return self.cached_image, False
        sequence, msg = item
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr_throttle(2.0, "cv_bridge conversion failed: %s", exc)
            self.cached_raw_sequence = sequence
            self.cached_image = None
            self.cached_debug_image = None
            return None, True
        self.cached_raw_sequence = sequence
        self.cached_image = image
        self.cached_debug_image = None
        return image, True

    @staticmethod
    def draw_detections(image, detections):
        for x1, y1, x2, y2, class_name, score in detections:
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {score:.2f}"
            y = max(15, y1 - 5)
            cv2.putText(
                image,
                label,
                (x1, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return image

    def create_windows(self):
        if not self.gui_enabled:
            return False
        try:
            if self.show_raw:
                cv2.namedWindow(self.raw_window, cv2.WINDOW_NORMAL)
            if self.show_debug:
                cv2.namedWindow(self.debug_window, cv2.WINDOW_NORMAL)
            self.windows_ready = True
            self.last_window_check = 0.0
            return True
        except Exception as exc:
            self.windows_ready = False
            rospy.logwarn("YOLO viewer could not create OpenCV windows: %s", exc)
            return False

    def windows_are_valid(self):
        if not self.windows_ready:
            return False
        window_names = []
        if self.show_raw:
            window_names.append(self.raw_window)
        if self.show_debug:
            window_names.append(self.debug_window)
        for window_name in window_names:
            try:
                visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
            except Exception:
                return False
            if visible < 0:
                return False
        return True

    def recreate_windows(self):
        if not self.gui_enabled:
            return
        with self.gui_lock:
            now = time.time()
            if now - self.last_recreate_time < self.window_retry_seconds:
                return
            self.last_recreate_time = now
            self.windows_ready = False
            try:
                cv2.destroyAllWindows()
                if self.create_windows():
                    rospy.loginfo("YOLO viewer OpenCV windows recreated")
            except Exception as exc:
                self.windows_ready = False
                rospy.logwarn_throttle(
                    2.0,
                    "YOLO viewer window recreation failed: %s",
                    exc,
                )

    def repaint_windows(self):
        with self.gui_lock:
            if not self.windows_ready:
                raise RuntimeError("OpenCV viewer windows are not ready")
            now = time.time()
            if now - self.last_window_check >= 1.0:
                if not self.windows_are_valid():
                    raise RuntimeError("OpenCV viewer window is no longer valid")
                self.last_window_check = now
            if self.show_raw and self.cached_image is not None:
                cv2.imshow(self.raw_window, self.cached_image)
            if self.show_debug and self.cached_debug_image is not None:
                cv2.imshow(self.debug_window, self.cached_debug_image)
            if hasattr(cv2, "pollKey"):
                return cv2.pollKey()
            return cv2.waitKey(1)

    def run(self):
        sleep_seconds = 1.0 / self.display_fps
        while not rospy.is_shutdown():
            raw_item, detection_item = self.take_latest()
            if self.gui_enabled:
                try:
                    image, raw_changed = self.decode_image(raw_item)
                    if image is not None:
                        detection_sequence, detections = detection_item
                        detections_changed = detection_sequence != self.cached_detection_sequence
                        if self.show_debug and (
                            raw_changed
                            or detections_changed
                            or self.cached_debug_image is None
                        ):
                            self.cached_detection_sequence = detection_sequence
                            self.cached_debug_image = self.draw_detections(
                                image.copy(),
                                detections,
                            )

                    key = self.repaint_windows()
                    self.last_display_time = time.time()
                    if key & 0xFF in (27, ord("q")):
                        rospy.signal_shutdown("YOLO viewer closed by user")
                        break
                except Exception as exc:
                    now = time.time()
                    if now - self.last_gui_error_time >= self.gui_error_restart_seconds:
                        rospy.logwarn("YOLO viewer GUI update failed: %s", exc)
                        self.last_gui_error_time = now
                    self.recreate_windows()
            time.sleep(sleep_seconds)

    def close(self):
        if self.gui_enabled:
            with self.gui_lock:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass


def main():
    rospy.init_node("yoloworld_debug_viewer")
    viewer = StableYoloWorldDebugViewer()
    rospy.on_shutdown(viewer.close)
    viewer.run()


if __name__ == "__main__":
    main()
