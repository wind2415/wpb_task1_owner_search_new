#!/usr/bin/env python3
import os
import threading
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated.*",
    category=FutureWarning,
    module="huggingface_hub.file_download",
)

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from perception_msgs.msg import Detection2D, Detection2DArray
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


def parse_classes(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return ["person"]


def resolve_model_path(model_path):
    expanded = os.path.expanduser(os.path.expandvars(model_path))
    if os.path.isdir(expanded):
        raise ValueError(f"model_path is a directory, not a .pt file: {expanded}")
    if os.path.exists(expanded):
        return expanded
    if os.path.sep in expanded:
        raise ValueError(f"model_path does not exist: {expanded}")
    return model_path


class AsyncYoloWorldNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_log_time = 0.0
        self.last_output_time = 0.0
        self.last_input_time = 0.0
        self.inference_error_count = 0
        self.inference_success_count = 0
        self.paused = False
        self.pending_image = None
        self.condition = threading.Condition()
        self.stop_event = threading.Event()

        self.image_topic = rospy.get_param("~image_topic", "/kinect2/qhd/image_color_rect")
        self.model_path = resolve_model_path(rospy.get_param("~model_path", "yolov8s-world.pt"))
        self.classes = parse_classes(rospy.get_param("~classes", ["person"]))
        self.device = rospy.get_param("~device", "cuda:0")
        self.require_gpu = bool(rospy.get_param("~require_gpu", True))
        self.imgsz = int(rospy.get_param("~imgsz", 640))
        self.conf = float(rospy.get_param("~conf", 0.25))
        self.iou = float(rospy.get_param("~iou", 0.45))
        self.max_det = int(rospy.get_param("~max_det", 50))
        self.process_every_n = max(1, int(rospy.get_param("~process_every_n", 1)))
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))
        self.warmup_enabled = bool(rospy.get_param("~warmup", True))
        self.log_interval = float(rospy.get_param("~log_interval", 5.0))

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            message = "CUDA requested but torch.cuda.is_available() is false"
            if self.require_gpu:
                raise RuntimeError(message)
            rospy.logwarn("%s; falling back to CPU", message)
            self.device = "cpu"
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True

        from ultralytics import YOLOWorld

        rospy.loginfo("Loading YOLO-World model: %s", self.model_path)
        self.model = YOLOWorld(self.model_path)
        self.model.set_classes(self.classes)
        rospy.loginfo("YOLO-World classes: %s", ", ".join(self.classes))
        rospy.loginfo("YOLO-World device: %s", self.device)
        self.warmup_model()

        self.det_pub = rospy.Publisher("~detections", Detection2DArray, queue_size=1)
        self.debug_pub = (
            rospy.Publisher("~debug_image", Image, queue_size=1)
            if self.publish_debug
            else None
        )

        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.pause_sub = rospy.Subscriber("~pause", Bool, self.pause_callback, queue_size=1)
        self.worker = threading.Thread(
            target=self.inference_loop,
            name="yoloworld-inference",
            daemon=True,
        )
        self.worker.start()
        rospy.on_shutdown(self.close)
        rospy.loginfo(
            "Subscribed image topic: %s; latest-frame mode enabled, process_every_n=%d",
            self.image_topic,
            self.process_every_n,
        )

    def warmup_model(self):
        if not self.warmup_enabled:
            return
        warmup_size = max(32, self.imgsz)
        warmup_image = np.zeros((warmup_size, warmup_size, 3), dtype=np.uint8)
        start = time.time()
        try:
            with torch.inference_mode():
                self.model.predict(
                    source=warmup_image,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    max_det=self.max_det,
                    verbose=False,
                )
            rospy.loginfo("YOLO-World warm-up complete in %.2fs", time.time() - start)
        except Exception as exc:
            rospy.logwarn("YOLO-World warm-up failed; continuing: %s", exc)

    def pause_callback(self, msg):
        with self.condition:
            paused = bool(msg.data)
            changed = paused != self.paused
            self.paused = paused
            if self.paused:
                self.pending_image = None
            self.condition.notify_all()
        if not changed:
            return
        if self.paused:
            rospy.loginfo("YOLO-World inference paused")
        else:
            rospy.loginfo("YOLO-World inference resumed")

    def image_callback(self, msg):
        with self.condition:
            if self.paused:
                return
            self.frame_count += 1
            if self.frame_count % self.process_every_n != 0:
                return
            received_at = time.time()
            self.pending_image = (msg, received_at)
            self.last_input_time = received_at
            self.condition.notify()

    def inference_loop(self):
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            with self.condition:
                while (
                    self.pending_image is None
                    and not self.stop_event.is_set()
                    and not rospy.is_shutdown()
                ):
                    self.condition.wait(timeout=0.5)
                if self.stop_event.is_set() or rospy.is_shutdown():
                    return
                pending_image = self.pending_image
                self.pending_image = None
                if self.paused:
                    continue
            try:
                image_msg, received_at = pending_image
                self.process_image(image_msg, received_at)
            except Exception as exc:
                self.handle_inference_error(exc)

    def handle_inference_error(self, exc):
        self.inference_error_count += 1
        if self.device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        rospy.logerr_throttle(
            2.0,
            "YOLO-World worker recovered from frame-processing error "
            "(errors=%d successes=%d): %s",
            self.inference_error_count,
            self.inference_success_count,
            exc,
        )

    def process_image(self, msg, received_at=None):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logerr_throttle(2.0, "cv_bridge image conversion failed: %s", exc)
            return

        start = time.time()
        try:
            with torch.inference_mode():
                results = self.model.predict(
                    source=image,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    max_det=self.max_det,
                    verbose=False,
                )
        except Exception as exc:
            self.handle_inference_error(exc)
            return

        result = results[0]
        det_array = Detection2DArray()
        det_array.header = msg.header

        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

            for box, score, class_id in zip(xyxy, confs, class_ids):
                x1, y1, x2, y2 = [int(round(value)) for value in box]
                x1 = max(0, min(x1, image.shape[1] - 1))
                x2 = max(0, min(x2, image.shape[1] - 1))
                y1 = max(0, min(y1, image.shape[0] - 1))
                y2 = max(0, min(y2, image.shape[0] - 1))

                det = Detection2D()
                det.header = msg.header
                det.class_id = int(class_id)
                det.class_name = (
                    str(names.get(int(class_id), class_id))
                    if isinstance(names, dict)
                    else str(class_id)
                )
                det.score = float(score)
                det.xmin = min(x1, x2)
                det.ymin = min(y1, y2)
                det.xmax = max(x1, x2)
                det.ymax = max(y1, y2)
                det.center_x = int((det.xmin + det.xmax) / 2)
                det.center_y = int((det.ymin + det.ymax) / 2)
                det_array.detections.append(det)

        self.det_pub.publish(det_array)

        if self.debug_pub is not None:
            debug = self.draw_debug(image.copy(), det_array.detections)
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

        now = time.time()
        self.last_output_time = now
        self.inference_success_count += 1
        if now - self.last_log_time >= self.log_interval:
            elapsed_ms = (now - start) * 1000.0
            input_age = now - received_at if received_at else 0.0
            rospy.loginfo(
                "YOLO-World detections=%d inference=%.1fms input_age=%.2fs "
                "device=%s successes=%d errors=%d",
                len(det_array.detections),
                elapsed_ms,
                input_age,
                self.device,
                self.inference_success_count,
                self.inference_error_count,
            )
            self.last_log_time = now

    @staticmethod
    def draw_debug(image, detections):
        for det in detections:
            color = (0, 255, 0)
            cv2.rectangle(image, (det.xmin, det.ymin), (det.xmax, det.ymax), color, 2)
            label = f"{det.class_name} {det.score:.2f}"
            y = max(15, det.ymin - 5)
            cv2.putText(
                image,
                label,
                (det.xmin, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return image

    def close(self):
        self.stop_event.set()
        with self.condition:
            self.pending_image = None
            self.condition.notify_all()
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)


def main():
    rospy.init_node("yoloworld_node")
    try:
        AsyncYoloWorldNode()
    except Exception as exc:
        rospy.logfatal("Failed to start yoloworld_node: %s", exc)
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
