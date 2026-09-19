#!/usr/bin/env python3
# coding=utf-8

import json
import math
import os
import struct
import sys
import threading
import time
import wave

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from perception_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    from sound_play.msg import SoundRequest
except Exception:
    SoundRequest = None


def package_dir():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def clamp_int(value, low, high):
    return int(max(low, min(high, value)))


def normalize_vector(vector):
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-8 or not np.isfinite(norm):
        return None
    return array / norm


class PersonReidOwnerTest:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_image_stamp = None
        self.latest_detections = []
        self.latest_detections_stamp = None
        self.extractor = None
        self.reid_backend = "torchreid"
        self.owner_embedding = None
        self.owner_embedding_bank = None
        self.owner_color_embedding = None
        self.owner_profile_meta = {}
        self.match_consecutive_count = 0
        self.last_announce_time = 0.0

        base_dir = package_dir()
        default_profile_dir = os.path.join(base_dir, "data", "reid_owner")
        default_torchreid_root = os.path.join(base_dir, "third_party", "deep-person-reid")

        self.image_topic = rospy.get_param("~image_topic", "/kinect2/qhd/image_color_rect")
        self.detections_topic = rospy.get_param("~detections_topic", "/perception/person_detections_2d")
        self.say_topic = rospy.get_param("~say_topic", "/voice/say")
        self.sound_topic = rospy.get_param("~sound_topic", "/robotsound")
        self.status_topic = rospy.get_param("~status_topic", "/person_reid_owner_test/status")

        self.profile_dir = os.path.expanduser(rospy.get_param("~profile_dir", default_profile_dir))
        self.profile_path = os.path.expanduser(
            rospy.get_param("~profile_path", os.path.join(self.profile_dir, "owner_profile.npz"))
        )
        self.metadata_path = os.path.expanduser(
            rospy.get_param("~metadata_path", os.path.join(self.profile_dir, "owner_profile.json"))
        )
        self.torchreid_root = os.path.expanduser(rospy.get_param("~torchreid_root", default_torchreid_root))
        self.reid_model_name = rospy.get_param("~reid_model_name", "osnet_x0_25")
        self.reid_model_path = os.path.expanduser(rospy.get_param("~reid_model_path", ""))
        self.reid_device = rospy.get_param("~reid_device", "cpu")
        self.require_cuda = bool(rospy.get_param("~require_cuda", False))
        self.allow_color_fallback = bool(rospy.get_param("~allow_color_fallback", False))

        self.startup_timeout = float(rospy.get_param("~startup_timeout", 35.0))
        self.image_max_age = float(rospy.get_param("~image_max_age", 1.5))
        self.detections_max_age = float(rospy.get_param("~detections_max_age", 1.5))
        self.detection_min_score = float(rospy.get_param("~detection_min_score", 0.30))
        self.detection_min_area_ratio = float(rospy.get_param("~detection_min_area_ratio", 0.015))
        self.crop_padding = float(rospy.get_param("~crop_padding", 0.08))
        self.crop_top_padding = max(
            self.crop_padding,
            float(rospy.get_param("~crop_top_padding", 0.20)),
        )
        self.crop_bottom_padding = max(
            0.0,
            float(rospy.get_param("~crop_bottom_padding", self.crop_padding)),
        )
        self.top_k_candidates = max(1, int(rospy.get_param("~top_k_candidates", 3)))

        self.reuse_existing_profile = bool(rospy.get_param("~reuse_existing_profile", False))
        self.record_seconds = max(0.5, float(rospy.get_param("~record_seconds", 3.0)))
        self.record_sample_count = max(1, int(rospy.get_param("~record_sample_count", 6)))
        self.record_min_samples = max(1, int(rospy.get_param("~record_min_samples", 3)))
        self.record_sample_interval = max(0.05, float(rospy.get_param("~record_sample_interval", 0.30)))
        self.save_crops = bool(rospy.get_param("~save_crops", True))

        self.match_threshold = float(rospy.get_param("~match_threshold", 0.70))
        self.match_required_consecutive = max(1, int(rospy.get_param("~match_required_consecutive", 2)))
        self.match_check_interval = max(0.05, float(rospy.get_param("~match_check_interval", 0.35)))
        self.announce_cooldown = max(0.0, float(rospy.get_param("~announce_cooldown", 5.0)))
        self.stop_after_first_match = bool(rospy.get_param("~stop_after_first_match", False))
        self.save_match_crops = bool(rospy.get_param("~save_match_crops", False))
        self.enable_lying_pose_enhancement = bool(rospy.get_param("~enable_lying_pose_enhancement", True))
        self.lying_aspect_ratio_threshold = max(
            1.0, float(rospy.get_param("~lying_aspect_ratio_threshold", 1.25))
        )
        self.lying_crop_padding = max(
            self.crop_padding, float(rospy.get_param("~lying_crop_padding", 0.22))
        )
        self.lying_match_threshold = float(rospy.get_param("~lying_match_threshold", 0.60))
        self.lying_min_reid_score = float(rospy.get_param("~lying_min_reid_score", 0.50))
        self.lying_required_consecutive = max(
            self.match_required_consecutive,
            int(rospy.get_param("~lying_required_consecutive", 3)),
        )
        self.lying_reid_weight = max(0.0, float(rospy.get_param("~lying_reid_weight", 0.75)))
        self.lying_color_weight = max(0.0, float(rospy.get_param("~lying_color_weight", 0.25)))

        self.say_wait_for_subscribers = bool(rospy.get_param("~say_wait_for_subscribers", True))
        self.say_wait_timeout = float(rospy.get_param("~say_wait_timeout", 15.0))
        self.tts_chars_per_second = max(0.1, float(rospy.get_param("~tts_chars_per_second", 6.0)))
        self.tts_min_wait = max(0.0, float(rospy.get_param("~tts_min_wait", 1.0)))
        self.tts_extra_wait = max(0.0, float(rospy.get_param("~tts_extra_wait", 0.4)))
        self.recording_text = rospy.get_param("~recording_text", "正在记录")
        self.record_done_text = rospy.get_param("~record_done_text", "记录结束")
        self.owner_found_text = rospy.get_param("~owner_found_text", "识别到主人")
        self.record_failed_text = rospy.get_param("~record_failed_text", "没有记录到足够的人体画面")
        self.ding_text = rospy.get_param("~ding_text", "叮")
        self.use_sound_play_ding = bool(rospy.get_param("~use_sound_play_ding", True))
        self.ding_frequency_hz = float(rospy.get_param("~ding_frequency_hz", 880.0))
        self.ding_duration = max(0.05, float(rospy.get_param("~ding_duration", 0.18)))
        self.ding_volume = float(rospy.get_param("~ding_volume", 1.0))

        os.makedirs(self.profile_dir, exist_ok=True)

        self.say_pub = rospy.Publisher(self.say_topic, String, queue_size=5)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=5, latch=True)
        self.sound_pub = (
            rospy.Publisher(self.sound_topic, SoundRequest, queue_size=2)
            if SoundRequest is not None
            else None
        )
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24
        )
        self.detection_sub = rospy.Subscriber(
            self.detections_topic, Detection2DArray, self.detections_callback, queue_size=1
        )

    def image_callback(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Image conversion failed: %s", exc)
            return
        with self.lock:
            self.latest_image = image
            self.latest_image_stamp = time.time()

    def detections_callback(self, message):
        with self.lock:
            self.latest_detections = list(message.detections)
            self.latest_detections_stamp = time.time()

    def publish_status(self, event, **fields):
        payload = {"event": event, "time": time.time()}
        payload.update(fields)
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def wait_for_tts(self):
        if not self.say_wait_for_subscribers:
            return
        deadline = time.time() + self.say_wait_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            if self.say_pub.get_num_connections() > 0:
                return
            rospy.sleep(0.05)
        rospy.logwarn("No TTS subscriber connected on %s", self.say_topic)

    def speak(self, text, wait=True):
        rospy.loginfo("TTS: %s", text)
        self.say_pub.publish(String(data=text))
        if wait:
            wait_seconds = max(self.tts_min_wait, len(text) / self.tts_chars_per_second) + self.tts_extra_wait
            rospy.sleep(wait_seconds)

    def ding_wav_path(self):
        return os.path.join(self.profile_dir, "ding.wav")

    def ensure_ding_wav(self):
        path = self.ding_wav_path()
        if os.path.exists(path) and os.path.getsize(path) > 128:
            return path
        sample_rate = 16000
        sample_count = int(sample_rate * self.ding_duration)
        fade_count = max(1, int(sample_rate * min(0.02, self.ding_duration / 4.0)))
        amplitude = int(32767 * max(0.0, min(1.0, self.ding_volume)) * 0.55)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for sample_index in range(sample_count):
                phase = 2.0 * math.pi * self.ding_frequency_hz * sample_index / sample_rate
                envelope = 1.0
                if sample_index < fade_count:
                    envelope = sample_index / float(fade_count)
                elif sample_index > sample_count - fade_count:
                    envelope = max(0.0, (sample_count - sample_index) / float(fade_count))
                value = int(amplitude * envelope * math.sin(phase))
                wav_file.writeframes(struct.pack("<h", value))
        return path

    def play_ding(self):
        if self.use_sound_play_ding and self.sound_pub is not None:
            path = self.ensure_ding_wav()
            request = SoundRequest()
            request.sound = SoundRequest.PLAY_FILE
            request.command = SoundRequest.PLAY_ONCE
            request.volume = max(0.0, min(1.0, self.ding_volume))
            request.arg = path
            self.sound_pub.publish(request)
            rospy.loginfo("Ding played with sound_play: %s", path)
            return
        self.speak(self.ding_text, wait=False)

    def wait_for_camera_inputs(self):
        deadline = time.time() + self.startup_timeout
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                has_image = self.latest_image is not None
                has_detections = self.latest_detections_stamp is not None
            if has_image and has_detections:
                self.publish_status("camera_ready")
                return True
            rate.sleep()
        with self.lock:
            has_image = self.latest_image is not None
            has_detections = self.latest_detections_stamp is not None
        raise RuntimeError(
            "camera/detection inputs not ready: image=%s topic=%s detections=%s topic=%s"
            % (has_image, self.image_topic, has_detections, self.detections_topic)
        )

    def init_reid_backend(self):
        if self.reid_model_path and not os.path.exists(self.reid_model_path):
            raise RuntimeError("reid_model_path does not exist: %s" % self.reid_model_path)
        if os.path.isdir(self.torchreid_root) and self.torchreid_root not in sys.path:
            sys.path.insert(0, self.torchreid_root)
        try:
            import torch
            from torchreid.utils import FeatureExtractor
        except Exception as exc:
            if not self.allow_color_fallback:
                raise RuntimeError(
                    "Torchreid dependencies are not ready. Run: roscd wpb_task1_owner_search && tools/setup_person_reid.sh"
                ) from exc
            rospy.logwarn("Torchreid unavailable; using color-histogram fallback: %s", exc)
            self.reid_backend = "color_hist"
            return
        if self.reid_device.startswith("cuda") and not torch.cuda.is_available():
            if self.require_cuda:
                raise RuntimeError("CUDA requested for Re-ID but torch.cuda.is_available() is false")
            rospy.logwarn("CUDA unavailable for Re-ID; falling back to CPU")
            self.reid_device = "cpu"
        self.extractor = FeatureExtractor(
            model_name=self.reid_model_name,
            model_path=self.reid_model_path,
            device=self.reid_device,
            verbose=False,
        )
        self.reid_backend = "torchreid"
        rospy.loginfo(
            "Person Re-ID ready: backend=torchreid model=%s weights=%s device=%s",
            self.reid_model_name,
            self.reid_model_path or "auto-pretrained",
            self.reid_device,
        )

    def snapshot(self):
        now = time.time()
        with self.lock:
            image = None if self.latest_image is None else self.latest_image.copy()
            image_stamp = self.latest_image_stamp
            detections = list(self.latest_detections)
            detections_stamp = self.latest_detections_stamp
        if image is None or image_stamp is None or now - image_stamp > self.image_max_age:
            return None, []
        if detections_stamp is None or now - detections_stamp > self.detections_max_age:
            return image, []
        return image, detections

    def person_candidates(self, image, detections):
        image_height, image_width = image.shape[:2]
        image_area = float(max(1, image_height * image_width))
        candidates = []
        for detection in detections:
            label = str(getattr(detection, "class_name", "") or "").strip().lower()
            if label and label != "person":
                continue
            score = float(getattr(detection, "score", 0.0) or 0.0)
            if score < self.detection_min_score:
                continue
            xmin = clamp_int(getattr(detection, "xmin", 0), 0, image_width - 1)
            ymin = clamp_int(getattr(detection, "ymin", 0), 0, image_height - 1)
            xmax = clamp_int(getattr(detection, "xmax", 0), 0, image_width - 1)
            ymax = clamp_int(getattr(detection, "ymax", 0), 0, image_height - 1)
            if xmax <= xmin or ymax <= ymin:
                continue
            area_ratio = ((xmax - xmin) * (ymax - ymin)) / image_area
            if area_ratio < self.detection_min_area_ratio:
                continue
            width = xmax - xmin
            height = ymax - ymin
            aspect_ratio = float(width) / float(max(1, height))
            priority = score * area_ratio
            candidates.append(
                {
                    "bbox": [xmin, ymin, xmax, ymax],
                    "score": score,
                    "area_ratio": area_ratio,
                    "aspect_ratio": aspect_ratio,
                    "priority": priority,
                }
            )
        candidates.sort(key=lambda item: item["priority"], reverse=True)
        return candidates[: self.top_k_candidates]

    def crop_candidate(self, image, candidate, padding=None):
        image_height, image_width = image.shape[:2]
        xmin, ymin, xmax, ymax = candidate["bbox"]
        width = xmax - xmin
        height = ymax - ymin
        crop_padding = self.crop_padding if padding is None else float(padding)
        top_padding = self.crop_top_padding if padding is None else max(self.crop_top_padding, crop_padding)
        bottom_padding = self.crop_bottom_padding if padding is None else max(self.crop_bottom_padding, crop_padding)
        pad_x = int(width * crop_padding)
        pad_top = int(height * top_padding)
        pad_bottom = int(height * bottom_padding)
        crop_xmin = clamp_int(xmin - pad_x, 0, image_width - 1)
        crop_ymin = clamp_int(ymin - pad_top, 0, image_height - 1)
        crop_xmax = clamp_int(xmax + pad_x, 0, image_width - 1)
        crop_ymax = clamp_int(ymax + pad_bottom, 0, image_height - 1)
        if crop_xmax <= crop_xmin or crop_ymax <= crop_ymin:
            return None, None
        crop = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax].copy()
        if crop.size == 0:
            return None, None
        return crop, [crop_xmin, crop_ymin, crop_xmax, crop_ymax]

    def extract_embeddings(self, crops):
        if not crops:
            return []
        if self.reid_backend == "color_hist":
            embeddings = [self.color_hist_embedding(crop) for crop in crops]
            return [embedding for embedding in embeddings if embedding is not None]
        rgb_crops = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops]
        features = self.extractor(rgb_crops)
        if hasattr(features, "detach"):
            features = features.detach().cpu().numpy()
        embeddings = []
        for feature in np.asarray(features):
            embedding = normalize_vector(feature)
            if embedding is not None:
                embeddings.append(embedding)
        return embeddings

    @staticmethod
    def color_hist_embedding(crop):
        resized = cv2.resize(crop, (96, 192), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        return normalize_vector(hist.reshape(-1))

    def color_similarity(self, crop):
        if self.owner_color_embedding is None:
            return None
        color_embedding = self.color_hist_embedding(crop)
        if color_embedding is None:
            return None
        return float(np.dot(self.owner_color_embedding, color_embedding))

    def is_lying_candidate(self, candidate):
        if not self.enable_lying_pose_enhancement:
            return False
        xmin, ymin, xmax, ymax = candidate["bbox"]
        width = xmax - xmin
        height = ymax - ymin
        aspect_ratio = float(width) / float(max(1, height))
        return aspect_ratio >= self.lying_aspect_ratio_threshold

    def reid_query_variants(self, crop, lying_pose):
        variants = [("raw", crop)]
        if lying_pose and self.enable_lying_pose_enhancement:
            variants.append(("rot90_cw", cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)))
            variants.append(("rot90_ccw", cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)))
        return variants

    def reid_similarity(self, embedding):
        if self.owner_embedding_bank is not None:
            scores = np.dot(self.owner_embedding_bank, embedding)
            return float(np.max(scores))
        return float(np.dot(self.owner_embedding, embedding))

    def fused_lie_score(self, reid_score, color_score):
        if color_score is None or self.lying_color_weight <= 0.0:
            return reid_score
        total_weight = max(1e-6, self.lying_reid_weight + self.lying_color_weight)
        reid_weight = self.lying_reid_weight / total_weight
        color_weight = self.lying_color_weight / total_weight
        return reid_weight * reid_score + color_weight * color_score

    @staticmethod
    def result_is_match(result, default_threshold):
        if result is None:
            return False
        score = float(result.get("score", -1.0))
        reid_score = float(result.get("reid_score", score))
        threshold = float(result.get("match_threshold", default_threshold))
        min_reid_score = float(result.get("min_reid_score", -1.0))
        return score >= threshold and reid_score >= min_reid_score

    def profile_sample_dir(self):
        sample_dir = os.path.join(self.profile_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        return sample_dir

    def match_sample_dir(self):
        match_dir = os.path.join(self.profile_dir, "matches")
        os.makedirs(match_dir, exist_ok=True)
        return match_dir

    def save_crop(self, crop, directory, prefix, index):
        if not self.save_crops:
            return ""
        filename = "%s_%03d_%d.jpg" % (prefix, index, int(time.time() * 1000))
        path = os.path.join(directory, filename)
        cv2.imwrite(path, crop)
        return path

    def save_owner_profile(self, embeddings, sample_meta, color_embeddings=None):
        mean_embedding = normalize_vector(np.mean(np.vstack(embeddings), axis=0))
        if mean_embedding is None:
            raise RuntimeError("owner embedding is invalid")
        embedding_bank = [mean_embedding]
        for embedding in embeddings:
            normalized = normalize_vector(embedding)
            if normalized is not None:
                embedding_bank.append(normalized)
        embedding_bank = np.vstack(embedding_bank).astype(np.float32)
        npz_payload = {
            "embedding": mean_embedding.astype(np.float32),
            "embedding_bank": embedding_bank,
        }
        mean_color_embedding = None
        if color_embeddings:
            mean_color_embedding = normalize_vector(np.mean(np.vstack(color_embeddings), axis=0))
            if mean_color_embedding is not None:
                npz_payload["color_embedding"] = mean_color_embedding.astype(np.float32)
        np.savez(self.profile_path, **npz_payload)
        metadata = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": self.reid_backend,
            "model_name": self.reid_model_name,
            "model_path": self.reid_model_path,
            "device": self.reid_device,
            "match_threshold": self.match_threshold,
            "lying_pose_enhancement": self.enable_lying_pose_enhancement,
            "lying_match_threshold": self.lying_match_threshold,
            "lying_min_reid_score": self.lying_min_reid_score,
            "lying_required_consecutive": self.lying_required_consecutive,
            "has_color_embedding": mean_color_embedding is not None,
            "samples": sample_meta,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        self.owner_embedding = mean_embedding
        self.owner_embedding_bank = embedding_bank
        self.owner_color_embedding = mean_color_embedding
        self.owner_profile_meta = metadata
        self.publish_status("profile_saved", profile_path=self.profile_path, samples=len(sample_meta))

    def load_owner_profile(self):
        if not os.path.exists(self.profile_path):
            return False
        profile = np.load(self.profile_path, allow_pickle=False)
        embedding = normalize_vector(profile["embedding"])
        if embedding is None:
            raise RuntimeError("owner profile exists but embedding is invalid: %s" % self.profile_path)
        self.owner_embedding = embedding
        bank = []
        if "embedding_bank" in profile.files:
            for vector in np.asarray(profile["embedding_bank"]):
                normalized = normalize_vector(vector)
                if normalized is not None:
                    bank.append(normalized)
        if not bank:
            bank.append(embedding)
        self.owner_embedding_bank = np.vstack(bank).astype(np.float32)
        self.owner_color_embedding = None
        if "color_embedding" in profile.files:
            self.owner_color_embedding = normalize_vector(profile["color_embedding"])
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as metadata_file:
                self.owner_profile_meta = json.load(metadata_file)
        self.publish_status("profile_loaded", profile_path=self.profile_path)
        return True

    def record_owner(self):
        self.publish_status("recording_started")
        self.speak(self.recording_text, wait=True)
        self.play_ding()
        embeddings = []
        color_embeddings = []
        sample_meta = []
        sample_dir = self.profile_sample_dir()
        deadline = time.time() + self.record_seconds
        next_sample_time = 0.0
        rate = rospy.Rate(30)
        while not rospy.is_shutdown() and time.time() < deadline and len(embeddings) < self.record_sample_count:
            now = time.time()
            if now < next_sample_time:
                rate.sleep()
                continue
            next_sample_time = now + self.record_sample_interval
            image, detections = self.snapshot()
            if image is None:
                rospy.logwarn_throttle(1.0, "Waiting for fresh camera image on %s", self.image_topic)
                rate.sleep()
                continue
            candidates = self.person_candidates(image, detections)
            if not candidates:
                rospy.logwarn_throttle(1.0, "Waiting for person detection on %s", self.detections_topic)
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
            sample_index = len(embeddings) + 1
            path = self.save_crop(crop, sample_dir, "owner", sample_index)
            color_embedding = self.color_hist_embedding(crop)
            if color_embedding is not None:
                color_embeddings.append(color_embedding)
            embeddings.append(extracted[0])
            sample_meta.append(
                {
                    "index": sample_index,
                    "image": path,
                    "bbox": candidates[0]["bbox"],
                    "crop_bbox": crop_bbox,
                    "det_score": candidates[0]["score"],
                    "area_ratio": candidates[0]["area_ratio"],
                }
            )
            self.publish_status("recording_sample", sample=sample_index, required=self.record_min_samples)
            rospy.loginfo("Owner Re-ID sample %d captured: %s", sample_index, path or "not saved")
            rate.sleep()
        if len(embeddings) < self.record_min_samples:
            self.publish_status("recording_failed", samples=len(embeddings), required=self.record_min_samples)
            self.speak(self.record_failed_text, wait=True)
            raise RuntimeError("only captured %d/%d usable Re-ID samples" % (len(embeddings), self.record_min_samples))
        self.save_owner_profile(embeddings, sample_meta, color_embeddings=color_embeddings)
        self.speak(self.record_done_text, wait=True)
        self.publish_status("recording_done", samples=len(embeddings))

    def evaluate_current_frame(self):
        image, detections = self.snapshot()
        if image is None:
            return None
        candidates = self.person_candidates(image, detections)
        if not candidates:
            return None
        query_crops = []
        records = []
        for candidate in candidates:
            lying_pose = self.is_lying_candidate(candidate)
            padding = self.lying_crop_padding if lying_pose else self.crop_padding
            crop, crop_bbox = self.crop_candidate(image, candidate, padding=padding)
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
        best_result = None
        for record in records:
            variant_scores = []
            for query_index, variant_name in zip(record["query_indexes"], record["variant_names"]):
                variant_scores.append((self.reid_similarity(embeddings[query_index]), variant_name))
            reid_score, best_variant = max(variant_scores, key=lambda item: item[0])
            color_score = self.color_similarity(record["crop"])
            lying_pose = record["meta"].get("lying_pose", False)
            if lying_pose:
                score = self.fused_lie_score(reid_score, color_score)
                match_threshold = self.lying_match_threshold
                min_reid_score = self.lying_min_reid_score
                required_consecutive = self.lying_required_consecutive
            else:
                score = reid_score
                match_threshold = self.match_threshold
                min_reid_score = -1.0
                required_consecutive = self.match_required_consecutive
            result = {
                "score": float(score),
                "reid_score": float(reid_score),
                "color_score": None if color_score is None else float(color_score),
                "match_threshold": match_threshold,
                "min_reid_score": min_reid_score,
                "required_consecutive": required_consecutive,
                "best_variant": best_variant,
                "candidate": record["meta"],
                "crop": record["crop"],
                "num_candidates": len(records),
            }
            if best_result is None or result["score"] > best_result["score"]:
                best_result = result
        return best_result

    def maybe_save_match_crop(self, crop, score):
        if not self.save_match_crops:
            return ""
        return self.save_crop(crop, self.match_sample_dir(), "match_%.2f" % score, 1)

    def recognition_loop(self):
        self.publish_status("recognition_started", threshold=self.match_threshold)
        rate = rospy.Rate(max(1.0, 1.0 / self.match_check_interval))
        while not rospy.is_shutdown():
            result = self.evaluate_current_frame()
            if result is None:
                self.match_consecutive_count = 0
                rate.sleep()
                continue
            score = result["score"]
            match_threshold = result.get("match_threshold", self.match_threshold)
            required_consecutive = result.get("required_consecutive", self.match_required_consecutive)
            matched = self.result_is_match(result, self.match_threshold)
            if matched:
                self.match_consecutive_count += 1
            else:
                self.match_consecutive_count = 0
            self.publish_status(
                "match_score",
                score=score,
                reid_score=result.get("reid_score", score),
                color_score=result.get("color_score"),
                matched=matched,
                consecutive=self.match_consecutive_count,
                threshold=match_threshold,
                required_consecutive=required_consecutive,
                lying_pose=result.get("candidate", {}).get("lying_pose", False),
                best_variant=result.get("best_variant", "raw"),
            )
            if self.match_consecutive_count >= required_consecutive:
                now = time.time()
                if now - self.last_announce_time >= self.announce_cooldown:
                    crop_path = self.maybe_save_match_crop(result["crop"], score)
                    rospy.loginfo("Owner recognized: score=%.3f crop=%s", score, crop_path or "not saved")
                    self.speak(self.owner_found_text, wait=False)
                    self.publish_status("owner_recognized", score=score, crop=crop_path)
                    self.last_announce_time = now
                    if self.stop_after_first_match:
                        return
            rate.sleep()

    def run(self):
        self.wait_for_tts()
        self.wait_for_camera_inputs()
        self.init_reid_backend()
        if self.reuse_existing_profile and self.load_owner_profile():
            rospy.loginfo("Loaded existing owner Re-ID profile: %s", self.profile_path)
        else:
            self.record_owner()
        if self.owner_embedding is None:
            raise RuntimeError("owner profile is not ready")
        self.recognition_loop()


def main():
    rospy.init_node("person_reid_owner_test")
    node = PersonReidOwnerTest()
    try:
        node.run()
    except Exception as exc:
        rospy.logerr("Person Re-ID owner test failed: %s", exc)
        node.publish_status("error", message=str(exc))
        raise


if __name__ == "__main__":
    main()
