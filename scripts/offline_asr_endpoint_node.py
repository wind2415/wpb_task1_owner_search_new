#!/usr/bin/env python3

import audioop
from collections import deque
import os
import shutil
import subprocess
import tempfile
import wave

import rospy
from std_msgs.msg import String


class OfflineAsrEndpointNode:
    def __init__(self):
        self.publish_topic = self._param(
            "publish_topic", "/voice/asr_text"
        )
        self.legacy_publish_topic = self._param(
            "legacy_publish_topic", "/xfyun/iat"
        )
        self.model_size = self._param("model_size", "small.en")
        self.hf_endpoint = self._param("hf_endpoint", "https://hf-mirror.com")
        self.device = str(self._param("device", "cpu")).strip()
        self.compute_type = str(self._param("compute_type", "int8")).strip()
        self.language = str(self._param("language", "en")).strip()
        self.sample_rate = max(8000, int(self._param("sample_rate", 16000)))
        self.chunk_seconds = max(
            0.05, float(self._param("chunk_seconds", 0.20))
        )
        self.silence_seconds = max(
            self.chunk_seconds,
            float(self._param("silence_seconds", 0.80)),
        )
        self.pre_roll_seconds = max(
            0.0, float(self._param("pre_roll_seconds", 0.30))
        )
        self.min_voice_seconds = max(
            self.chunk_seconds,
            float(self._param("min_voice_seconds", 0.60)),
        )
        self.max_utterance_seconds = max(
            self.min_voice_seconds,
            float(self._param("max_utterance_seconds", 15.0)),
        )
        self.energy_threshold = max(
            0, int(self._param("energy_threshold", 300))
        )
        self.beam_size = max(1, int(self._param("beam_size", 1)))
        self.capture_device = str(self._param("capture_device", "default")).strip()
        self.capture_backend = str(
            self._param("capture_backend", "auto")
        ).strip().lower()
        self.capture_source = str(self._param("capture_source", "")).strip()
        self.capture_volume = str(self._param("capture_volume", "70%")).strip()
        self.capture_gain = max(1.0, float(self._param("capture_gain", 2.0)))

        self.pub = rospy.Publisher(self.publish_topic, String, queue_size=10)
        self.legacy_pub = None
        if self.legacy_publish_topic and self.legacy_publish_topic != self.publish_topic:
            self.legacy_pub = rospy.Publisher(
                self.legacy_publish_topic, String, queue_size=10
            )

        self._load_runtime()

    @staticmethod
    def _param(name, default):
        return rospy.get_param("~%s" % name, rospy.get_param("/asr/%s" % name, default))

    def _load_runtime(self):
        if self.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self.hf_endpoint)
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Missing offline ASR dependencies. Run tools/setup_offline_voice.sh first. Detail: %s"
                % exc
            )

        rospy.loginfo(
            "Loading faster-whisper model %s (%s/%s, beam_size=%d)",
            self.model_size,
            self.device,
            self.compute_type,
            self.beam_size,
        )
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def run(self):
        bytes_per_sample = 2
        chunk_bytes = max(
            2, int(self.sample_rate * self.chunk_seconds * bytes_per_sample)
        )
        silence_chunk_count = max(
            1, int((self.silence_seconds + self.chunk_seconds - 0.000001) / self.chunk_seconds)
        )
        pre_roll_chunk_count = max(
            1, int((self.pre_roll_seconds + self.chunk_seconds - 0.000001) / self.chunk_seconds)
        )
        min_voice_bytes = max(
            2, int(self.sample_rate * self.min_voice_seconds * bytes_per_sample)
        )
        max_utterance_bytes = max(
            min_voice_bytes,
            int(self.sample_rate * self.max_utterance_seconds * bytes_per_sample),
        )

        command, capture_name = self._capture_command()
        rospy.loginfo(
            "offline_asr_endpoint_node ready: %s (%s) -> %s; chunk=%.2fs silence=%.2fs",
            " ".join(command),
            capture_name,
            self.publish_topic,
            self.chunk_seconds,
            self.silence_seconds,
        )

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        pre_roll = deque(maxlen=pre_roll_chunk_count)
        utterance = bytearray()
        voiced_bytes = 0
        speech_started = False
        silent_chunks = 0
        try:
            while not rospy.is_shutdown():
                raw = self._read_audio_chunk(process, chunk_bytes, capture_name)
                if len(raw) < bytes_per_sample:
                    continue
                raw = audioop.mul(raw, bytes_per_sample, self.capture_gain)
                has_voice = audioop.rms(raw, bytes_per_sample) >= self.energy_threshold

                if not speech_started:
                    if has_voice:
                        utterance = bytearray(b"".join(pre_roll))
                        utterance.extend(raw)
                        voiced_bytes = len(raw)
                        speech_started = True
                        silent_chunks = 0
                        rospy.loginfo("Speech started; collecting utterance")
                    else:
                        pre_roll.append(raw)
                    continue

                utterance.extend(raw)
                if has_voice:
                    voiced_bytes += len(raw)
                    silent_chunks = 0
                else:
                    silent_chunks += 1

                phrase_finished = silent_chunks >= silence_chunk_count
                phrase_too_long = len(utterance) >= max_utterance_bytes
                if not phrase_finished and not phrase_too_long:
                    continue

                if voiced_bytes >= min_voice_bytes:
                    self._transcribe_and_publish(bytes(utterance))
                else:
                    rospy.loginfo("Ignoring short audio event")

                pre_roll.clear()
                utterance = bytearray()
                voiced_bytes = 0
                speech_started = False
                silent_chunks = 0
                rospy.loginfo("Waiting for speech")
        finally:
            self._close_capture(process)

    @staticmethod
    def _read_audio_chunk(process, chunk_bytes, capture_name):
        chunks = []
        remaining = chunk_bytes
        while remaining > 0:
            chunk = process.stdout.read(remaining)
            if not chunk:
                error = ""
                if process.stderr is not None:
                    error = process.stderr.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    "%s stopped: %s" % (capture_name, error or "no audio data")
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    @staticmethod
    def _close_capture(process):
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _capture_command(self):
        pulse_requested = self.capture_backend in ("pulse", "pulseaudio")
        pulse_allowed = pulse_requested or (
            self.capture_backend == "auto"
            and self.capture_device in ("", "default")
        )
        if pulse_allowed:
            pulse_command = self._pulse_capture_command()
            if pulse_command:
                return pulse_command, "PulseAudio Xbox source"
            if pulse_requested:
                raise RuntimeError(
                    "PulseAudio capture requested, but no Xbox/NUI source or parec was found"
                )

        return [
            "arecord",
            "-q",
            "-D",
            self.capture_device,
            "-f",
            "S16_LE",
            "-r",
            str(self.sample_rate),
            "-c",
            "1",
            "-t",
            "raw",
        ], "ALSA"

    def _pulse_capture_command(self):
        parec = shutil.which("parec")
        pactl = shutil.which("pactl")
        if not parec or not pactl:
            rospy.logwarn("PulseAudio tools unavailable: pactl=%s parec=%s", pactl, parec)
            return None

        source = self.capture_source or self._find_pulse_source(pactl)
        if not source:
            rospy.logwarn("No Xbox/NUI/Sensor PulseAudio source found")
            return None

        for command in (
            [pactl, "set-source-mute", source, "0"],
            [pactl, "set-source-volume", source, self.capture_volume],
            [pactl, "set-default-source", source],
        ):
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode != 0:
                rospy.logwarn("PulseAudio command failed: %s", " ".join(command))

        rospy.loginfo(
            "Using PulseAudio source: %s (volume=%s, software_gain=%.2fx)",
            source,
            self.capture_volume,
            self.capture_gain,
        )
        return [
            parec,
            "--device",
            source,
            "--format=s16le",
            "--rate",
            str(self.sample_rate),
            "--channels",
            "1",
            "--raw",
        ]

    @staticmethod
    def _find_pulse_source(pactl):
        result = subprocess.run(
            [pactl, "list", "short", "sources"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) < 2 or fields[1].endswith(".monitor"):
                continue
            source_name = fields[1]
            lowered = source_name.lower()
            if any(token in lowered for token in ("xbox", "nui", "sensor")):
                return source_name
        return ""

    def _transcribe_and_publish(self, raw_audio):
        wav_file = self._create_temp_wav()
        wav_path = wav_file.name
        wav_file.close()
        try:
            with wave.open(wav_path, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(self.sample_rate)
                output.writeframes(raw_audio)

            rospy.loginfo("Endpoint detected; transcribing audio")
            segments, _info = self.model.transcribe(
                wav_path,
                language=self.language,
                vad_filter=True,
                beam_size=self.beam_size,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            if text:
                self._publish_text(text)
            else:
                rospy.loginfo("Whisper returned empty text")
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    @staticmethod
    def _create_temp_wav():
        for directory in ("/dev/shm", tempfile.gettempdir()):
            try:
                return tempfile.NamedTemporaryFile(
                    prefix="offline_asr_phrase_",
                    suffix=".wav",
                    dir=directory,
                    delete=False,
                )
            except OSError:
                continue
        raise RuntimeError("Unable to create a temporary WAV file")

    def _publish_text(self, text):
        message = String(data=text)
        self.pub.publish(message)
        if self.legacy_pub:
            self.legacy_pub.publish(message)
        rospy.loginfo("Recognized: %s", text)


if __name__ == "__main__":
    rospy.init_node("offline_asr_endpoint_node")
    node = OfflineAsrEndpointNode()
    node.run()
