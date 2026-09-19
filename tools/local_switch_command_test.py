#!/usr/bin/env python3
# coding: utf-8
"""Run offline_voice_bridge's standalone switch-command test from this package."""

import importlib.util
import math
import os
import re
import runpy
import struct
import sys
import wave


UNTIL_RESULT_FLAG = "--until-result"
DEFAULT_READY_DING_WAV = "/dev/shm/local_switch_ready_ding.wav"
DEFAULT_QWEN_MODEL = "qwen3.5:0.8b"


def env_flag(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "disabled")


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def ensure_ready_ding_wav():
    wav_path = os.path.expanduser(os.environ.get("LOCAL_SWITCH_READY_DING_WAV", DEFAULT_READY_DING_WAV))
    sample_rate = max(8000, int(env_float("LOCAL_SWITCH_READY_DING_SAMPLE_RATE", 16000)))
    duration = max(0.03, env_float("LOCAL_SWITCH_READY_DING_SECONDS", 0.16))
    frequency = max(100.0, env_float("LOCAL_SWITCH_READY_DING_FREQUENCY", 880.0))
    volume = max(0.0, min(1.0, env_float("LOCAL_SWITCH_READY_DING_VOLUME", 0.65)))
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

    out_dir = os.path.dirname(wav_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames))
    return wav_path


def play_ready_ding(module, args):
    if args.no_play or not env_flag("LOCAL_SWITCH_READY_DING", True):
        return
    try:
        wav_path = ensure_ready_ding_wav()
        player = os.environ.get("LOCAL_SWITCH_READY_DING_PLAYER") or args.player
        speaker_device = os.environ.get("LOCAL_SWITCH_READY_DING_SPEAKER_DEVICE") or args.speaker_device
        module.local_melo_tts_zh.play_wav(wav_path, player, speaker_device)
    except Exception as exc:
        print("WARN: 录音提示音播放失败: %s" % exc)


def force_ollama_cpu_options(module):
    original_call_llm = getattr(module, "call_llm", None)
    if original_call_llm is None or getattr(original_call_llm, "_task1_cpu_wrapped", False):
        return

    def cpu_call_llm(args, transcript):
        payload = {
            "model": args.llm_model,
            "stream": False,
            "think": False,
            "format": "json",
            "keep_alive": args.llm_keep_alive,
            "messages": [{"role": "user", "content": module.LLM_PROMPT_TEMPLATE % transcript}],
            "options": {
                "temperature": 0,
                "top_p": 0.7,
                "num_predict": args.llm_max_tokens,
                "num_ctx": 1024,
                "num_gpu": 0,
            },
        }
        request = module.urllib.request.Request(
            args.llm_url,
            data=module.json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = module.time.time()
        try:
            with module.urllib.request.urlopen(request, timeout=args.llm_timeout) as response:
                raw = response.read().decode("utf-8")
        except TimeoutError as exc:
            raise RuntimeError(
                "LLM 请求超时（%.1f 秒）。首次运行会加载模型，可重试或调大 --llm-timeout。%s"
                % (module.time.time() - start, exc)
            )
        except module.urllib.error.URLError as exc:
            raise RuntimeError("LLM 请求失败，请确认 Ollama 正在运行且已拉取 %s: %s" % (args.llm_model, exc))

        result = module.json.loads(raw)
        content = result.get("message", {}).get("content", "").strip()
        if not content:
            raise RuntimeError("LLM 返回空内容: %s" % raw)
        action = module.parse_action(content)
        return action, content, module.time.time() - start

    cpu_call_llm._task1_cpu_wrapped = True
    module.call_llm = cpu_call_llm


def keyword_fallback(transcript):
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


def find_external_script():
    here = os.path.dirname(os.path.abspath(__file__))
    catkin_src_dir = os.path.abspath(os.path.join(here, "..", ".."))
    candidates = []

    override = os.environ.get("LOCAL_SWITCH_COMMAND_TEST")
    if override:
        candidates.append(os.path.expanduser(override))
    candidates.append(
        os.path.join(catkin_src_dir, "offline_voice_bridge", "tools", "local_switch_command_test.py")
    )

    for script_path in candidates:
        script_path = os.path.abspath(script_path)
        if os.path.exists(script_path):
            return script_path
    return None


def load_external_script(script_path):
    tools_dir = os.path.dirname(script_path)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    spec = importlib.util.spec_from_file_location("offline_local_switch_command_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load %s" % script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_default_qwen_model():
    if any(
        argument == "--llm-model" or argument.startswith("--llm-model=")
        for argument in sys.argv[1:]
    ):
        return
    sys.argv.extend(["--llm-model", DEFAULT_QWEN_MODEL])


def run_until_result(script_path):
    module = load_external_script(script_path)
    force_ollama_cpu_options(module)
    args = module.build_parser().parse_args()
    using_microphone = not args.input_text and not args.input_wav

    try:
        if not args.input_text:
            asr_model = module.load_asr_model(args)
            tts_ctx = module.load_tts(args)
        else:
            asr_model = None
            tts_ctx = None
        if not args.no_llm_warmup:
            module.warmup_llm(args)
        print("模型已预加载。开始测试；未识别到文本会自动进入下一轮。")

        round_index = 1
        while True:
            print("\n=== 第 %d 轮 ===" % round_index)
            if args.input_text:
                transcript = args.input_text.strip()
                print("输入文本: %s" % transcript)
            else:
                if using_microphone:
                    play_ready_ding(module, args)
                transcript = module.transcribe(args, asr_model, round_index)

            if not transcript:
                print("没有识别到文本，跳过本轮。")
                if not using_microphone:
                    return 2
                round_index += 1
                continue

            action, source = module.classify(args, transcript)
            if action == "unknown":
                fallback_action = keyword_fallback(transcript)
                if fallback_action in ("on", "off"):
                    action = fallback_action
                    source = "%s+local_keyword" % source
            print("判断结果: %s (来源: %s)" % (action, source))
            if action == "unknown":
                print("没有听出开关指令，继续下一轮。")
                if not using_microphone:
                    return 2
                round_index += 1
                continue
            if tts_ctx is not None:
                module.speak_result(args, action, tts_ctx)
            return 0
    except KeyboardInterrupt:
        print("\n已退出连续测试。")
        return 0
    except Exception as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1


def main():
    until_result = UNTIL_RESULT_FLAG in sys.argv[1:]
    if until_result:
        sys.argv = [arg for arg in sys.argv if arg != UNTIL_RESULT_FLAG]
    ensure_default_qwen_model()

    script_path = find_external_script()
    if script_path:
        if until_result:
            return run_until_result(script_path)
        sys.path.insert(0, os.path.dirname(script_path))
        sys.argv[0] = script_path
        runpy.run_path(script_path, run_name="__main__")
        return 0

    sys.stderr.write(
        "ERROR: cannot find offline_voice_bridge/tools/local_switch_command_test.py\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
