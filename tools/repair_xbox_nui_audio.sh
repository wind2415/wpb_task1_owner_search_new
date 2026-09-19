#!/usr/bin/env bash
set -euo pipefail

pattern='xbox|nui|sensor'
source_name="${1:-}"
source_volume="${XBOX_NUI_SOURCE_VOLUME:-150%}"

if ! command -v pactl >/dev/null 2>&1; then
    echo "错误：找不到 pactl。请在机器人主机的普通终端安装 pulseaudio-utils 后重试。" >&2
    exit 2
fi

if ! command -v arecord >/dev/null 2>&1; then
    echo "错误：找不到 arecord。请在机器人主机的普通终端安装 alsa-utils 后重试。" >&2
    exit 2
fi

if [[ -z "$source_name" ]]; then
    source_name="$(
        pactl list short sources |
            awk -v pattern="$pattern" '{ line = tolower($0); if (line ~ pattern && $2 !~ /\.monitor$/) { print $2; exit } }'
    )"
fi

if [[ -z "$source_name" ]]; then
    echo "错误：没有找到 Xbox/NUI/Sensor 的 PulseAudio 输入源。" >&2
    echo "当前输入源：" >&2
    pactl list short sources >&2
    exit 3
fi

echo "选择输入源: $source_name"
pactl set-source-mute "$source_name" 0
pactl set-source-volume "$source_name" "$source_volume"
pactl set-default-source "$source_name"
pactl suspend-source "$source_name" 0 2>/dev/null || true

echo
echo "已完成：解静音、音量 $source_volume、设为默认输入。当前状态："
pactl list short sources | awk -v source="$source_name" '$2 == source'

tmp_file="$(mktemp --suffix=.s16le)"
trap 'rm -f "$tmp_file"' EXIT

echo
echo "开始录音 5 秒，请对着机器人麦克风说话..."
capture_status=0
if command -v parec >/dev/null 2>&1; then
    set +e
    timeout 7s parec \
        --device="$source_name" \
        --format=s16le \
        --rate=16000 \
        --channels=1 \
        --raw \
        --file-format=raw \
        > "$tmp_file"
    capture_status=$?
    set -e
else
    set +e
    timeout 7s arecord -q \
        -D default \
        -f S16_LE \
        -r 16000 \
        -c 1 \
        -d 5 \
        -t raw \
        "$tmp_file"
    capture_status=$?
    set -e
fi

if [[ "$capture_status" -ne 0 && "$capture_status" -ne 124 && "$capture_status" -ne 143 ]]; then
    echo "录音失败，退出码: $capture_status" >&2
    exit "$capture_status"
fi

python3 - "$tmp_file" <<'PY'
import audioop
import sys

path = sys.argv[1]
with open(path, "rb") as stream:
    raw = stream.read()

rms = audioop.rms(raw, 2) if raw else 0
peak = audioop.max(raw, 2) if raw else 0
print("录音字节数: %d" % len(raw))
print("RMS: %d" % rms)
print("Peak: %d" % peak)
if peak == 0:
    print("结果：仍然没有音频样本；请检查 Xbox NUI Sensor 的硬件静音/麦克风权限。")
elif rms < 30:
    print("结果：已采集到样本，但电平很低；请把输入音量提高或靠近麦克风讲话。")
else:
    print("结果：Xbox NUI Sensor 已有声音输入。")
PY
