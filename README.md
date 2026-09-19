# WPB Task 1 Owner Search Real Robot

This package migrates the simulation owner-search workflow to the real WPB Home robot through owner action recognition.

Current scope:

```text
real hardware bringup
  -> map localization and move_base navigation
  -> YOLO-World person detection on GPU
  -> InsightFace owner verification on CPU
  -> run the same owner-search task at living_room, kitchen, bedroom, then canteen
  -> center verified owner in the Kinect camera
  -> sample nine camera frames across 5 seconds
  -> use YOLO-Pose first for waving and sudden-fall transitions
  -> send three representative frames to Qwen only for static actions
  -> speak the owner's action
  -> if the owner is waving, approach with Kinect point cloud and ask for help
  -> if the owner is lying, split floor fall vs sofa/bed/chair lying by point-cloud height
  -> approach fallen/lying/sitting owners by odometry; run the arm assist only for falls
  -> after normal sitting/elevated lying approach, listen for and report an electrical-switch command
```

Not included yet:

```text
simulation action hint
Gazebo-specific topics
```

## Required Real Robot Inputs

The launch file starts the same core drivers used by the WPB Home examples:

- `/dev/ftdi` through `wpb_home_bringup/wp_home_core` for `/odom` and `/cmd_vel`
- `/dev/rplidar` through `rplidar_ros`, filtered to `/scan`
- `kinect2_bridge` for `/kinect2/qhd/image_color_rect`
- `jie_ware/lidar_loc` localization, `move_base`, and `wpbh_local_planner`
- Offline voice bridge with PiperTTS on `/voice/say` and `sound_play` playback; the switch-command step runs `tools/local_switch_command_test.py` as a standalone subprocess by default
- asynchronous YOLO-World person detection on `cuda:0`; the local viewer
  overlays the latest detections on the latest raw Kinect frame without
  requiring a second annotated image topic

The task node waits for `/kinect2/qhd/image_color_rect`, `/scan`, and `/odom` before it starts moving. This is intentional for real robot safety.

Speech output is routed through `offline_voice_bridge`, not the xfyun stack. The task publishes prompt text to `/voice/say`; `offline_tts_node.py` uses PiperTTS to generate a wav and `sound_play` plays it through the audio device on the machine running the launch file. The default real-robot launch loads the owner's reference photos from `data/owner/` and starts searching immediately. The optional runtime enrollment stage remains available by setting `owner_enrollment_enabled:=true`.

The real launch uses an asynchronous YOLO worker and a latest-frame debug viewer.
Incoming camera frames are not queued behind a slow inference; stale frames are
dropped while the viewer always renders the newest original Kinect frame with
the latest available detection boxes. The display is not resized or routed
through the detector's `imgsz`, so `yolo_imgsz` affects inference workload only,
not display quality. The viewer continuously repaints its cached frame and
recreates OpenCV windows after a prolonged GUI failure. The detector keeps its
worker alive after frame-processing errors, releases CUDA cache during
recovery, and the launch file respawns the process if it exits unexpectedly.
Set `yolo_publish_debug:=true` only when another node needs the full annotated
debug image topic; the local viewer does not need that duplicate image stream.

## Files To Prepare

Put or update the upright owner reference photos here:

```bash
/home/ubuntu20/catkin_ws/src/wpb_task1_owner_search/data/owner/
```

The compatibility face verifier accepts either one image file or a directory of
images. For better recognition from side views and downward/upward angles, place
several single-person upright photos in this directory, for example
`owner_front.jpg`, `owner_left.jpg`, `owner_right.jpg`, and `owner_down.jpg`. Do
not use this directory for lying-owner calibration photos; keep those outside the
package for validation so they do not broaden the runtime reference set. Avoid
group photos or photos where the owner's face is very small; the loader uses the
largest detected face in each reference photo.

During verification the node chooses the face crop from the YOLO person-box shape. If the box is taller than wide, it treats the person as upright and checks only the upper crop without rotation. If the box is wider than tall, it treats the person as lying and checks left/right side crops, because the face is often at one horizontal end of the body box.

For lying candidates, the verifier tries rotated versions of each candidate face crop, so sideways faces can still be matched against the upright owner references. Reference-photo rotation is disabled by default because it can create unstable high-similarity matches for sideways non-owners. If the first raw pass is still below threshold, the verifier uses CLAHE contrast normalization and 1.5x upsampling as a fallback for the lying side crops.

Prepare a real robot map and waypoint file:

```bash
/home/ubuntu20/catkin_ws/src/wpb_home/wpb_home_tutorials/maps/map.yaml
/home/ubuntu20/catkin_ws/src/wpb_home/wpb_home_tutorials/maps/map.pgm
/home/ubuntu20/waypoints.xml
```

Keep the map files in the real WPB Home tutorials map directory. Do not reuse the simulation package map directory.

If `map.yaml` and `map.pgm` are in the same directory, keep the YAML image entry relative:

```yaml
image: map.pgm
```

The waypoint file must contain these task waypoints:

```text
living_room
kitchen
bedroom
canteen
```

After those room tasks finish, the robot navigates to `exit`. If the saved exit
waypoint is still named `1`, the task node renames that waypoint to `exit` on
startup when the waypoint file is writable; it also treats `1` as an `exit`
alias so the final navigation still works if the rename cannot be written.

If your map is temporarily elsewhere, pass it through the `map:=...` launch argument.

## Run

Recommended for repeated real-robot tests: start the robot/camera/YOLO stack once, then rerun only the task node.

For the voice enrollment and owner-search launch, RViz is started by default so
you can initialize the robot pose before navigation:

```bash
roslaunch wpb_task1_owner_search owner_voice_reid_test.launch
```

In RViz, choose `2D Pose Estimate`, click the robot's actual position on the
map, and drag in the robot's facing direction. This publishes `/initialpose`
for the active localization node. The launch reuses the existing task Kinect
node and starts the base, lidar, map, localization, move_base, waypoint
manager, and RViz only once. If those navigation nodes are already running,
use `start_navigation:=false` and keep `start_rviz:=true`.

During enrollment, the saved owner crop keeps a larger top margin than the
other sides so the upper face and hair are not cut off. This does not change
YOLO inference frequency or the Re-ID/InsightFace pipeline; tune it with
`crop_top_padding:=0.25` only if the camera is mounted unusually high.

The first enrollment phase now starts from a front view and asks the owner to
turn slowly toward the side while capturing up to `10` samples across `4.5`
seconds. A second side-view prompt and capture phase remains afterward, so the
profile contains both the continuous front-to-side transition and a stable
side reference.

After reaching `living_room`, owner search uses short scan steps instead of
continuous rotation: it turns `0.18` radians, stops for `0.35` seconds, and
then continues. Adjust `scan_step_angle` or `scan_pause` if the camera needs
more stabilization time.

For lying-owner recognition, the query path now treats InsightFace as the
primary identity signal, searches both sides of a horizontal person crop,
tries rotated views, and only performs an enlarged retry when the original
face search fails. The body Re-ID score remains an auxiliary check. The launch
also accepts two lying matches within a `1.5` second window, so one dropped
frame does not immediately clear the result. Confirm the startup status shows
`face_model_ready=true` and `face_ready=true`; if `face_ready=false`, the
loaded profile has no saved face embedding and must be registered again with
InsightFace enabled.

Terminal 1, after robot PC reboot or after fully stopping the stack:

```bash
cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch wpb_task1_owner_search task1_owner_search_bringup.launch
```

Wait until the camera, point-cloud, and detection topics are alive:

```bash
rostopic hz /kinect2/qhd/image_color_rect
rostopic hz /kinect2/qhd/points
rostopic hz /perception/person_detections_2d
```

The annotated `/perception/yoloworld/debug_image` topic is disabled by default
to avoid copying and publishing a full camera image twice. Enable it with
`yolo_publish_debug:=true` when an external subscriber needs that topic.

Terminal 2, rerun this command for each attempt:

```bash
cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch wpb_task1_owner_search task1_owner_search_task_only.launch
```

The recommended two-terminal flow starts TTS/sound playback in
`task1_owner_search_bringup.launch`. Background ASR is disabled by default to
avoid competing with YOLO, pose estimation, and Qwen; the task's default
`direct_asr` switch-command path loads its own recognizer only when needed. If
only `task_only` is running, start its voice chain explicitly:

```bash
roslaunch wpb_task1_owner_search task1_owner_search_task_only.launch start_voice:=true
```

If runtime owner enrollment or the ROS-topic ASR path is explicitly enabled,
start `start_asr:=true` and make sure no second microphone capture process is
running at the same time.

The real-robot launch now uses the Xbox source through PulseAudio directly,
because the desktop PulseAudio service owns the sensor capture node. The ASR
node automatically finds `alsa_input.*Xbox_NUI_Sensor*` and falls back to ALSA
only when PulseAudio is unavailable. It sets the source to `150%` and applies a
software gain of `2.0x` before voice detection. You can also pass the exact
source:

```bash
roslaunch wpb_task1_owner_search task1_owner_search_real.launch \
  asr_capture_backend:=pulse \
  asr_capture_source:=alsa_input.usb-Microsoft_Xbox_NUI_Sensor_012548143447-02.multichannel-input
```

To make the microphone louder or quieter, change these launch arguments:

```bash
roslaunch wpb_task1_owner_search task1_owner_search_real.launch \
  asr_capture_volume:=200% \
  asr_capture_gain:=2.5
```

Start with `150%` and `2.0`; if the audio becomes distorted or recognition gets
worse, reduce `asr_capture_gain` to `1.5` or `asr_capture_volume` to `125%`.

If the Xbox source appears in the input list but its level does not move, repair
the host audio route without stopping the robot:

```bash
bash ~/catkin_ws/src/wpb_task1_owner_search/tools/repair_xbox_nui_audio.sh
```

The helper finds the Xbox/NUI/Sensor PulseAudio source, unmutes it, sets its
volume to 100%, makes it the default input, and records five seconds through
the shared `default` device to report RMS and peak levels. Run it in the
robot-host user's normal desktop terminal, not inside a restricted container.

If the robot is already at the living room and you only want to test owner search and action recognition:

```bash
roslaunch wpb_task1_owner_search task1_owner_search_task_only.launch navigate_enabled:=false
```

Single-command full launch is still available, but avoid using it repeatedly in a tight loop because it restarts Kinect2, point-cloud nodelets, YOLO, voice, RViz, and the task every time:

```bash
cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch wpb_task1_owner_search task1_owner_search_real.launch
```

By default this opens the YOLO person-box debug window and does not show a separate raw Kinect window. `Owner YOLO-Person Box Real` shows the YOLO-World person boxes on the camera image. If you are running headless over SSH, no OpenCV window will appear unless `DISPLAY` is available; in that case view the topic in RViz/ImageView instead:

```bash
rosrun image_view image_view image:=/perception/yoloworld/debug_image
```

To disable the viewer window while keeping YOLO detections running:

```bash
roslaunch wpb_task1_owner_search task1_owner_search_real.launch show_yolo_viewer:=false
```

If the robot base, lidar, Kinect, localization, and move_base are already running:

```bash
roslaunch wpb_task1_owner_search task1_owner_search_real.launch start_robot:=false
```

The real launch defaults to `jie_ware/lidar_loc`, which publishes `map -> odom` from the map and `/scan`. To compare with the original AMCL path, run:

```bash
roslaunch wpb_task1_owner_search task1_owner_search_real.launch use_jie_lidar_loc:=false
```

If the robot is already at the living room and you only want to test owner search:

```bash
roslaunch wpb_task1_owner_search task1_owner_search_real.launch start_robot:=false navigate_enabled:=false
```

For hardware debug only, if you want to test scanning without owner face verification, edit `config/task1_owner_search_real.yaml`:

```yaml
face_verify_required: false
allow_unverified_owner: true
```

Do not use that setting in the final competition flow because the robot may accept a guest as the owner.

## Owner Action Recognition

After InsightFace confirms the owner, the robot tries to center the owner in the Kinect image. If centering is unstable or times out, it silently skips centering, says `识别中。`, samples `/kinect2/qhd/image_color_rect` for five seconds, and announces the detected action.

The action recognizer uses the verified local Qwen vision logic from
`offline_voice_bridge/scripts/qwen_action_recognition_node.py` together with its
YOLO-pose helper. It samples nine full-camera frames over five seconds and runs
YOLO-pose first on all nine frames. A pose-confirmed `waving` or
`sudden_fall` result is returned directly without sending the action frames to
Qwen. When neither dynamic action is found, three representative JPEG frames go
to the local Ollama Qwen vision model for sitting, lying, and already-fallen
recognition, while the existing organized Kinect point-cloud support-surface
analysis remains in place. During action recognition the YOLO-World detector is
paused to avoid GPU contention. The Qwen request is warmed up before the task
reaches the owner; a second request is never started while warm-up is still
active.
The default action images are resized to 320 pixels wide and the action request
uses a 4096-token context. This is intentional: the Qwen vision encoder can
consume thousands of tokens for one full-size Kinect image, and the older 2048
context limit rejects a multi-frame request before inference starts. The
0.8B Qwen vision request defaults to `num_gpu: 0` so it does not compete with
YOLO-World for GPU memory; override `action_llm_num_gpu` only if the machine
has enough VRAM and the slower CPU path is unacceptable.

The configured Qwen model must support image input:

```bash
ollama list
roslaunch wpb_task1_owner_search task1_owner_search_real.launch \
  action_llm_model:=qwen3.5:0.8b
```

Pose model path:

```bash
/home/ubuntu20/catkin_ws/src/wpr_task1_owner_search/models/pose/yolo11n-pose.pt
```

Current supported action announcements:

- owner is sitting
- owner is lying down on sofa/bed/chair
- owner may have suddenly fallen
- owner is already lying on the floor after a fall
- owner is waving
- action is uncertain

Fall detection is treated as two related states: `sudden_fall` means the camera
saw an active upright-to-floor transition, while `fallen` means the owner is
already on the floor when observed. A horizontal body on a chair, sofa, or bed
is retained as `lying`; only a confident floor-relative point-cloud result is
reported as `fallen`.

Sitting, lying, floor fall, sudden fall, and waving are all decided by the new
Qwen/pose/point-cloud fusion result. Waving keeps priority over sitting or
lying when raised-wrist motion is present. Sitting and lying speech includes
the visible chair, sofa, or bed when Qwen can identify it.

When the detected action is `waving`, the real robot does not use simulation-only model-state hints or a Gazebo 3D goal. It samples `/kinect2/qhd/points` inside the verified owner's Kinect 2D person box, estimates the owner's 3D position relative to the robot, converts that relative offset from `base_footprint` into a `map` goal with TF, then sends a `move_base` goal to reach the configured standoff distance. The default waving standoff is 0.45 m with 0.05 m finish tolerance, keeping the final target within 0.50 m before asking `请问您需要什么帮助？`, then waiting `waving_help_pause_seconds` before moving to the next waypoint.

The waving approach defaults to a 25-second owner-position sampling window before handing the obstacle-avoiding approach to `move_base`. It stops and cancels the action as soon as the robot has moved into the configured `waving_approach_safety_radius` (default 0.48 m), then starts the voice interaction instead of continuing to chase the final goal. A short stationary finish is accepted only after odometry confirms that the robot moved. `waving_approach_plan_detour_ratio`, `waving_approach_plan_detour_margin`, and `waving_approach_plan_turn_limit` reject unusually large or unstable paths before sending a candidate goal. Waving navigation does not retry the same goal after clearing costmaps unless `waving_approach_retry_after_clear` is explicitly enabled. `approach_navigation_enabled: true` is the normal path; `approach_direct_fallback_enabled: false` prevents the robot from reverting to blind forward motion if `move_base` cannot plan the near-owner approach. If it cannot finish, the node logs the concrete reason and does not ask the help prompt from a far position.

When the detected action is `sudden_fall` or `fallen`, the robot announces the
fall result, snapshots the owner's 3D position, approaches by the existing
approach logic, then advances a short extra distance and runs the arm assist
motion. The new action labels are mapped into the existing fall-approach and
arm-approach branches; navigation, lidar guards, move_base goals, and arm
commands are not replaced.

Fall, non-fall lying, and sitting states use the same snapshot approach mode: the robot records a single relative 3D target and, by default, transforms the standoff point into the `map` frame before sending it to `move_base` instead of manually driving the measured distance by wheel odometry. The extra-close nudge also uses a transformed `map` goal first, so it participates in obstacle avoidance; direct `/cmd_vel` movement is only used if `approach_navigation_enabled` is disabled or `approach_direct_fallback_enabled` is explicitly enabled. `/scan` remains active as a forward safety guard. `fall_approach_fast_finish_tolerance` and `fall_approach_extra_close_finish_tolerance` prevent the final near-owner nudge from crawling for the last few centimeters. Fall and floor-lying cases always complete the `/wpb_home/mani_ctrl` arm sequence before leaving the waypoint: extend with `name=['lift','gripper']`, hold briefly, retract, then wait `fall_assist_arm_completion_wait`.

For a normal `sitting` or elevated `lying` owner, the robot enters the electrical-switch voice interaction even if the approach or extra forward nudge is blocked, stuck, or cannot complete; it stops the base first, then says `请指示。`. In the default `direct_asr` mode, the ready ding is played first, recording starts immediately after the ding finishes, and each recording window is 5 seconds. The owner should start speaking as soon as the ding ends and finish the command within that 5-second window. The recorded PCM is software-amplified before faster-whisper (`electrical_switch_asr_input_gain: 3.0`, auto gain target peak 70%, max gain 8.0), so quieter sitting/lying speech is easier to recognize. The task node classifies the transcript with keyword rules plus the local Ollama endpoint, records `on`, `off`, or `unknown`, and publishes it latched on `/electrical_switch/state`. `fallen` and `sudden_fall` paths do not enter this interaction and retain the arm-assist behavior.

The optional standalone script mode loops through the configured instruction window until one round produces a switch judgment. Set `electrical_switch_script_until_result: false`, or switch `electrical_switch_instruction_source` to `ros_topic`, only if you want the older one-shot/topic recognizer behavior.

## Useful Checks

```bash
rostopic hz /odom
rostopic hz /scan
rostopic hz /kinect2/qhd/points
rosrun tf tf_echo map odom
rosnode list | grep -E 'lidar_loc|amcl'
rostopic hz /kinect2/qhd/image_color_rect
rostopic echo /perception/person_detections_2d
rostopic hz /perception/yoloworld/debug_image
watch -n 1 nvidia-smi
rostopic info /voice/say
rostopic info /voice/asr_text
rostopic echo /electrical_switch/state
```

## Troubleshooting Repeated Runs

If `task1_find_owner_real` exits while waiting for hardware and the log says no messages arrived from `/kinect2/qhd/image_color_rect`, the Kinect topic may be advertised but publishing at 0 Hz. Check it before running the task:

```bash
rostopic hz /kinect2/qhd/image_color_rect
```

If the rate stays at 0 Hz, fully stop the launch, restart `kinect2_bridge`, or unplug/replug the Kinect USB cable before starting the task again.

For cleanup after repeated interrupted runs:

```bash
~/catkin_ws/src/wpb_task1_owner_search/tools/task1_cleanup_runtime.sh
```

If `nvidia-smi` cannot communicate with the NVIDIA driver, YOLO on `cuda:0` will not be reliable. Reboot the robot PC or fix the NVIDIA driver state before testing YOLO again.

If `move_base` reports `Failed to get a plan`, confirm that `/home/ubuntu20/waypoints.xml` still contains the intended `living_room` pose. A bad or accidentally re-saved waypoint near an obstacle can prevent planning even when localization and lidar are working.

Quick speech test:

```bash
rostopic pub -1 /voice/say std_msgs/String "data: '我已经识别到主人。'"
```

Raw robot microphone and Chinese ASR test:

```bash
rosrun wpb_task1_owner_search robot_mic_test.py --list-devices --seconds 12
```

The test first loads the local faster-whisper Chinese model, then captures
audio in 4-second windows. Speak near the robot after the command starts. If
the microphone is receiving audio, the printed RMS value should jump and lines
should show `VOICE`; each active window also prints `识别文本`. To test only
the microphone level without loading ASR:

```bash
rosrun wpb_task1_owner_search robot_mic_test.py --no-asr --seconds 10
```

To save the captured audio for playback:

```bash
rosrun wpb_task1_owner_search robot_mic_test.py --seconds 10 --save-wav /tmp/robot_mic_test.wav
aplay /tmp/robot_mic_test.wav
```

If `default` does not work but `arecord -l` shows another capture card, pass it
explicitly:

```bash
rosrun wpb_task1_owner_search robot_mic_test.py --device plughw:CARD=Generic_1,DEV=0 --seconds 10
```

Expected task completion log:

```text
Owner accepted at bbox=..., confidence=..., reason=face accepted
Owner centered in camera: center=...
Recognizing owner action for 5.0s with robot camera
Captured owner action frame 9/9
Owner dynamic action verdict: action=waving place=unknown confidence=... features=... frames=9 total=...
```

For a static action, the remaining log includes the Qwen request:

```text
Sending owner action request to Qwen: model=... images=3 ... num_ctx=4096
Qwen owner action response received in ...s
Owner action verdict: action=... place=... qwen=... used=... pose_dynamic=... ground=... frames=9/3 total=...
task1_find_owner_real finished: success=True
```

If the log stops after `Captured owner action frame 9/9`, inspect the next
stage log. `Owner dynamic action verdict` means YOLO-Pose completed the decision
without Qwen. `Sending owner action request to Qwen` means the dynamic gate did
not fire and the node is waiting for Ollama. The node automatically retries a
multi-frame context error with the latest frame, while retaining all nine frames
for pose-based waving and sudden-fall detection.

## Person Re-ID Owner Test

This lightweight test is separate from the full navigation task. It opens the Kinect RGB stream, uses YOLO-World person boxes, records several full-body crops at the ding cue, builds an owner Re-ID embedding, then keeps watching the camera and says `识别到主人` when the same person appears again.

### GitHub Re-ID setup

The package vendors Torchreid (`KaiyangZhou/deep-person-reid`) under:

```bash
/home/ubuntu20/catkin_ws/src/wpb_task1_owner_search/third_party/deep-person-reid
```

Run the dependency setup on the robot Python environment:

```bash
cd /home/ubuntu20/catkin_ws/src/wpb_task1_owner_search
./tools/setup_person_reid.sh
```

For a lightweight Re-ID-trained OSNet-x0.25 weight from the Torchreid model zoo, run:

```bash
DOWNLOAD_REID_WEIGHT=1 ./tools/setup_person_reid.sh
```

If the weight download succeeds, pass it explicitly at launch time:

```bash
roslaunch wpb_task1_owner_search person_reid_owner_test.launch \
  reid_model_path:=/home/ubuntu20/catkin_ws/src/wpb_task1_owner_search/models/reid/osnet_x0_25_msmt17.pth
```

Without `reid_model_path`, Torchreid will use its built-in pretrained initialization path for the selected model, which may need network access the first time.

### Run the camera + voice test

```bash
cd /home/ubuntu20/catkin_ws
catkin_make
source devel/setup.bash
roslaunch wpb_task1_owner_search person_reid_owner_test.launch
```

Default behavior:

1. Waits for `/kinect2/qhd/image_color_rect` and `/perception/person_detections_2d`.
2. Says `正在记录`.
3. Plays a short ding through `sound_play` and starts sampling person crops.
4. Saves crops and `owner_profile.npz` under `data/reid_owner/`.
5. Says `记录结束`.
6. Announces `识别到主人` when the live Re-ID score stays above threshold for consecutive frames.

Useful launch overrides:

```bash
roslaunch wpb_task1_owner_search person_reid_owner_test.launch \
  start_camera:=false \
  start_yolo:=false \
  reuse_existing_profile:=true \
  match_threshold:=0.72 \
  reid_device:=cuda:0
```

Use `start_camera:=false` or `start_yolo:=false` when those nodes are already running. Use `allow_color_fallback:=true` only for camera/voice smoke tests when Torchreid dependencies are not installed; it is not real person re-identification.
