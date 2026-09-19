# wpb_task1_owner_search 工程架构与主流程技术文档

本文档基于当前工程目录中的 `README.md`、`launch/*.launch`、`config/task1_owner_search_real.yaml`、`scripts/task1_find_owner_real.py`、`scripts/yoloworld_async.py` 与 `scripts/yoloworld_debug_viewer_stable.py` 等文件整理，重点解释该包在 ROS/Catkin 工作空间中的定位、软件架构、启动链路、主节点功能与数据流。

## 1. 工程定位

`wpb_task1_owner_search` 是一个面向真实 WPB Home 机器人（无模拟器）的“主人寻访与动作识别”任务包。它不是仅提供单一 API 的普通视觉识别工程，而是一个完整的 ROS 任务编排软件：

- 负责在地图上下文中按房间 waypoint 依次导航；
- 使用YOLO-World执行真实摄像头中的人物目标检测；
- 使用 InsightFace 从参考图片中构建主人脸特征库，做验证；
- 使用 Qwen / YOLO-Pose / 点云分析器识别主人动作；
- 对不同动作（挥手、跌倒、躺卧、坐着）采用不同近距接近、智能交互与语音/开关指令路径。

该工程的重点不是纯图像识别，而是“一个真实机器人任务闭环”：感知 -> 核心任务状态流转 -> 定位/导航 -> 自主近距接近 -> 动作识别 -> 语音/电气互动。

## 2. 工程目录结构

典型目录如下：

```text
wpb_task1_owner_search/
├── CMakeLists.txt              # catkin 安装声明
├── package.xml                 # ROS package 依赖声明
├── README.md                   # 运行使用说明
├── launch/                     # 启动文件
├── config/                     # YAML 参数配置
├── scripts/                    # Python 主节点、YOLO 工作线程、调试视图
├── tools/                      # 本地开关脚本、设备修复脚本等
├── data/owner/                 # 参考主人照片目录
├── maps/                       # 地图文件
└── docs/                       # 文档目录
```

其中最重要的代码文件：

- `scripts/task1_find_owner_real.py`：主任务节点，负责整体任务状态机；
- `scripts/yoloworld_async.py`：异步 YOLO-World 检测节点；
- `scripts/yoloworld_debug_viewer_stable.py`：YOLO 检测框调试图形窗口；
- `launch/task1_owner_search_real.launch`：真实机启动入口；
- `config/task1_owner_search_real.yaml`：任务全部参数配置中心；
- `tools/local_switch_command_test.py`：电气开关指令交互脚本。

## 3. 依赖关系与外部集成

本包依赖多个 ROS/Catkin 实体。真正的运行是“任务包 + 栈式外部依赖”组合：

1. 基座与底盘：
   - `wpb_home_bringup`：负责 WPB Home 底盘驱动，发布 `/odom`、消费 `/cmd_vel`；
   - `wpb_home_lidar_filter`：对雷达扫描做过滤后发布 `/scan`；
   - `rplidar_ros`：雷达节点；
   - `move_base`：导航控制器；
   - `waterplus_map_tools`：waypoint 管理。

2. 视觉与感知：
   - `kinect2_bridge`：提供颜色图像与点云；
   - `yoloworld_perception`：YOLO-World 人体检测模型提供 `perception/person_detections_2d`；
   - `offline_voice_bridge`：离线语音/ASR/TTS 流；
   - `jie_ware`：雷达定位/区域定位服务；
   - `yoloworld_perception` 与额外模型目录：Action/pose 模型路径。

3. 人脸与动作识别：
   - `InsightFace`：主人脸识别和特征库初始化；
   - `offline_voice_bridge/scripts/qwen_action_recognition_node.py`：提供 Qwen 动作识别能力与 YOLO-Pose/PointCloud Ground Analyzer。

因此，从工程层面看，这个包不依赖单脚本，而是取构成 ROS 系统的多个强耦合部件：导航、定位、图像、点云、语音、ASR、动作识别与开关控制。

## 4. 启动架构

工程的入口以 `launch/task1_owner_search_real.launch` 为主，参数较多，能全面配置实际机器人运行所需的多个节点。

### 4.1 一次性启动方式

典型全栈启动：

```bash
roslaunch wpb_task1_owner_search task1_owner_search_bringup.launch
```

该 launch 会组织：

- `wpb_real_navigation.launch`：启动底盘、雷达、Kinect、Map Server、AMCL/`jie_ware`、`move_base`、`wp_manager`；
- `yoloworld_async.py`：单独启动 YOLO-World 人物检测；
- `yoloworld_debug_viewer_stable.py`：调试窗口；
- `offline_voice_bridge`：离线语音链路；
- `task1_find_owner_real.py`：主任务节点。

### 4.2 启动参数说明

`task1_owner_search_real.launch` 在 ROS 参数层定义了大量任务参数，例如：

- `start_robot`, `start_yolo`, `start_voice`, `start_task`
- `map`, `waypoint_file`, `waypoint_name`
- `owner_image_path`, `owner_enrollment_enabled`
- `action_llm_url`, `action_llm_model`
- `action_frame_count`, `action_sample_seconds`, `action_image_max_width`
- `navigate_enabled`, `start_move_base`

这些参数共同推动一条“真实环境任务运行”的参数化配置量级。`config/task1_owner_search_real.yaml` 中保存了较完整的机器人任务行为参数，而 `launch/task1_owner_search_real.launch` 中再将参数映射到 Python 节点对象的 `rospy.get_param("~...")` 配置项。

## 5. 配置文件结构

配置文件 `config/task1_owner_search_real.yaml` 总体上可以分成几大组：

### 5.1 地图与航点

```yaml
waypoint_name: living_room
task_waypoint_names: [living_room, kitchen, bedroom, canteen]
exit_waypoint_name: exit
exit_waypoint_aliases: ["1"]
waypoint_file: /home/ubuntu20/waypoints.xml
```

- `task_waypoint_names` 决定扫描城市/房间环路；
- `exit_waypoint_name` 决定最终返回出口/停靠点；
- `waypoint_file` 与地图 / `wp_manager` 一起消费，形成导航地标。

### 5.2 感知与相机主题

```yaml
image_topic: /kinect2/qhd/image_color_rect
detections_topic: /perception/person_detections_2d
points_topic: /kinect2/qhd/points
scan_topic: /scan
odom_topic: /odom
cmd_vel_topic: /cmd_vel
say_topic: /voice/say
asr_topic: /voice/asr_text
```

这些主题是任务核心依赖：

- `/kinect2/qhd/image_color_rect`：图像帧；
- `/perception/person_detections_2d`：YOLO-World 2D 结果；
- `/kinect2/qhd/points`：Kinect 点云；
- `/scan`：激光雷达；
- `/odom`：里程计；
- `/voice/say`：TTS 语音输出；
- `/voice/asr_text`：ASR 文本。

### 5.3 主人识别与动作识别

```yaml
face_verify_enabled: true
face_verify_required: true
owner_image_path: /home/ubuntu20/catkin_ws/src/wpb_task1_owner_search/data/owner
face_model_name: buffalo_sc
action_recognition_enabled: true
action_core_path: /home/ubuntu20/catkin_ws/src/offline_voice_bridge/scripts/qwen_action_recognition_node.py
action_model_path: /home/ubuntu20/catkin_ws/src/wpr_task1_owner_search/models/pose/yolo11n-pose.pt
```

这里清晰体现：

- 主人验证使用 `InsightFace`，输入路径数组（或单张图）由 `owner_image_path`指向；
- 动作识别核心包括 `PoseActionAnalyzer`、`PointCloudGroundAnalyzer` 与 Qwen 动作 LLM；
- `action_model_path` 指向 YOLO-Pose 模型。

## 6. 主代码结构：`task1_find_owner_real.py`

核心代码文件 `task1_find_owner_real.py` 定义了 `RealOwnerSearchBeforeAction` 类，它是整个任务逻辑的中心。它最主要的结构特征为：

1. 在 `__init__()` 中初始化 ROS 主题和参数对象；
2. 加载主人验证图片、初始化人脸模型；
3. 初始化动作识别器（Qwen 动作核心和 YOLO-Pose/点云分析器）；
4. `run()` 作为任务总流程入口；
5. `run_owner_task_at_waypoint()` 负责单个 waypoint 处理；
6. `scan_for_owner()` / `scan_for_owner_by_time()` / `approach_*` etc. 实现识别与接近、姿态判断、动作处理。

本代码巨大的特征在于：它用一个 Python 类把大量 ROS 节点参数、图像/检测/点云/里程计订阅队列、路径位姿管理、移动平台控制、语音输出、 ASR 输入、等待/修正逻辑、Qwen 动作分析全部集中在同一逻辑体中。

## 7. 主流程：`run()` 与 `run_owner_task_at_waypoint()`

`RealOwnerSearchBeforeAction.run()` 的流程非常清晰：

```python
self.set_yolo_paused(False)
self.rename_exit_waypoint_alias_if_needed()
self.wait_for_hardware_inputs()
self.enroll_owner()
self.init_action_recognizer()
self.ensure_qwen_action_warmup()

# 如果需要就说出开始语句
for waypoint_name in task_waypoint_names:
    found_owner = self.run_owner_task_at_waypoint(waypoint_name)

# 任务完成后返回出口
if return_to_exit_when_complete:
    self.navigate_to_waypoint(exit_waypoint_name)
```

这意味着：

- 第一步确认 YOLO 检测暂停/恢复状态；
- 检查并修正 `waypoint.xml` 中退出点 alias；
- 等待相机、激光、里程计等硬件主题有效；
- 如果允许超级用户/owner enrollment，则启动 ASR 人名输入和 face embeddings 采集；
- 初始化动作识别模型和 Qwen 动作辅助路径；
- 执行按 room 顺序寻找 owner 的任务状态机；
- 完成后导航返回 `exit` 口。

`run_owner_task_at_waypoint()` 本质上是单个地点的完整交互状态机，通常包括：

1. navigate 到 waypoint；
2. 等待/采样检测图像与 YOLO 检测；
3. 通过 YOLO person box 检索候选人；
4. 通过 InsightFace 验证候选是否为主人；
5. 如果成功验证，尝试整体“owner centering”；
6. 调用动作识别（YOLO-Pose -> Qwen/PointCloud）识别动作；
7. 根据动作分类进入 `approach_waving_owner`、`approach_fallen_owner` 或 `wait_for_electrical_switch_instruction()`；
8. 完成后继续下一个 waypoint 的循环。

## 8. 主人验证与候选筛选

架构里最关键一个模块是主人验证的“候选筛选 + 高分人脸匹配”。

### 8.1 人脸模板加载

`load_owner_images()`：

- 如果 `owner_image_path` 是目录，就读取目录中的 `.jpg/.jpeg/.png/.bmp/.webp`；
- 如果是单文件，就直接读取；
- 程序把图片路径与 OpenCV 读出的图像对象形成元组列表。

### 8.2 人脸模型初始化

`init_face_recognizer()`：

- 从 `~/.insightface/models/<model_name>` 查找模型缓存；
- 如果模型没缓存且 `face_auto_download` 没启用，会给出 warning；
- 如果 `InsightFace` 导入成功，则调用 `FaceAnalysis.prepare(...)`；
- 通过 `face_app.get(image)` 抽取人脸 embedding；
- 利用 `owner_reference_images` 与 `owner_face_embedding` 的平均特征得到模板，供后续验证。

### 8.3 候选验证方式

`task1_find_owner_real.py` 由于堆栈参数较复杂，主验证也有“按姿态/姿态-比对”分支：

- 如果YOLO person box 更高更瘦，说明人很直立：优先顶部区块做匹配；
- 如果 box 更宽、更扁，说明人多为躺卧/侧身：做左右侧面 crop、带旋转检查和对比增强；
- 如首次验证失败，使用 CLAHE 对比度增强、1.5x upsampling 等对躺卧侧向 crop 做回退增强。

这套机制把复杂的识别场景编入参数：`face_crop_top_ratio`, `face_crop_lying_side_ratio`, `face_crop_lying_extra_side_ratios`, `face_crop_try_rotations`。

## 9. 动作识别链路

本工程的动作识别不是“单一模型”。它采用了一个三层/三路识别分流模型：

1. YOLO-Pose 动态动作门：
   - `PoseActionAnalyzer`
   - 先看人物关键点动作，如 waving、falling/sudden_fall 过渡动作；
   - 对动态动作，优先做高速识别，避免等待 Qwen 的图片理解。

2. Qwen 实时视觉/LLM 判断：
   - `call_qwen_action(images, prompt)` 发送图像序列到 Ollama 的 `qwen3.5:0.8b` 或别的模型；
   - 通过 `base64` 编码的 JPEG 图像帧向 LLM 请求 JSON 结构动作结果；
   - 输出结构通常为 `{ "action": ..., "place": ... }`。

3. 点云地面/场景支持分析：
   - `PointCloudGroundAnalyzer`：查看点云高度/地面高度/家具高度，区分 lying on ground / lying on bed / sofa / chair 等场景；
   - 对 physical surface semantics 进行分类。

因此，动作识别的主体是：

```text
YOLO-Pose 动态动作门
     -> Qwen LLM 图像动作理解
     -> 点云表面几何支持分析
```

而且整个触发链路通过 `load_qwen_action_core()` 动态加载离线动作识别核心脚本，最大程度复用 `offline_voice_bridge` 约定的动作识别模块。

## 10. 接近逻辑：waving 与 fallen/sitting/lying 的不同路径

主任务脚本的设计非常鲜明：按动作分派不同接近路径。

### 10.1 waving

当识别到主人挥手动作时：

- 使用 `approach_waving_owner()`；
- 候选接近点会根据 `waving_approach_candidate_distances` 与角度集合生成要靠近的位置；
- 通过 `make_plan` 服务检测路径可达性；
- 如果 transform/point cloud/障碍物可见性不足，则在 ROS 的 `move_base` 航点/局部规划器支持下做安全回退。

### 10.2 fallen / lying / sitting：

如果动作被归类为 `sudden_fall`, `fallen`, `falling`, `lying_ground`, `lying`, `sitting`：

- 执行 `approach_fallen_owner()`；
- 先采样点云位置（`fall_approach_position_sample_seconds`），然后利用 `move_base` 的绝对/相对导航；
- 若需要，进一步执行 `perform_fall_assist_arm_motion()` 为跌倒/躺卧人工手臂帮助动作；
- 若 `normalized_action` 属于 `lying`、`sitting`，通常在接近后继续语音等待电器开关指令：`wait_for_electrical_switch_instruction()`。

### 10.3 electrical switch instruction

在主人正常坐着或者处于非危险姿态，接近完成后，工程会转入电气开关指令处理：

- `electrical_switch_instruction_enabled = true`
- `electrical_switch_script_path` 指向：`tools/local_switch_command_test.py`
- 输入由 `direct_asr` 或 `Ollama` 一类模型完成；
- 输出结构状态被写成 `String` 发布到 `/electrical_switch/state`；
- 也可以通过 `aplay` 播放 `ready_ding` 提示音。

这部分构成了任务“找到主人、识别动作、最后交互/执行任务”的后半段闭环。

## 11. `yoloworld_async.py` 与可视化节点

YOLO 这条链路在工程中非常重要。

### 11.1 YOLO 异步消费者

`yoloworld_async.py`：

- 使用 `rospy` 订阅图像主题并回调；
- 以 `process_every_n` 的节流方式降低计算频率；
- 使用 `CUDA`/GPU（例如 `device=cuda:0`）去跑 `yolov8s-world-person.pt`；
- 输出 `perception/person_detections_2d` 检测消息。

### 11.2 调试视图

`yoloworld_debug_viewer_stable.py`：

- 按照最新的图像帧与检测消息渲染带框图像；
- 提供视觉视图避免无调试信号导致任务不清楚要检测到哪一个人；
- 此外通过 `show_yolo_viewer` 与 `yolo_publish_debug` 控制图像显示和调试图发布行为。

这套设置让任务可以在真实相机流上直接“看到视角中的人的裁剪与作用，提高调试效率”。

## 12. 代码的主依赖与 ROS 消息握手

`task1_find_owner_real.py` 是真正的 ROS 任务控制节点，节点初始化时会绑定：

```python
self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback)
self.det_sub = rospy.Subscriber(detections_topic, Detection2DArray, self.detections_callback)
self.points_sub = rospy.Subscriber(points_topic, PointCloud2, self.pointcloud_callback)
self.scan_sub = rospy.Subscriber(scan_topic, LaserScan, self.scan_callback)
self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
self.asr_sub = rospy.Subscriber(asr_topic, String, self.asr_callback)
```

进一步：

- `cmd_vel` 通过 `rospy.Publisher(self.cmd_vel_topic, Twist)` 下发；
- `say_topic` 使用 `rospy.Publisher(self.say_topic, String)` 发布语音文本；
- `electrical_switch_state_pub` 发布自然语言/状态标识；
- `mano_ctrl_pub` 发布机械臂命令；
- `move_base` 的 `SimpleActionClient` 连接导航服务。

从消息结构可以看出，任务节点不是单纯“跟踪视觉目标”，它是 ROS 中连接多个交互通道的综合低层服务：图像、雷达、点云、里程计、导航、语音、ASR、硬件控制等均通过同一主节点编排。

## 13. 典型工程执行链路总结

```text
launch/task1_owner_search_real.launch
    -> task1_find_owner_real.py
        -> load owner images
        -> init face recognizer
        -> init action recognizer
        -> call Qwen/YOLO-Pose/PointCloud analyzer
        -> navigate through waypoints
        -> detect owner by YOLO-World
        -> verify owner face by InsightFace
        -> center owner and sample frames
        -> classify action
        -> branch: waving -> approach
                  fallen/lying/sitting -> fall or seating/lying approach
        -> ask electrical switch command if needed
```

这条链路显示工程不是简单地“搜索一个人”而是有清晰的任务状态机：从多房间候选搜索开始，经历主人验证、动作识别、动作耦合、近距接近、语音交互和电器控制，形成端到端闭环。

## 14. 注意事项与工程使用建议

1. 使用前必须准备：
   - `waypoints.xml`，至少包含 `living_room`, `kitchen`, `bedroom`, `canteen` 和退出点；
   - 真实地图 `/map.yaml` 和 `/map.pgm`；
   - `data/owner` 下的主人正面/侧面/俯视等参考图。

2. 模型建议：
   - `YOLO-World` 人体检测模型应对应 `yolov8s-world-person.pt`；
   - Qwen 动作识别使用 `olist/Ollama` 或本地 `qwen_action_recognition_node.py` 的动作分析器；
   - `InsightFace` 模型必须提前下载或缓存到 `~/.insightface/models`。

3. 运行时：
   - 相机/雷达/里程计/导航主题必须全部活跃；
   - YOLO 常属于高负载节点，默认使用 `cuda:0`，如与动作识别共用显存需注意合并/切换；
   - 若动作识别或电器指令脚本未安装/未加载，任务仍能进入识别，但行动分支会被降级或延后。

## 15. 写在最后

这套工程的核心代码集中在一个大而全的 ROS 节点 `RealOwnerSearchBeforeAction` 中，真正保证运行链路的是：

- `launch` 文件把各种环境、驱动、导航与感知节点串起来；
- `config` 配置请启动时候一次性传入；
- `scripts/task1_find_owner_real.py` 承担法律意义上的“任务状态机”；
- `yoloworld_async.py` / `debug_viewer_stable.py` 负责图像回环可视化与联网检测；
- Qwen 动作识别 core 与点云分析器承担动作分类推断。

因此，阅读此工程时，最重要的不是单看某一个函数，而是理解它在 ROS 中的任务闭环：地图/waypoint 导航 -> YOLO 人体检测 -> 人脸验证 -> 动作理解 -> 接近/语音及后续交互。
