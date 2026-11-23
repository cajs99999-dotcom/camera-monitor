# 教室摄像头监控

Windows 上的六路教室监控，包含 WebRTC 实时预览、RF-DETR-L 人体检测、RTMPose-L 17 点骨架和带六个机位预设的教室 3D 查看器。

## 获取项目文件

克隆时安装 [Git LFS](https://git-lfs.com/)，然后在仓库根目录执行 `git lfs pull`。教室版检测权重、RTMPose-L ONNX 和 TensorRT 引擎、教室 GLB 以及 MediaMTX Windows 可执行文件均通过 Git LFS 保存。

## 在 Windows 上启动

准备 Python 3.11 环境 `.venv-pose`，并确保 FFmpeg 在 `PATH` 中。现用的模型环境记录见下文。

```powershell
Copy-Item .\mediamtx\mediamtx.example.yml .\mediamtx\mediamtx.yml
.\.venv-pose\Scripts\python.exe -B .\main.py --check-app
```

在本地的 `mediamtx/mediamtx.yml` 中，将 `cam1` 到 `cam6` 的 `source: publisher` 改为各自的 RTSP 地址。也可以用 `MTX_PATHS_CAM1_SOURCE` 到 `MTX_PATHS_CAM6_SOURCE` 向中继进程传入地址。这个本地配置已在 Git 中忽略。

从仓库根目录分别在两个 PowerShell 窗口启动。

```powershell
.\mediamtx\mediamtx.exe .\mediamtx\mediamtx.yml
```

```powershell
.\.venv-pose\Scripts\python.exe -B -u .\main.py
```

打开 [监控首页](http://127.0.0.1:8767/web/index.html) 或 [教室 3D](http://127.0.0.1:8767/web/classroom/index.html)。`--check-app` 校验入口、发布页面和推理流水线文件，执行时无需摄像头或模型初始化。

如需让同一局域网的其他设备访问，在启动后端前将 `CAMERA_MONITOR_HOSTS` 设为实际访问的主机名或 IPv4 地址（多个值以逗号分隔），启动时增加 `--host 0.0.0.0`。在本地 MediaMTX 配置中同步调整 `webrtcAddress` 和 `webrtcLocalUDPAddress` 的绑定，为 `webrtcAdditionalHosts` 补充监控主机的局域网地址，并将实际网页来源（含协议和 8767 端口）加入 `webrtcAllowOrigins`。

## 已验证的 GPU 环境

原部署使用 Windows x64、Python 3.11.17、单张 NVIDIA GeForce RTX 4060 Laptop GPU、驱动 617.42、PyTorch 2.14.1+cu130、torchvision 0.29.1+cu130、RF-DETR 1.12.1、RTMLib 0.0.16、OpenCV 4.11.0.86、aiohttp 3.14.5、ONNX Runtime GPU 1.32.0、TensorRT 11.4.0.106 和 CUDA 12.6 DLL。MediaMTX 为 1.21.3。本仓库保留模型和验收文件，Python 与 NVIDIA SDK 需要本机安装。

当前 TensorRT 引擎及验收文件校验 GPU UUID、驱动和 DLL 指纹。迁移到其他设备需要重新构建并验证引擎，同时更新对应的验收和哈希。SDK 和 CUDA DLL 路径定义在 `ai/pose_tensorrt.py` 与 `ai/tensorrt_runtime.py`。

## 3D 查看器开发

查看器的发布文件在 `web/classroom`。在 `classroom` 目录运行 `npm ci` 和 `npm run build` 即可重新生成页面。详细说明见 [classroom/README.md](classroom/README.md)。

MediaMTX 和 Three.js 的第三方许可证分别位于 [mediamtx/LICENSE](mediamtx/LICENSE) 和 [web/classroom/THREE_LICENSE.txt](web/classroom/THREE_LICENSE.txt)。
