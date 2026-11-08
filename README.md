# 看图识猫

一个猫品种识别网页。上传一张照片，页面并排显示传统视觉和视觉大模型的结果：各自给出一个品种，传统视觉标出猫的位置与轮廓，大模型描述外观、姿态和背景。识别后可以继续对这张照片提问。

品种分类支持20类。两条路线读取同一张原图，并行计算。网页支持注册、登录、查看历史照片和修改照片名称，用 Docker 在 Linux 上运行。

## 运行环境

- Ubuntu 22.04 / 24.04，或 Windows 上的 WSL2 Ubuntu。
- 支持 BF16 的 NVIDIA 显卡，建议16GB显存，并安装适用的 NVIDIA 驱动。
- Docker Engine、支持 GPU 配置的 Docker Compose 插件、NVIDIA Container Toolkit。
- 容器内使用 Python 3.12、PyTorch 2.9.1、torchvision 0.24.1 和 CUDA 12.8。

以下命令都在项目根目录执行。启动前先关闭占用同一套模型的本地服务或训练进程。

## Linux 启动

1. 安装 Docker 和 NVIDIA 容器运行时。已有这些组件时可以跳过：

   ```bash
   sudo bash scripts/install_linux_host.sh
   ```

   如果网络无法访问 Docker 官方软件源，可使用 Ubuntu 软件源：

   ```bash
   sudo env CATVISION_DOCKER_SOURCE=ubuntu bash scripts/install_linux_host.sh
   ```

2. 准备模型文件。自训练分类器和 LoRA 适配器随仓库提供，官方模型用下载脚本获取。下载脚本需要 Python、Pillow 和 curl，可以单独建立一个虚拟环境：

   ```bash
   sudo apt-get install -y python3-venv curl
   python3 -m venv .venv
   .venv/bin/python -m pip install 'Pillow>=11,<13'
   .venv/bin/python scripts/download_models.py --only all
   ```

   已有官方模型时，按下面的“模型文件”放入对应位置即可。

3. 初始化配置，构建镜像并启动：

   ```bash
   python3 scripts/setup_stage3.py
   bash scripts/start_stage3.sh --build
   docker compose logs -f worker
   ```

   等日志显示模型已就绪，在浏览器打开 `http://127.0.0.1:8080/`。退出日志查看可按 `Ctrl+C`，容器会继续在后台运行。

后续启动已有镜像：

```bash
bash scripts/start_stage3.sh
```

`.env` 由初始化脚本生成，保存数据库、队列和 JWT 配置，不上传到仓库。已有 `.env` 时脚本保留原配置，示例见 `.env.example`。

模型放在其他目录时，修改 `.env` 中的 `CATVISION_MODEL_ROOT`。服务器需要从外部访问时，将 `BIND_ADDRESS` 改为 `0.0.0.0`，并在云平台放行 `WEB_PORT` 对应的 TCP 端口，默认8080。数据库和消息队列通过容器内部网络访问。

## Windows 启动

先准备 WSL2 Ubuntu，并在 Ubuntu 内完成上面的 Docker 安装、模型下载和首次构建。Windows 使用已有的显卡驱动，不在 WSL 内另装 Linux 显卡内核驱动。

默认 WSL 发行版名称为 `CatVision`，在项目根目录打开 PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_stage3.ps1 -Build
```

以后可双击 `启动项目.cmd`。发行版名称不同时，用 `wsl --list --verbose` 查看名称，再通过 `-Distribution` 指定：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_stage3.ps1 -Distribution Ubuntu -Build
```

停止服务：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stop_stage3.ps1
```

自定义发行版名称时，停止命令也需要加相同的 `-Distribution` 参数。

## 模型文件

模型默认放在 `weights` 目录。程序加载的文件如下：

| 文件或目录 | 用途 | 获取方式 |
| --- | --- | --- |
| `yolo26n.pt` | 检测猫的位置 | 下载脚本 |
| `sam2.1_t.pt` | 分割猫的轮廓 | 下载脚本 |
| `qwen3-vl-4b/` | Qwen3-VL-4B 基础模型 | 下载脚本 |
| `breed-swin-v3-20-fp16.pt` | 完整画面分类器 | 仓库中提供 |
| `breed-swin-v3-detail20-fp16.pt` | 细节分类器 | 仓库中提供 |
| `qwen-cat-lora-v4/` | 猫品种分类 LoRA 适配器 | 仓库中提供 |

官方模型和项目训练的权重合计约9GB，下载分片和 Docker 镜像还会占用额外空间。下载脚本会校验文件摘要，日志保存在 `artifacts/run-logs`。运行网页不需要下载训练图片，也不需要重新训练模型。

## 页面怎么用

1. 第一次打开网页，点击“注册”创建网站账号，再登录。这个账号与云平台、服务器系统账号分开。
2. 在“共同输入”中选择猫的照片，支持 JPG、PNG、WebP，单张不超过12MB。
3. 点击“开始识别”，查看两条路线的品种结果、标注图、原图描述和耗时。
4. 在“关于这张图片的问题”中输入问题，再点击“提问”，例如“猫的耳朵有什么特点？”。
5. 从“我的历史照片”打开旧记录。修改“照片名称”后，点击“保存名称”。

识别和问答期间，选图、拖放与历史切换会锁定。更换照片时会清空上一张图的结果和问答；打开历史记录时显示该记录的原图和结果。每次任务使用单独的任务ID，问答与对应的图片任务绑定。

## 代码入口

| 位置 | 内容 |
| --- | --- |
| `catvision/stage3_ui.py` | Streamlit 页面，组织上传、结果、问答和历史操作 |
| `catvision/stage3_api.py` | FastAPI 服务入口，处理认证和图片任务 |
| `catvision/stage3_auth.py` | 密码哈希和 JWT 校验 |
| `catvision/stage3_store.py` | 用户、任务和结果的数据库读写 |
| `catvision/stage3_queue.py` | RabbitMQ 任务投递 |
| `catvision/stage3_worker.py` | 加载模型、消费任务、保存结果 |
| `catvision/traditional.py` | YOLO 检测、SAM 分割与 Swin 分类 |
| `catvision/vlm.py` | 视觉大模型调用 |
| `configs/active-classifier.json` | 分类器权重和组合配置 |
| `configs/active-vlm.json` | Qwen 适配器与推理配置 |
| `compose.yaml` | 服务、端口、环境变量和数据挂载 |
| `deploy/nginx.conf` | 网页与 API 的反向代理 |
| `scripts/` | 环境安装、启动、模型下载、训练与评测脚本 |

启动后，接口文档位于 `http://127.0.0.1:8080/docs`。注册、登录之外的图片和任务接口需要 Bearer JWT，用户只能访问自己的记录。

## 日志与数据

网页和 API 在 CPU 容器中运行，一个常驻 GPU 工作端加载模型。同一张照片的两条识别路线并行计算，不同用户提交的任务通过队列排队。

用户和任务记录保存在 PostgreSQL 命名卷中，消息队列使用独立命名卷。图片、结果和模型日志保存在 `runtime/stage3`，普通停止或重建镜像会保留这些数据。

```bash
docker compose ps
docker compose logs --tail=100 api worker
docker compose stop
```

`docker compose down -v` 会删除数据库和队列卷，不用于日常停止。

模型工作端发生错误时会保存日志并停止。查看 `docker compose logs worker` 和 `runtime/stage3/model-logs`，处理错误后再显式恢复：

```bash
docker compose run --rm worker python scripts/recover_stage3_worker.py --acknowledge
docker compose up -d worker
```

## 识别方法与评测

传统视觉使用 YOLO26n 定位、SAM2.1 分割，以及两个 Swin-Tiny 分类器。完整画面模型采用320像素等比例补边，细节模型采用224像素中心视图，两者分数按75%和25%组合，并平均原图与水平翻转后的预测。

视觉大模型使用 Qwen3-VL-4B-Instruct，以 LoRA 微调品种分类。描述与问答使用同一基础模型，并在生成时暂时关闭品种分类适配器。模型在启动时加载，后续请求复用显存中的实例。

同一组1514张测试照片覆盖20个猫品种，分类结果如下：

| 方法 | 正确数量 | 准确率 |
| --- | --- | --- |
| 传统分类器 | 1292 / 1514 | 85.34% |
| Qwen 品种分类 | 1317 / 1514 | 86.99% |

这组 Qwen 评测采用 batch=4。逐图输出、各品种统计和训练记录保存在 `artifacts/optimization-v4`；原始1872张测试集上的传统分类器结果为1600张正确，准确率85.47%。网页的耗时对应当前图片，分类分数对应单张照片的品种判断。

## 模型与数据来源

检测与分割使用 [Ultralytics](https://github.com/ultralytics/ultralytics) 和 [SAM 2](https://github.com/facebookresearch/sam2)，分类器使用 [torchvision Swin Transformer](https://pytorch.org/vision/stable/models/swin_transformer.html)，视觉大模型为 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)。

训练图片来自 [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) 和 [CatBreedsRefined-7k 第2版](https://www.kaggle.com/datasets/doctrinek/catbreedsrefined-7k)。预训练模型和数据集按各自发布页所列许可使用。
