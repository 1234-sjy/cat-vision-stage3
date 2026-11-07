# 看图识猫

一个在本地运行的猫品种识别工具。上传一张照片，页面并排显示传统视觉和视觉大模型的结果：各自给出一个品种，传统视觉标出猫的位置与轮廓，大模型描述外观、姿态和背景。识别后可以继续对这张照片提问。

品种分类支持20类。两条路线读取同一张原图，并行计算。

## 运行环境

- Python 3.12。
- 支持 BF16 的 NVIDIA 显卡，建议16GB显存。
- PyTorch 2.9.1、torchvision 0.24.1，使用 CUDA 12.8 版本。

以下命令都在项目根目录执行。

## Windows 启动

1. 安装 Python 3.12，安装时勾选加入 PATH，并安装适用于显卡的 NVIDIA 驱动。
2. 双击 `安装环境.cmd`，等待依赖安装结束。脚本会在项目目录建立 `.venv` 环境。
3. 在项目目录打开 PowerShell，下载官方模型：

   ```powershell
   .venv/Scripts/python.exe scripts/download_models.py --only all
   ```

4. 双击 `启动项目.cmd`。启动时会先加载模型；终端出现 `Uvicorn running on http://127.0.0.1:8765` 后，在浏览器打开这个地址。

运行期间保留终端窗口，按 `Ctrl+C` 停止服务。

也可以全程使用命令行：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/download_models.py --only all
.venv/Scripts/python.exe -m catvision.server
```

Linux 使用 Python 3.12 创建虚拟环境，将上述命令中的 `.venv/Scripts/python.exe` 换成 `.venv/bin/python`。

## 模型文件

模型统一放在 `weights` 目录。程序加载的文件如下：

| 文件或目录 | 用途 | 获取方式 |
| --- | --- | --- |
| `yolo26n.pt` | 检测猫的位置 | 下载脚本 |
| `sam2.1_t.pt` | 分割猫的轮廓 | 下载脚本 |
| `qwen3-vl-4b/` | Qwen3-VL-4B 基础模型 | 下载脚本 |
| `breed-swin-v3-20-fp16.pt` | 完整画面分类器 | 仓库中提供 |
| `breed-swin-v3-detail20-fp16.pt` | 细节分类器 | 仓库中提供 |
| `qwen-cat-lora-v4/` | 猫品种分类 LoRA 适配器 | 仓库中提供 |

官方模型和项目训练的权重合计约9GB。已有官方模型时，按上表放入对应位置即可。下载脚本会校验文件摘要，运行日志保存在 `artifacts/run-logs`。

## 页面怎么用

1. 选择或拖入一张猫的照片，支持 JPG、PNG、WebP，单张不超过12MB。
2. 点击“开始识别”，查看两条路线的品种结果、标注图、原图描述和耗时。
3. 在“再问这张图”中输入问题，例如“这只猫的耳朵有什么特点？”。多个问题可以用问号分开，一次最多4项。
4. 当前任务结束后，可以换一张照片继续识别。

识别和问答期间，选图与拖放控件会锁定。更换照片时会清空上一张图的结果和问答。原图、标注图、结果和问答保存在 `runtime/results`，每次任务使用单独的目录。

传统视觉结果下的“分类分数”对应这张照片的品种判断；测试集准确率列在下面的评测表中。

## 代码入口

| 位置 | 内容 |
| --- | --- |
| `catvision/server.py` | FastAPI 服务入口，组织图片任务和问答 |
| `web/index.html` | 上传、结果展示与图片问答页面 |
| `catvision/traditional.py` | YOLO 检测、SAM 分割与 Swin 分类 |
| `catvision/vlm.py` | 视觉大模型调用 |
| `catvision/vlm_features.py` | 原图描述与问答提示 |
| `configs/active-classifier.json` | 分类器权重和组合配置 |
| `configs/active-vlm.json` | Qwen 适配器与推理配置 |
| `scripts/` | 模型下载、数据准备、训练与评测脚本 |

启动服务后，接口文档位于 `http://127.0.0.1:8765/docs`。图片识别用 `POST /api/recognize`，通过返回的任务ID调用 `GET /api/jobs/{id}` 查询结果；图片问答用 `POST /api/jobs/{id}/questions`。

## 识别方法与评测

传统视觉使用 YOLO26n 定位、SAM2.1 分割，以及两个 Swin-Tiny 分类器。完整画面模型采用320像素等比例补边，细节模型采用224像素中心视图，两者分数按75%和25%组合，并平均原图与水平翻转后的预测。

视觉大模型使用 Qwen3-VL-4B-Instruct，以 LoRA 微调品种分类。描述与问答使用同一基础模型，并在生成时暂时关闭品种分类适配器。模型在启动时加载，后续请求复用显存中的实例。

同一组1514张测试照片覆盖20个猫品种，分类结果如下：

| 方法 | 正确数量 | 准确率 |
| --- | --- | --- |
| 传统分类器 | 1292 / 1514 | 85.34% |
| Qwen 品种分类 | 1317 / 1514 | 86.99% |

这组 Qwen 评测采用 batch=4。逐图输出、各品种统计和训练记录保存在 `artifacts/optimization-v4`；原始1872张测试集上的传统分类器结果为1600张正确，准确率85.47%。

## 模型与数据来源

检测与分割使用 [Ultralytics](https://github.com/ultralytics/ultralytics) 和 [SAM 2](https://github.com/facebookresearch/sam2)，分类器使用 [torchvision Swin Transformer](https://pytorch.org/vision/stable/models/swin_transformer.html)，视觉大模型为 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)。

训练图片来自 [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) 和 [CatBreedsRefined-7k 第2版](https://www.kaggle.com/datasets/doctrinek/catbreedsrefined-7k)。预训练模型和数据集按各自发布页所列许可使用。
