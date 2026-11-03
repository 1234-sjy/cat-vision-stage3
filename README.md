# 看图识猫 第三阶段

在第二阶段双模型基础上，新增登录、图片任务队列和数据库历史，以Linux+Docker运行完整Web应用。网页用Streamlit并排展示同一张照片的两条路线，支持原图观察和继续问答。

第三阶段入口由nginx统一转发，CPU Web进程与常驻GPU工作端分开。部署和依赖见[第三阶段运行说明](docs/第三阶段运行说明.md)，配置在`compose.yaml`、`Dockerfile.web`、`Dockerfile.worker`和`.env.example`。本机可用WSL2部署，使用已有显卡，不需要购买云服务器或模型API。

```bash
python3 scripts/setup_stage3.py
bash scripts/start_stage3.sh --build
```

打开`http://127.0.0.1:8080`，在网页注册应用账号。后续启动用`bash scripts/start_stage3.sh`；已配置好CatVision WSL环境的Windows电脑可以双击“启动项目.cmd”。旧的第二阶段单机入口保留为“启动第二阶段本地版.cmd”。

下面保留第二阶段模型方法、数据与成绩，作为第三阶段复用的模型依据。第三阶段未重新训练模型。

## 第二阶段模型与实验记录

上传一张照片，同时查看传统视觉与本地视觉大模型的结果。每条路线各输出一个品种；大模型另外说明可见外观、姿态和背景，支持继续对这张图提问。识别页只展示本次图片的结果。

## 当前方法

传统路线用YOLO26n检测、SAM2.1分割，Swin-Tiny识别20个猫品种。完整画面模型采用320像素等比例补边，细节模型采用224像素中心视图；验证集选定完整画面75%、细节25%，分别平均原图和水平翻转后的分数。猫狗判断同样融合。检测漏掉时，原图分类继续运行。

另一条路线使用本地Qwen3-VL-4B-Instruct，冻结视觉编码器和通用基础参数，用训练集的原始品种标签微调文本层LoRA。它独立读取相同原图，分类阶段返回一个品种代码，不读取传统路线的品种或分数。rank8、alpha16，768×768推理像素预算，分类最多24个输出token，BF16 AMP。图像切块投影使用与原卷积等价的矩阵运算。适配器及提示由固定验证集选定，配置见`configs/active-vlm.json`。

普通启动即预加载、常驻模型；两条路线使用独立线程与各自复用的CUDA流。实际推理没有人为暂停。不同上传请求排队，避免重复加载大模型。每次重新读图推理，未缓存识别答案。

## 实际测试

| 范围 | 传统分类器或路线 | 视觉大模型 |
| --- | --- | --- |
| 20类独立测试1872张 | 1600/1872（85.47%） | 此范围未全量运行 |
| 同1514张，Qwen另留测试 | 1292/1514（85.34%） | 1317/1514（86.99%） |
| 原120张批量分类记录，每类6张 | 97/120（80.83%） | 98/120（81.67%） |
| 原120张当前逐图调用，开启描述 | 本轮未重跑A | 100/120（83.33%） |

1514张由原测试集排除历史本地Qwen推理用过的图片及身份关联组后固定，覆盖20类。Qwen按batch4评测；传统列从同一冻结分类器的原1872张逐图记录取对应图片，SHA256逐图核对。原120张用于回归，改进前Qwen为67/120（55.83%）。训练与选模型只用训练/验证集，各组分母分别记录。

本次实机演示的Qwen耗时包含分类与图像描述：传统0.41秒，Qwen4.82秒，两条路线并行总耗时4.82秒。模型在启动时预加载，视频保留真实选图与等待，并演示图片问答（本次1.00秒）。原始测试输出和各品种成绩见`artifacts/optimization-v4`。

## 原图观察与问答

对照PDF的“图像观察与问答”和LAVIS参考示例，B路线保留品种判断，并对同一原图生成外观、姿态和背景描述。描述与问答临时关闭只学品种代码的LoRA，调用同一Qwen底座；退出上下文后恢复分类适配器，不另加载一份大模型。模型不读取A的答案、检测框、分割图或文件名，也不按品种名生成固定特征。

描述最多384个输出token，问答最多256个；配置见`configs/feature-description.json`，提示见`catvision/vlm_features.py`。这些上限与分类的24个token分开。失败和不完整解析保留原始返回，页面显示实际文本，不补造特征。

识别完成后，在“再问这张图”输入问题。`POST /api/jobs/{图片任务ID}/questions`接收`{"question":"问题"}`，返回问答任务ID；用`GET /api/jobs/{任务ID}`取进度与回答。多个问号分开逐项回答，一次最多4项；逐项返回和原文一起保存。问答重读该图片任务保存的原图，使用已有GPU请求队列，避免与另一请求的适配器切换交叉运行。单次识别内A/B仍并行。

图片理解的逐例图像、原始文字和验证记录见`artifacts/image-understanding-v3`。固定120张回归在描述开启时重新逐张运行；它验证分类有无受影响，不是描述准确率评测。1514张的86.99%是品种分类成绩，不能换算成每句描述或回答的正确率。参考阅读记录见`docs/课程依据/LAVIS参考阅读`，没有安装或提交整套参考项目作为自己的实现。

## 环境与启动

Python3.12、torch2.9.1、torchvision0.24.1；开发机为RTX5080Laptop16GB，CUDA12.8。完整双路线使用NVIDIA显卡，建议16GB显存。

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/download_models.py --only all
.venv/Scripts/python.exe -m catvision.server
```

浏览器打开`http://127.0.0.1:8765`。也可双击“安装环境.cmd”“启动项目.cmd”。已有解释器由`CATVISION_PYTHON`或`scripts/start.ps1 -PythonExe`指定。训练、下载和驻留服务由GPU租约隔离；训练前关闭网页服务。失败保留日志并停止，没有自动重试或重启。

两份自训练权重`weights/breed-swin-v3-20-fp16.pt`与`breed-swin-v3-detail20-fp16.pt`各约55MB，随代码入库。FP16用于磁盘存储，Swin以FP32执行。官方YOLO/SAM/Qwen从完整保留包恢复，或按上面的脚本获取并校验摘要。运行网页无需训练集，也无需重新训练。Qwen适配器`weights/qwen-cat-lora-v4/adapter_model.safetensors`约66MB，也随Git入库；官方4B底座仍按下载脚本获取或从完整保留包恢复。

## 数据与训练

来源为Oxford-IIIT Pet和CatBreedsRefined-7k第2版。筛选后猫训练4157张、验证925张、测试1872张；狗训练999张、验证500张。原始文件保留，清单中的排除项没有删除。原950张训练图已经扩充，裁剪或翻转视图不另算独立样本。

清单记录SHA256、pHash、可恢复的原动物ID和摄影会话关联。已知同源组放在同一划分，冻结摘要见`data/freeze.json`。扩充来源重新命名了图片，部分图片缺少标签品种的典型特征；实际标签按来源保留，审计观察见`data/summary.json`。

正式20类模型从通用ImageNet初始化；早期继承旧分类器的尝试在来源审查后停止，其权重和日志保留。先训练20轮，再比较输入方式，以完整画面320像素、同物种MixUp及细节补充继续优化。最终按验证结果选权重和组合比例，选定后才运行测试。

加速训练先实测批量32、64、128、192的显存需求与吞吐量，再选择已验证的最快批量。224像素阶段采用batch128，接电实测约510训练图/秒，训练加验证约12.7秒/轮；320像素阶段采用batch64，约240训练图/秒。CPU8线程、4个数据加载进程、BF16训练、无固定等待。每秒GPU占用、显存、温度和功耗见各训练目录的`gpu-telemetry.jsonl`。

复现训练可恢复完整保留包中的`原始数据/cat-data`，直接使用仓库冻结清单，通过`--data-root`传入实际位置。自行获取扩充数据时使用`prepare_expanded.py --download --out-dir runtime/data-acquisition`，再用`--extract-only`恢复路径；Oxford按其官方地址获取。重新训练指定新的`--output`和`--checkpoint`，保留已有历史。

```powershell
python scripts/download_swin.py
python scripts/train_expanded.py --data-root <cat-data目录> --epochs 20 --output artifacts/reproduction/initial --checkpoint weights/reproduction.pt
```

后续阶段使用`scripts/train_accelerated.py`，源检查点轮次由`--start-epoch`指定，完整画面选项为`--image-size 320 --preprocess letterbox --mixup 0.2`。各阶段参数、来源摘要、选模记录和不可覆盖的每轮检查点均保留在本地完整保留包。

### Qwen轻量微调

训练集为4157张猫和999张狗。只监督assistant的品种代码与结束符，题目、图像和补齐位置不计算训练损失。固定选择验证集为200张猫与50张狗；最佳为第2轮。LoRA更新16,515,072个参数，学习率0.0001，有效batch16，microbatch4，BF16、梯度检查点和余弦学习率。原始配置、断点、恢复记录及验证输出全部保留。

```powershell
python scripts/train_vlm_lora.py --data-root <cat-data目录> --output artifacts/reproduction/vlm --adapter-output weights/reproduction-lora
```

需要接续时显式传`--resume-from <检查点目录>`并指定新的输出目录，恢复优化器、调度器、随机状态和抽样位置，不自动重试或重启。最新用户要求下，默认不记录显卡状态。独立测试在选模完成后运行；不使用传统分类器的预测当训练标签。

## 说明与交付

[项目介绍与个人感悟](docs/项目介绍与个人感悟.docx) · [完整课程提要](docs/课程依据/第二阶段指定课程实读提要.txt) · [提交与复现要求](docs/提交与复现要求.md)。旧12类结果放在历史记录中。

上传包使用Git bundle恢复全部真实提交后推送。代码、依赖、启动和模型获取说明统一入库；不重新git init、不压成单个提交、不倒填日期。本地开发时间与GitHub收到推送的时间分别保留。大型原始数据、全部模型与训练资料通过完整保留包保存，打包不按gitignore或git archive过滤。

参考：
- https://github.com/IDEA-Research/Grounded-SAM-2
- https://github.com/Candy-CY/Pytorch-garbage-classify
- https://github.com/salesforce/LAVIS
- https://pytorch.org/vision/stable/models/swin_transformer.html
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- https://www.robots.ox.ac.uk/~vgg/data/pets/
- https://www.kaggle.com/datasets/doctrinek/catbreedsrefined-7k

Oxford与扩充图片按各自数据源许可使用；模型与代码遵循相应上游许可。开发和排查使用AI辅助，提交材料保留真实实验过程。
