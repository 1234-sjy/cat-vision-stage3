# 看图识猫 · 第二阶段改进版

上传同一张猫图，两条路线独立判断，各返回一个品种。

- 传统视觉：YOLO26n与SAM2.1标出猫的位置、轮廓；独立微调的Swin-Tiny从完整原图识别12种猫，并用单独训练的猫/狗头处理狗负例。检测漏掉时仍运行原图分类。
- 视觉大模型：本地Qwen3-VL-4B-Instruct通过Transformers SDK读取相同原图及固定任务说明，给出单品种与简短特征。两路不传递答案。

验证集比较原图、裁剪及融合后，原图与25%主体融合并列最好；最终选计算更少的原图方案。SAM分割真实运行并用于轮廓展示，当前分类融合权重为0，没有把它写成准确率提升的来源。

## 已测结果

| 同题范围 | 第一阶段保存结果 | 第二阶段改进版 |
|---|---:|---:|
| 全部1183张猫，原图分类器 | 1051/1183，88.84% | 1067/1183，90.19% |
| 旧共同120张，流程回归 | 97/120，80.83% | 107/120，89.17% |
| 新共同120张，独立测试 | 103/120，85.83% | 106/120，88.33% |

旧第二阶段在旧共同120张上为88/120（73.33%），其中15张检测漏掉而没有运行分类器。这个硬门槛已移除。新共同120张从旧120之外提前固定，种子20261102，每类10张；只在验证集选模、选阈值和融合方式，测试失败都计入分母。全量1183张包含之前排查过的旧120张；全量原图成绩与实际带定位/分割的120张流程成绩分别记录。100张独立官方测试狗图全部被物种头挡住；狗负例验证不等于所有物体的开放类别测试。

两路线在新共同120张上的完整对照、原始输出和耗时见`artifacts/evaluation/comparison-v2`及`artifacts/comparison-v2`。第一阶段对照只读取之前保存的预测，没有载入其训练权重。新版权重与第一阶段独立训练。

[项目介绍与个人感悟 Word文档](docs/项目介绍与个人感悟.docx) · [课程提要](docs/课程依据/第二阶段指定课程实读提要.txt) · [改进记录](docs/改进说明.txt)

## 环境与启动

Python3.12、PyTorch2.9.1、torchvision0.24.1，开发使用CUDA12.8版PyTorch与RTX5080Laptop16GB。完整双路线建议NVIDIA显卡及16GB显存。路径从项目根目录确定，外部数据由参数传入。

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m catvision.server
```

也可双击“安装环境.cmd”，再双击“启动项目.cmd”。已有环境通过`CATVISION_PYTHON`或`scripts/start.ps1 -PythonExe`指定解释器。浏览器打开 http://127.0.0.1:8765 。

自训练权重`weights/breed-swin-v2-fp16.pt`约55MB，随源码入库；`configs/active-classifier.json`指定当前权重、阈值与融合方式。FP16仅用于存储，模型以FP32执行。旧`breed-best.pt`约11MB保留为历史。较大的官方检测、分割和Qwen权重从完整保留包恢复；只有源码时可单独执行下载：

```powershell
.venv/Scripts/python.exe scripts/download_models.py --only all
```

不要同时执行下载和模型计算。原始pip freeze保留在`artifacts/environment`；`requirements-lock.txt`已去掉私人wheel路径，记录可恢复的包版本和官方CUDA来源。其他显卡应选择兼容的PyTorch/CUDA组合。

## 数据与复现

猫950张训练、238张验证、1183张官方测试。新增狗负例1000张训练、500张验证，均来自官方trainval；官方test不进入训练。猫训练随机使用原图或已保存主体视图，不把视图当成新的独立来源。清单与SHA256在`artifacts/data`和`artifacts/optimization-v2`。

从头复现本轮训练需要另取通用ImageNet初始化：先单独运行`python scripts/download_swin.py`，完成后再运行`python scripts/train_transformer.py --data-root <Oxford原始目录>`。训练输出目录已有历史时脚本会停止，不覆盖、不自动续训。正式结果已有权重，运行网页不用重新训练。

本轮训练batch8、workers0、CPU线程2，每批同步后等待0.05秒；独立验证最多batch2、每批等待0.2秒；网页单图推理。训练23轮，最佳15轮，只用验证结果选检查点。GPU入口使用`ModelTask`锁，失败时保留完整日志并停止，不自动重试或重启。`configs/frozen-v2.json`在测试前记录模型、代码与新共同清单身份；评估缓存核对实际输入字节、权重和配置。

## 保留资料

旧版测试、权重、日志、完整交付包均保留。新完整包通过`scripts/package_project.py`按来源目录收集，包含Git忽略的数据、所有每轮权重、下载分段和历史记录；不使用git archive作为完整保留包。两阶段PDF、课程要求提要、68集视频、原始Oxford数据及全部第二阶段模型资料继续保留。

## 参考与许可

方法参考Grounded-SAM-2的框提示分割、Pytorch-garbage-classify的训练/验证组织和LAVIS的图文SDK调用。本项目独立实现猫品种任务，AI辅助开发，指标来自本机实际运行。

- https://github.com/IDEA-Research/Grounded-SAM-2
- https://github.com/Candy-CY/Pytorch-garbage-classify
- https://github.com/salesforce/LAVIS
- https://pytorch.org/vision/stable/models/swin_transformer.html
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- Oxford数据与许可：https://www.robots.ox.ac.uk/~vgg/data/pets/

Ultralytics、SAM、Qwen与Swin遵循各自上游许可。

## GitHub提交与开发历史

开发中的真实Git提交随源码一并保留。交接时通过Git bundle恢复仓库，再将原有提交整体推送到GitHub；不要只用网页上传最终文件，也不要重新git init、压成单个提交或改写提交日期。代码、requirements.txt、requirements-lock.txt、安装启动脚本和运行说明都在同一仓库。

本地提交记录与GitHub推送时间分别保留；仓库恢复方式见docs/提交与复现要求.md。

## 快速识别模式

双击“快速识别.cmd”，或运行`python -m catvision.server --fast-demo`。启动时先串行加载两路模型，准备好后再选图识别。模型常驻约10GB显存，推理取消人为暂停，CPU使用8线程；两条路线仍依次计算，每次重新读取图片推理，不复用识别答案。训练参数与原有测试成绩不变。

普通启动方式继续在每次请求后释放模型。界面同时标注原图分类器1183张测试的90.19%与两路共同120张的88.33%和72.50%，三个数字对应的范围不同。
