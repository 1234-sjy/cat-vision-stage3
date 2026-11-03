# 看图识猫 第二阶段

上传一张照片，同时查看传统视觉与本地视觉大模型的结果。每条路线只输出一个品种，识别页只展示本次图片的结果。

## 当前方法

传统路线用YOLO26n检测、SAM2.1分割，Swin-Tiny识别20个猫品种。完整画面模型采用320像素等比例补边，细节模型采用224像素中心视图；验证集选定完整画面75%、细节25%，分别平均原图和水平翻转后的分数。猫狗判断同样融合。检测漏掉时，原图分类继续运行。

另一条路线使用本地Qwen3-VL-4B-Instruct，独立读取相同原图与固定任务说明。它不读取传统路线的品种或分数。提示与两行输出格式在验证集选择，参数见`configs/active-vlm.json`。图像切块投影改成与原卷积等价的矩阵运算，官方权重和图片像素预算保持原设置。

普通启动即预加载、常驻模型；两条路线使用独立线程与各自复用的CUDA流。实际推理没有人为暂停。不同上传请求排队，避免重复加载大模型。每次重新读图推理，未缓存识别答案。

## 实际测试

| 范围 | 传统分类器或路线 | 视觉大模型 |
| --- | --- | --- |
| 20类独立测试1872张，原图分类 | 1600/1872（85.47%） | 未在全量1872张运行 |
| 其中Oxford原12类子集1177张 | 1042/1177（88.53%） | — |
| 其中扩充来源695张，覆盖20类 | 558/695（80.29%） | — |
| 同一批120张，每类6张，完整流程 | 97/120（80.83%） | 67/120（55.83%） |

1872张是自然保留的测试分布，120张是每类均取6张的对照分布。两份清单在测试前固定，训练和选模型只用训练/验证图片。原始逐图结果、失败状态与模型身份见`artifacts/optimization-v3/frozen-test`和`shared-test`。各测试范围分别记录。

120次连续识别的总耗时中位数为0.91秒，CUDA保留显存最高约10.46GiB。记录期间供电曾切到电池，出现低频时延；逐图耗时全部保留。代码取消了推理休息，演示视频保留真实等待。

20类包括原有12类，以及美国短尾猫、美国卷耳猫、美国短毛猫、异国短毛猫、曼岛猫、挪威森林猫、苏格兰折耳猫、土耳其安哥拉猫。品种表见`artifacts/optimization-v3/data/classes.json`。猫狗头用狗负例训练，品种头的输出范围为这20类；Qwen可输出other。

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

两份自训练权重`weights/breed-swin-v3-20-fp16.pt`与`breed-swin-v3-detail20-fp16.pt`各约55MB，随代码入库。FP16用于磁盘存储，Swin以FP32执行。官方YOLO/SAM/Qwen从完整保留包恢复，或按上面的脚本获取并校验摘要。运行网页无需训练集，也无需重新训练。

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
