# 看图识猫 · 第二阶段

本项目让两种方法识别同一张猫图，各给出一个品种：

- 传统视觉：YOLO检测主要猫，SAM2根据检测框分割主体，再由独立微调的分类器识别品种。
- 视觉大模型：本地Qwen3-VL-4B读取原图与固定任务提示，给出一个品种和简短外观特征。

分类器训练、两条路线的实际预测、本地界面、项目文档和运行视频均已完成。Qwen、YOLO和SAM的文件已通过官方SHA256核对。共同测试在预测前固定为每个品种10张，共120张；失败也计入总数。

| 方法 | 正确图片 | Top-1 | 宏平均F1 |
|---|---:|---:|---:|
| YOLO检测 + SAM分割 + 品种分类 | 88/120 | 73.33% | 0.7782 |
| 本地Qwen3-VL-4B | 80/120 | 66.67% | 0.6428 |

传统路线有15张未检测到猫；Qwen有5次格式错误、2次返回其他品种。逐图结果位于`artifacts/evaluation/comparison`，对比表与图表位于`artifacts/comparison`。训练的最佳验证Top-1约87.0%，与上表的独立测试成绩分开记录。

[项目说明PDF](docs/项目说明与实现感悟.pdf) · [运行与提交步骤](docs/运行与提交步骤.txt) · [完整课程提要](docs/课程依据/第二阶段指定课程实读提要.txt)

## 环境

Python 3.12，CUDA版PyTorch 2.9.1和torchvision 0.24.1。开发使用CUDA 12.8版PyTorch，显卡为RTX 5080 Laptop 16GB。新电脑应先按自己的显卡安装兼容的CUDA版PyTorch，再在独立环境中安装项目依赖。

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

代码从自身位置确定项目根目录；外部数据通过`--data-root`传入，不依赖开发机器的用户名或目录。

## 当前运行约定

下载与GPU工作分开执行，一次一个模型任务。检测验证batch=2、workers=0，每批同步GPU后等待0.2秒；单张图片推理batch=1。下载为顺序请求，失败后不自动重试。

正式训练已获用户授权并完成。训练batch=8、验证batch上限8、workers=0、CPU线程2，每批同步GPU后等待0.05秒。不要使用旧计划中的batch64配置。发生失败或异常中断后，先阅读`runtime/model-task-state.json`指向的完整日志，再人工决定如何处理；程序不会自行清除失败状态继续运行。

```powershell
python scripts/download_models.py --only vlm
python scripts/validate_detector.py --data-root <原始Oxford数据目录>
```

以上命令应分别执行，前一个成功退出后再运行后一个。`artifacts/run-logs`保存每次模型任务的输出、错误、异常堆栈和运行配置。

网页入口为`python -m catvision.server`，地址为`http://127.0.0.1:8765`。双击“安装环境.cmd”恢复独立环境，再双击“启动项目.cmd”；已有环境可通过`CATVISION_PYTHON`或`scripts/start.ps1 -PythonExe`指定解释器。自训练权重`weights/breed-best.pt`约11MB，随源码保留；较大的官方模型可从完整保留包恢复，或运行下载脚本获取。

## 数据与资料

`artifacts/data`保留训练950张、验证238张、官方测试1183张的来源清单，以及提前固定的120张共同测试清单。第一轮`data/classification`中有2138张视图，来自1188张训练/验证原图；原图与裁剪图共享来源。

各个模型的下载来源、自己训练的部分与直接调用的部分，见[模型来源与训练范围](docs/模型来源与训练范围.txt)。

要求原文与课程提要已另存于工作区的“第二阶段保留资料/要求依据”。项目接续时同时保留原始数据、生成视图、模型分片、运行记录和要求资料，详见[运行限制与数据保留](docs/运行限制与数据保留.txt)。

`scripts/package_project.py`用于收集完整保留包，显式接收原始数据、要求资料、视频来源及输出位置。它包含Git排除的数据、官方权重、分段和日志，使用无压缩ZIP减少CPU负载，不删除源文件。源码仓库保留本项目11MB分类权重，其他大文件通过完整保留包或官方来源恢复。实际运行视频位于完整包的`项目/artifacts/demo/实际运行演示.mp4`；长时间静止等待已压缩，界面显示的模型耗时未修改。

开发环境的可恢复版本保存在`requirements-lock.txt`，大型CUDA包来自PyTorch官方索引。原始pip freeze随完整保留包保存；执行安装时使用根目录依赖文件。

## 参考

方法参考Grounded-SAM-2的检测框引导分割、Pytorch-garbage-classify的训练/验证和图片接口组织，以及LAVIS的图像加问题调用流程。本项目独立实现猫品种任务。

- https://github.com/IDEA-Research/Grounded-SAM-2
- https://github.com/Candy-CY/Pytorch-garbage-classify
- https://github.com/salesforce/LAVIS
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

Oxford-IIIT Pet数据来源与许可：https://www.robots.ox.ac.uk/~vgg/data/pets/ 。Ultralytics代码及权重许可按其上游说明使用；Qwen与SAM各自遵循其模型许可。
