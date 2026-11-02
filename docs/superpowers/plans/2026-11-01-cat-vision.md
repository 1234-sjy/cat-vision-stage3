# Cat Vision Stage 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 独立完成猫品种识别的传统视觉组合与本地视觉大模型，并交付真实的同输入比较。

**Architecture:** Python 包分离数据、传统视觉、视觉语言模型、实验评价与本地服务。FastAPI 托管简单网页，模型在单个 GPU 工作队列中按阶段加载，结果保存为 JSON 与 PNG。

**Tech Stack:** Python 3.12、PyTorch CUDA 12.8、Ultralytics YOLO26/SAM2.1、Transformers Qwen3-VL、FastAPI、Pillow、OpenCV、NumPy。

**Spec:** docs/superpowers/specs/2026-11-01-cat-vision-design.md

## Global Constraints

- 新项目不修改或加载第一阶段训练权重。
- 两路线使用同一原图；12个猫品种各输出一个结论，不显示Top-3。
- 训练／验证／测试无跨集合来源泄漏；共同测试子集在测试推理前固定为每类10张。
- 图片推理在本机GPU执行，不调用收费模型服务。
- 自训练分类权重约11MB随源码保留；官方大型权重、原始数据、密钥、虚拟环境不入库，资源位置可配置。

## Review Focus

- 无猫、多个猫和未支持类别的结果不能伪装成高分正确识别。
- EXIF旋转、中文路径、错误图片和大图的读取与坐标一致。
- 大模型非JSON、缺字段和不在类别表的返回要保留原文并标记解析状态。
- 检测框／mask为空，不能切出空图；回退路径必须记录。
- 耗时区分加载与推理，缓存结果与本次实际运行有明确标记。

## Task 1: Data and shared contracts

Files: catvision/data.py、catvision/common.py、tests/test_data.py、tests/test_contracts.py、pyproject.toml、requirements.txt。
Produces: prepare_manifest(data_root: Path, out_dir: Path) -> dict；load_image(bytes_or_path) -> PIL.Image；parse_vlm_output(text: str) -> dict。
- [ ] 先写无泄漏划分、来源继承、错误图片和品种输出验证测试，运行看到尚未实现的失败。
- [ ] 实现清单与共同测试选择、图片规范化和结构化返回。
- [ ] 运行 unittest 测试及实际原始数据整理，记录统计；提交真实进展。

## Task 2: Detection and segmentation

Files: catvision/traditional.py、scripts/prepare_views.py、tests/test_geometry.py。
Consumes: Task 1 图像与类别契约。
Produces: extract_subject(image: PIL.Image) -> dict，含box、mask、crop、状态与各段耗时。
- [ ] 写空框／边界截断／mask尺寸与主体选择测试并观察失败。
- [ ] 接通YOLO26猫检测与SAM2.1框提示；生成训练／验证视图并保留来源。
- [ ] 运行真实图片检查与视图生成，保存版本和配置；提交。

## Task 3: Breed training

Files: scripts/train_breed.py、catvision/traditional.py、artifacts/training/。
Consumes: Task 2 视图数据；Produces: weights/breed-best.pt及训练记录。
- [ ] 用验证集选择YOLO26分类模型，记录固定参数与每轮结果。
- [ ] 实际GPU微调，按验证成绩保存权重；接通单品种预测与no-cat状态。
- [ ] 保存真实训练产物及版本，提交源码与小记录。

## Task 4: Local vision-language model

Files: catvision/vlm.py、scripts/download_models.py、configs/default.json、tests/test_contracts.py。
Consumes: 共同原图与固定类别表；Produces: LocalVLM.predict(image: PIL.Image) -> dict。
- [ ] 验证非法返回处理测试先失败，再实现规范化。
- [ ] 下载官方4B权重，以BF16/SDPA真实运行文本加图像，保存原始返回。
- [ ] 验证集试运行后固定提示；提交代码与配置。

## Task 5: Comparison

Files: scripts/evaluate.py、scripts/make_report.py、artifacts/comparison/。
Consumes: 固定共同测试清单、两路线predict；Produces: 逐图记录、指标、图表和分歧案例。
- [ ] 写失败计入总数与类别对齐的评价测试并观察失败，再实现评价。
- [ ] 分阶段真实运行两路线，记录耗时、错误和原始结果；输出比较表。
- [ ] 写实际结果与方法说明，使用humanizer-zh；提交。

## Task 6: Local app and delivery

Files: catvision/server.py、web/index.html、scripts/start.ps1、README.md、tests/test_api.py、docs/项目说明.txt。
Consumes: 两路线与评价记录；Produces: 本机网页、API、独立项目成果包与实际运行视频。
- [ ] 先写损坏图片和正常图片请求契约测试并观察失败，再实现上传／单GPU队列／并排展示。
- [ ] 真实运行本地界面，展示同输入双路线与每边一个品种。
- [ ] 保存可恢复依赖、模型说明、实际录像和成果；只做与功能有关的核对，不扩大验证范围。
