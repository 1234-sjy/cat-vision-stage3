"""Make charts, documentation and saved examples from two completed real runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import BREEDS, ROOT, load_image, save_json
from catvision.evaluation import score_predictions, validate_comparison


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def answer(prediction):
    return prediction.get("breed_zh") or {
        "no_cat": "未检测到猫", "unsupported": "其他品种",
        "parse_error": "返回格式错误", "error": "运行失败"}.get(prediction.get("status"), "未给出品种")


def charts(metrics, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    titles = {"traditional": "传统视觉", "vlm": "视觉大模型"}
    colors = ["#b66b43", "#287764"]
    for route, values in metrics.items():
        matrix = np.asarray(values["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(12, 9))
        plot = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=10)
        ax.set(xticks=range(13), yticks=range(12),
               xticklabels=[BREEDS[k] for k in values["row_labels"]] + ["未给出品种"],
               yticklabels=[BREEDS[k] for k in values["row_labels"]],
               xlabel="预测品种", ylabel="真实品种", title=f"{titles[route]} · 同一组120张测试图")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        for row in range(12):
            for col in range(13):
                if matrix[row, col]:
                    ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                            color="white" if matrix[row, col] >= 6 else "#26342f", fontsize=10)
        fig.colorbar(plot, ax=ax, label="图片数")
        fig.tight_layout()
        fig.savefig(directory / f"{route}-confusion.png", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for index, route in enumerate(titles):
        values = metrics[route]
        positions = np.arange(2) + (index - .5) * .32
        heights = [values["accuracy"] * 100, values["macro_f1"] * 100]
        bars = axes[0].bar(positions, heights, width=.32, label=titles[route], color=colors[index])
        axes[0].bar_label(bars, fmt="%.1f", padding=3)
    axes[0].set(xticks=[0, 1], xticklabels=["Top-1正确率", "宏平均F1"], ylim=(0, 110), ylabel="百分制")
    axes[0].legend(frameon=False)
    durations = [metrics[r]["median_inference_seconds"] or 0 for r in titles]
    bars = axes[1].bar(list(titles.values()), durations, color=colors)
    axes[1].bar_label(bars, fmt="%.2f秒", padding=3)
    axes[1].set(ylabel="秒 / 张", title="推理耗时中位数，权重加载单独记录")
    axes[0].set_title("相同测试输入的识别结果")
    fig.tight_layout()
    fig.savefig(directory / "comparison.png", dpi=160)
    plt.close(fig)


def make_examples(truth, lookups, data_root):
    groups = [[], [], [], []]
    for record in truth:
        key = record["image_id"]
        a, b = (lookups[r][key] for r in ["traditional", "vlm"])
        good_a = a.get("status") == "ok" and a.get("breed") == record["breed"]
        good_b = b.get("status") == "ok" and b.get("breed") == record["breed"]
        groups[0 if good_a and good_b else 1 if good_a else 2 if good_b else 3].append(record)
    selected = [(group[0], label) for group, label in zip(groups,
        ["两边正确", "传统正确", "大模型正确", "两边误判"]) if group]
    picked = {r["image_id"] for r, _ in selected}
    for record in truth:
        if len(selected) >= 4:
            break
        if record["image_id"] not in picked:
            selected.append((record, BREEDS[record["breed"]]))
            picked.add(record["image_id"])
    examples = []
    for record, label in selected:
        key = record["image_id"]
        directory = ROOT / "examples" / key
        directory.mkdir(parents=True, exist_ok=True)
        load_image(data_root / record["relative_path"]).save(directory / "input.png")
        visual = ROOT / "artifacts/evaluation/comparison/traditional" / f"{key}.jpg"
        if visual.exists():
            shutil.copy2(visual, directory / "traditional.jpg")
        result = {r: lookups[r][key] for r in ["traditional", "vlm"]}
        result.update(fresh_run=False, total_seconds=sum(v.get("inference_seconds", 0) for v in result.values()))
        example = {"image_id": key, "label": label, "truth": record["breed_zh"],
                   "input_url": f"/examples/{key}/input.png",
                   "visualization_url": f"/examples/{key}/traditional.jpg", "result": result}
        save_json(directory / "result.json", example)
        examples.append(example)
    save_json(ROOT / "examples/index.json", examples)
    return examples


def documentation(metrics, examples, lookups, training):
    a, b = (metrics[r] for r in ["traditional", "vlm"])
    cases = []
    for example in examples:
        key = example["image_id"]
        cases.append(f"{key}：真实为{example['truth']}；传统视觉给出{answer(lookups['traditional'][key])}，"
                     f"视觉大模型给出{answer(lookups['vlm'][key])}。")
    text = f"""看图识猫：检测分割与视觉大模型的同图比较

项目做什么

上传一张猫的照片，两条路线独立识别，各给出一个品种。传统路线另外画出检测框和分割轮廓；视觉大模型写出简短外观特征。支持Oxford-IIIT Pet的12种猫品种，多只猫时关注画面中面积最大的猫。

第二阶段的方法用在哪里

传统路线先用YOLO26n检测猫，再把检测框交给SAM2.1，取得主体轮廓。程序按主体范围加6%边距裁剪RGB图片，保留毛色和花纹，交给本项目微调的YOLO26s-cls分类器。没有猫框时返回未检测到猫，不强行给品种。

大模型路线在本机运行Qwen3-VL-4B-Instruct，通过Transformers SDK传入原图和固定任务说明。提示中列出12个品种代码，要求只返回一个品种和最多3条可见特征。两条路线互不传递答案。Qwen采用官方预训练参数；本项目训练的是猫品种分类器。

这里的注意力可以理解为：模型计算图像或文字中哪些部分彼此相关，再按相关程度汇总信息。我们调用开源模型的这项能力，完成图片加文字输入和结构化输出。OpenCV/Pillow负责读取、裁剪和轮廓展示，FastAPI提供本地图片上传接口。

数据怎么分

官方trainval中的猫分为950张训练图和238张验证图，官方测试保留1183张。训练来源图各生成原图和主体图两种视图，共1900张训练视图；验证集每张保留一张主体视图，共238张。两种训练视图来自同一张原图，不计成两张独立来源。跨集合SHA256重复为0。

在测试预测前，按固定随机种子20261101选出每个品种10张，共120张共同测试图。两条路线实际使用这同一清单，逐图保存输入摘要、输出、状态与耗时。训练、阈值和提示调整使用训练或验证数据，测试答案不进入推理提示。

训练过程

分类器以YOLO26s-cls通用预训练权重为起点，输出改为12类，微调参数。输入320像素，batch=8，workers=0，AdamW初始学习率0.0008，余弦衰减，最多40轮，按验证fitness保存最佳权重。上游fitness为Top-1与Top-5准确率的平均值，界面与本项目比较只展示一个品种和Top-1正确率。

训练使用{training['gpu']}，PyTorch {training['torch']}，CUDA {training['cuda']}。训练记录耗时约{training['training_seconds']/60:.1f}分钟。此前中断的已完成轮次通过检查点接续，原检查点与日志保留。下载、训练和两路批量推理分开执行；训练每批同步GPU后等待0.05秒，推理也留出短暂停顿。运行错误会停止任务，不自动重试。

相同120张测试图片的结果

传统视觉：正确{a['correct']}/120，Top-1正确率{a['accuracy']:.2%}，宏平均F1为{a['macro_f1']:.4f}。
视觉大模型：正确{b['correct']}/120，Top-1正确率{b['accuracy']:.2%}，宏平均F1为{b['macro_f1']:.4f}。
传统视觉状态统计：{json.dumps(a['status_counts'], ensure_ascii=False)}。
视觉大模型状态统计：{json.dumps(b['status_counts'], ensure_ascii=False)}。

每张图都计入总数，漏检、其他品种、格式错误和运行失败均不当成正确答案。宏平均F1先给每个品种计算一个综合分，再对12个品种取平均。

传统视觉推理耗时中位数{a['median_inference_seconds']:.2f}秒/张；视觉大模型{b['median_inference_seconds']:.2f}秒/张。权重加载与主动等待时间单独记录，网页首次运行还会花时间加载模型。单张图片上的分类分数是模型对这张图的输出，不是整组测试准确率；大模型没有生成百分比分数。

实际案例

""" + "\n".join(cases) + """

实现感悟

这次项目让我更关注识别流程中的每一步。分类器即使能认品种，前面的检测器漏掉猫，整条路线还是会失败，所以报告保留了未识别结果。把两条路线放到同一张图上，也能看到“说得清楚”和“品种判断正确”需要分别检查。

环境复现同样影响交付。项目用依赖版本、相对资源位置和数据参数组织运行；换电脑先恢复环境和权重，再启动服务。开发中发现中断下载留下了全零区域，文件长度仍然正常，最终靠官方SHA256发现并修复。后续保存权重时增加了落盘同步，并保留训练日志和检查点。

项目目前支持12类。传统分类器会把检测到的猫归入这12类之一；大模型可以返回其他品种。无毛猫、遮挡、复杂背景和相近品种值得继续补充样本。后续改进会优先检查检测漏失和具体混淆类别，再决定是否增加数据或调整模型。

参考和AI辅助

本项目参考Grounded-SAM-2的检测框引导分割流程、Pytorch-garbage-classify的训练/验证与图片接口组织，以及LAVIS的图像加问题调用方式。代码由AI辅助开发，训练与两路预测在本机实际运行，具体输出以保存的实验记录为准。

参考项目：https://github.com/IDEA-Research/Grounded-SAM-2
参考项目：https://github.com/Candy-CY/Pytorch-garbage-classify
参考项目：https://github.com/salesforce/LAVIS
数据来源：https://www.robots.ox.ac.uk/~vgg/data/pets/
Qwen官方模型：https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct

完整课程提要、视频索引、PDF原文、已有模型数据和历史日志随完整保留包保存。
"""
    (ROOT / "docs/项目说明与实现感悟.txt").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    truth = read(ROOT / "artifacts/data/comparison.json")
    predictions = {r: read(ROOT / f"artifacts/evaluation/comparison/{r}/predictions.json")
                   for r in ["traditional", "vlm"]}
    validate_comparison(truth, *predictions.values())
    metrics = {r: score_predictions(truth, values) for r, values in predictions.items()}
    directory = ROOT / "artifacts/comparison"
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / "summary.json", metrics)
    lookups = {r: {p["image_id"]: p for p in values} for r, values in predictions.items()}
    with (directory / "逐图对比.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["图片", "真实品种", "传统视觉", "视觉大模型", "传统状态", "大模型状态"])
        for record in truth:
            key = record["image_id"]
            writer.writerow([key, record["breed_zh"], answer(lookups["traditional"][key]),
                             answer(lookups["vlm"][key]), lookups["traditional"][key]["status"],
                             lookups["vlm"][key]["status"]])
    charts(metrics, directory)
    examples = make_examples(truth, lookups, args.data_root)
    documentation(metrics, examples, lookups, read(ROOT / "artifacts/training/final-selection.json"))
    print("REPORTS AND REAL EXAMPLES GENERATED", flush=True)


if __name__ == "__main__":
    main()
