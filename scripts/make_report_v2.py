"""Publish measured V2 results while keeping all V1 records."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT,BREEDS,load_image,save_json
from catvision.evaluation import validate_comparison,score_predictions
from make_report import charts,answer


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--stage1-predictions',type=Path,required=True)
    args=p.parse_args()
    out=ROOT/'artifacts/comparison-v2';out.mkdir(exist_ok=True)
    truth=read(ROOT/'artifacts/optimization-v2/comparison-v2.json')
    predictions={r:read(ROOT/f'artifacts/evaluation/comparison-v2/{r}/predictions.json') for r in ['traditional','vlm']}
    validate_comparison(truth,*predictions.values())
    metrics={r:score_predictions(truth,values) for r,values in predictions.items()}
    save_json(out/'summary.json',metrics);charts(metrics,out)
    lookups={r:{v['image_id']:v for v in values} for r,values in predictions.items()}
    old=read(ROOT/'artifacts/data/comparison.json')
    baseline={r['image_id']:r for r in csv.DictReader(args.stage1_predictions.open(encoding='utf-8-sig'))}
    baseline_new=sum(baseline[r['image_id']]['correct']=='True' for r in truth)
    baseline_old=sum(baseline[r['image_id']]['correct']=='True' for r in old)
    save_json(out/'stage1-saved-baseline.json',{'source':'previous stage1 optimized/predictions.csv; no stage1 model loaded',
                'full_correct':sum(r['correct']=='True' for r in baseline.values()),'full_count':len(baseline),
                'old120_correct':baseline_old,'new120_correct':baseline_new,
                'new120_image_ids':[r['image_id'] for r in truth]})
    with (out/'逐图对比.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['图片','真实品种','改进传统视觉','视觉大模型','第一阶段保存的预测','传统状态','大模型状态'])
        for r in truth:
            key=r['image_id'];a=lookups['traditional'][key];b=lookups['vlm'][key]
            writer.writerow([key,r['breed_zh'],answer(a),answer(b),BREEDS[baseline[key]['pred_breed']],a['status'],b['status']])
    # Preserve existing displayed examples before replacing the index.
    index=ROOT/'examples/index.json'
    if index.exists() and not (ROOT/'examples/index-v1.json').exists():shutil.copy2(index,ROOT/'examples/index-v1.json')
    groups=[[],[],[],[]]
    for r in truth:
        a=lookups['traditional'][r['image_id']];b=lookups['vlm'][r['image_id']]
        ca=a['status']=='ok' and a.get('breed')==r['breed'];cb=b['status']=='ok' and b.get('breed')==r['breed']
        groups[0 if ca and cb else 1 if ca else 2 if cb else 3].append(r)
    examples=[]
    for group,label in zip(groups,['两路正确','传统正确','大模型正确','两路待改进']):
        if not group:continue
        r=group[0];key=r['image_id'];directory=ROOT/'examples'/key;directory.mkdir(exist_ok=True)
        load_image(args.data_root/r['relative_path']).save(directory/'input.png')
        shutil.copy2(ROOT/f'artifacts/evaluation/comparison-v2/traditional/{key}.jpg',directory/'traditional.jpg')
        result={route:lookups[route][key] for route in ['traditional','vlm']}
        result['fresh_run']=False
        example={'image_id':key,'label':label,'truth':r['breed_zh'],
                 'input_url':f'/examples/{key}/input.png','visualization_url':f'/examples/{key}/traditional.jpg','result':result}
        save_json(directory/'result.json',example);examples.append(example)
    save_json(index,examples)
    history=ROOT/'docs/history/v1';history.mkdir(parents=True,exist_ok=True)
    for name in ['项目说明与实现感悟.txt','项目说明与实现感悟.pdf','答辩速查.txt']:
        path=ROOT/'docs'/name
        if path.exists() and not (history/name).exists():shutil.copy2(path,history/name)
    best=read(ROOT/'artifacts/optimization-v2/training/best.json')
    complete=read(ROOT/'artifacts/optimization-v2/training/complete.json')
    choice=read(ROOT/'artifacts/optimization-v2/selection.json')['selected']
    config=read(ROOT/'configs/active-classifier.json')
    a,b=metrics['traditional'],metrics['vlm']
    fullpath=ROOT/'artifacts/optimization-v2/full-original-test/metrics.json'
    full=read(fullpath) if fullpath.exists() else None
    full_text=(f"分类器只看原图的官方全量猫测试：{full['correct']}/{full['count']}，正确率{full['accuracy']:.2%}；"
               f"另测100张官方测试狗图，{full['dog_false_cat']}张被误判成猫。全量含此前排查过的旧120张；本轮未用于排查的新共同120张成绩单独列出。这个全量成绩对应原图分类分支，带定位和轮廓展示的实际流程成绩以上面的共同120张为准。") if full else ''
    text=f'''看图识猫：检测分割、Transformer与视觉大模型

项目做什么

上传一张猫图，两条路线独立读取同一输入，各给出一个品种。传统视觉画出猫框和分割轮廓；本地视觉大模型补充简短外观特征。支持Oxford-IIIT Pet的12种猫。照片以一只主要猫为宜；多只猫品种不同时，整图和主体的判断可能不同。

第二阶段的方法

传统路线用YOLO26n定位猫，以检测框提示SAM2.1分割，再按主体范围加6%边距裁剪RGB图片。本次独立微调Swin-Tiny分类器。验证集比较原图、主体图及融合后，完整原图的结果最好，因此本版品种分类只使用原图；YOLO与SAM负责主体位置和轮廓展示。检测漏掉时，分类器继续看原图。共享的Transformer特征分别送给12类品种头和猫/狗物种头；物种头认为不是猫时不输出猫品种。

Swin把图像分成小块，在窗口内计算各块之间的关系，再通过移动窗口交换信息。这让模型能够结合耳形、脸部、毛色和花纹。我们从官方ImageNet通用权重开始微调，没有套用第一阶段已经训练好的权重。检测和SAM采用官方参数，本项目训练Swin骨干与两个输出头。

另一条路线是本地Qwen3-VL-4B-Instruct。Transformers SDK真正接收原图和固定的文字说明，返回单个品种与最多3条可见特征。提示、192个输出token上限、BF16和SDPA设置沿用旧版；两条路线不传递彼此答案。

为什么改这次流程

第一阶段88.84%对应全部1183张测试；取出旧共同120张，第一阶段97张正确（80.83%），旧第二阶段88张（73.33%）。旧第二阶段15张没有检测框，分类器没有运行。在其余105张检测成功的图片上，第一阶段认对85张，旧第二阶段认对88张；整条路线的主要损失发生在漏检门槛。在238张验证图上，旧硬门槛流程认对197张，同一旧分类器直接看原图认对212张。这次先修正定位失败就终止识别的流程，再升级分类器。

数据和训练

猫仍是950张训练原图、238张验证原图和1183张官方测试原图。训练随机使用猫原图或已有主体视图，两种视图不算两张独立来源。另从官方trainval中固定1000张狗作训练负例，500张狗作验证负例，旧检测实验用过的50张狗保留在验证侧。猫/狗训练来源共1950张；验证共738张。狗负例帮助区分猫与狗，其成绩不代表所有背景和物体的拒识能力。

输入224像素，batch8、workers0、CPU线程2。先冻结骨干训练2轮，再微调全部参数；AdamW，骨干学习率0.000015，头部0.0005，余弦衰减，品种标签平滑0.05，物种损失系数0.5。每批同步GPU后等待0.05秒。训练完成{complete['epochs_completed']}轮，耗时{complete['seconds']/60:.1f}分钟；最佳检查点第{best['epoch']}轮。选模只看验证集原图品种正确数，同分时优先狗误判少的检查点。

选模后在验证集一次比较原图/主体权重和水平翻转平均。本次主体权重{config['subject_weight']:.2f}，水平翻转平均{'开启' if config['tta'] else '关闭'}，猫判断阈值{config['cat_threshold']:.4f}。选择后的验证结果为{choice['correct']}/238，{choice['false_cat_dogs']}/500张狗误报成猫。权重以FP16保存方便入库，推理载入FP32模型；导出后重新运行了上述验证。原FP32与每轮权重全部保留。

相同120张新测试图片

训练开始前固定种子20261102，每类10张，从未进入旧共同120张的官方测试图中抽取。旧测试只用于排查和回归，这批新图用于本次独立测试；参数冻结后不再根据测试结果调整。

改进传统视觉：{a['correct']}/120，Top-1正确率{a['accuracy']:.2%}，宏平均F1 {a['macro_f1']:.4f}。
本地视觉大模型：{b['correct']}/120，Top-1正确率{b['accuracy']:.2%}，宏平均F1 {b['macro_f1']:.4f}。
第一阶段已保存的同图预测：{baseline_new}/120，正确率{baseline_new/120:.2%}，只读取保存结果，没有加载第一阶段模型。

改进传统视觉的未给品种数{120-a['status_counts'].get('ok',0)}；视觉大模型格式错误{b['status_counts'].get('parse_error',0)}次，其他品种{b['status_counts'].get('unsupported',0)}次。每张都计入分母，失败不删除。单张分类分数表示模型对这张图的输出，整组测试正确率需要逐张对照真实标签后统计。

{full_text}

实际案例

'''+'\n'.join(f"{e['image_id']}：真实{e['truth']}；传统视觉{answer(e['result']['traditional'])}，大模型{answer(e['result']['vlm'])}。" for e in examples)+f'''

运行与实现体会

检测、分割提供位置和轮廓；本次品种判断保留完整原图，主体裁剪没有参与最终分类。

改进传统路线推理中位数{a['median_inference_seconds']:.2f}秒/张，视觉大模型{b['median_inference_seconds']:.2f}秒/张。权重加载和主动等待另记。下载、训练和两路推理依次执行，发生错误保存完整日志并停止，不自动重试或重启。

这次改动保留了检测与分割，让它们完成位置和轮廓展示。能否识别品种还要靠适合细粒度分类的特征，以及原图中完整的耳朵、脸部和花纹。验证集上的消融帮助找到了损失发生的位置，避免一看到分数低就盲目增加图片。后续补图应围绕相近品种、无毛猫、遮挡和多猫场景，并按来源隔开训练与测试。

环境使用独立Python环境和版本化依赖；资源按项目根目录寻找，外部数据通过参数传入。原始PDF、68集课程视频提要、参考仓库方法说明、旧版数据和日志都保留。第一阶段与第二阶段项目彼此独立。

参考和AI辅助

方法参考Grounded-SAM-2的检测框引导分割、Pytorch-garbage-classify的训练与验证组织、LAVIS的图文SDK调用。AI辅助编写和排查代码；报告中的训练、测试和逐图输出来自本机实际运行。

数据：https://www.robots.ox.ac.uk/~vgg/data/pets/
Swin：https://pytorch.org/vision/stable/models/swin_transformer.html
参考：https://github.com/IDEA-Research/Grounded-SAM-2
参考：https://github.com/Candy-CY/Pytorch-garbage-classify
参考：https://github.com/salesforce/LAVIS
Qwen：https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct
'''
    (ROOT/'docs/项目说明与实现感悟.txt').write_text(text,encoding='utf-8')
    print('V2 REPORT AND SAVED EXAMPLES GENERATED',flush=True)


if __name__=='__main__':main()
