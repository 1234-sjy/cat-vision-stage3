"""Image-grounded descriptions, generated independently of the breed answer."""
import json
import re
from .common import BREEDS

FEATURE_PROMPT_VERSION = 'visible-cat-features-v1'
FEATURE_PROMPT = (
    '观察原图中的主体猫，只描述实际可见的外观，不判断品种。'
    '严格输出两行，每行不超过20个汉字。'
    '第一行写毛长、主要毛色或花纹中最清楚的特征；'
    '第二行写耳朵、脸型或眼睛中最清楚的特征。'
    '不要标题、品种名称、推测、解释或百分比。'
)


def parse_visible_features(text):
    descriptions = []
    aliases = [*BREEDS, *BREEDS.values()]
    for line in text.strip().splitlines():
        line = re.sub(r'^\s*(?:[-*•]|[（(]?\d+[)）.、:：])\s*', '', line).strip()
        line = line.strip('` \t。；;')
        if (not line or re.fullmatch(r'(?:可见|外观)?特征[:：]?', line)
                or '%' in line or '％' in line
                or any(name.casefold() in line.casefold() for name in aliases)):
            continue
        if line not in descriptions:
            descriptions.append(line[:120])
        if len(descriptions) == 2:
            break
    return descriptions


OBSERVATION_PROMPT_VERSION = 'image-observation-v3'
OBSERVATION_PROMPT = '''观察这张原始照片，围绕画面中的主体猫描述实际可见的内容，不猜品种。
若有多只猫，交代数量，并以面积最大的猫为主体；没有猫时描述照片实际主体。
只输出一个JSON对象，字段如下：
appearance: 2至3条简短中文外观观察，包含能看清的毛发或皮肤、颜色/花纹，以及耳朵、脸或眼睛；
pose: 一两句描述主体的姿态、朝向和画面位置；
scene: 一两句描述实际看清的背景、接触物或物品。
有毛的部位描述毛发，明显裸露的部位才写皮肤。看不清的细节略过，不为了凑数量补充。
只写能从照片核对的形状、颜色和位置，不推测性格、情绪、健康、地点名称或拍摄背景故事。
每个字段控制在两句话以内，文字具体自然，不输出品种名称、百分比、Markdown或解释。
结构示例：{"appearance":["外观观察","外观观察"],"pose":"姿态与位置","scene":"可见背景"}。'''


def parse_observation(text):
    result = {'features':[], 'pose':'', 'scene':'',
              'observation_status':'parse_error', 'observation_raw_text':text}
    start=text.find('{')
    try:
        if start < 0:
            return result
        data,_=json.JSONDecoder().raw_decode(text[start:])
    except (ValueError,TypeError):
        return result
    if not isinstance(data,dict):
        return result
    appearance=data.get('appearance')
    if isinstance(appearance,list):
        result['features']=[x.strip() for x in appearance if isinstance(x,str) and x.strip()][:4]
    for field in ['pose','scene']:
        if isinstance(data.get(field),str):
            result[field]=data[field].strip()
    result['observation_status']=('ok' if result['features'] and result['pose'] and result['scene'] else 'partial')
    return result


QUESTION_PROMPT_VERSION = 'image-question-v4-separated'


def split_image_questions(question):
    parts=[p.strip() for p in re.split(r'(?<=[?？])\s*',question.strip()) if p.strip()]
    if len(parts)>4:
        raise ValueError('一次最多询问4个问题，请分开发送')
    return parts


def image_question_prompt(question):
    return ('你正在做图像观察问答。直接观察原始照片，用中文回答下面的问题。'
            '如果问题询问了多个方面，按问题顺序用1、2等编号逐项回答，不要遗漏。'
            '每项用一句话，只回答所问的形状、颜色、位置、姿态或可见物品，不加开场介绍。'
            '描述同一部位要前后一致，不猜品种、情绪、性格、健康或背景故事。'
            '只写能在画面中核对的内容，照片没有拍到的内容简短指出。\n问题：'+question)
