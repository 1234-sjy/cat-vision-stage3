"""Versioned visual-only instructions; never consume classifier predictions."""
from .common import BREEDS


def breed_prompt(version: str) -> str:
    candidates = '\n'.join(f'{code}: {name}' for code,name in BREEDS.items())
    if version == 'expanded-baseline-v3':
        return ('观察图片中面积最大的真实猫，根据可见外观识别一个品种。候选代码：\n'+candidates+
                '\n只输出JSON：{"has_cat":true,"breed":"候选代码","features":["可见特征"]}。'
                'features最多3条简短中文外观特征，没有猫时has_cat为false、breed为null。'
                '明确是候选之外的品种时breed为other。不要多个候选，不要百分比，不要根据文件名判断。')
    if version == 'morphology-lines-v3':
        original=breed_prompt('morphology-v3')
        return original[:original.index('\n只输出JSON')]+(
            '\n严格输出两行，不要JSON或Markdown。第一行只写一个英文品种代码。'
            '第二行写两条简短的中文可见外观特征，用顿号隔开。没有猫时第一行只写no_cat；'
            '范围外品种、普通家猫或混种时第一行写other。')
    if version != 'morphology-v3':
        raise ValueError('Unknown prompt version')
    return ('识别照片中主体猫的品种。先观察耳朵形状、脸和鼻口、毛长与花纹，'
            '再结合这些外观给出一个最匹配的品种。不要仅凭毛色判断。候选代码：\n'+candidates+
            '\n区分要点：Scottish_Fold耳朵向前折；American_Curl耳尖向后卷。'
            'Persian长毛扁脸，Exotic_Shorthair短毛扁脸。'
            'Maine_Coon体大、方形口鼻、长毛耳簇，Norwegian_Forest三角脸直鼻。'
            'British_Shorthair脸圆、毛密；Russian_Blue脸较窄、绿眼睛。'
            'Siamese短毛重点色，Ragdoll与Birman长毛蓝眼重点色，Birman通常四爪白手套。'
            'American_Shorthair可以有多种毛色，虎斑不是独立品种。'
            '普通家猫、混种或看不清特征时允许other，不要强行猜一个纯种。'
            '\n只输出JSON，先列2条简短中文可见特征，再填一个英文代码：'
            '{"features":["特征1","特征2"],"has_cat":true,"breed":"候选代码"}。'
            '没有真实猫时has_cat为false、breed为null。不要多个候选、不要百分比。')
