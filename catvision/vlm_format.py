"""A compact two-line response avoids fragile generated JSON punctuation."""
import re
from .common import BREEDS


def parse_breed_lines(text: str) -> dict:
    base={'status':'parse_error','has_cat':None,'breed':None,'breed_zh':None,
          'features':[],'raw_text':text}
    lines=[line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines: return base
    code=lines[0].strip().strip('`').strip()
    code=re.sub(r'^(?:breed|品种)\s*[:=：]\s*','',code,flags=re.IGNORECASE)
    if code.casefold()=='no_cat':
        return {**base,'status':'no_cat','has_cat':False}
    features=[v.strip(' -；;。') for v in re.split('[、；;\n]', '\n'.join(lines[1:])) if v.strip(' -；;。')]
    if code.casefold() in {'other','unknown'}:
        return {**base,'status':'unsupported','has_cat':True,'breed_zh':'其他品种',
                'features':features[:3]}
    aliases={name.replace('_',' ').casefold():name for name in BREEDS}
    aliases.update({name:code for code,name in BREEDS.items()})
    breed=aliases.get(code.replace('_',' ').casefold())
    if breed is None: return base
    return {**base,'status':'ok','has_cat':True,'breed':breed,'breed_zh':BREEDS[breed],
            'features':[feature[:160] for feature in features[:3]]}
