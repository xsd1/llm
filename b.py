import re
import json
import os
# with open('submit/result.json', 'r') as f:
#     data = json.load(f)
# for _ in data:
#     text = _['pred']
#     matches = re.findall(r"Response.* The questioner feels (\w+) because (.+).*###", text)
#     print(matches)
#     for match in matches:
#         feeling = match[0]
#         explanation = match[1]
#         print("Feeling:", feeling)
#         print("Explanation:", explanation)

base = '/data/xsd/data/train'
with open(os.path.join(base, 'train_1.json'), 'r') as f:
    data = json.load(f)

with open(os.path.join(base, 'train_2.json'), 'r') as f:
    data = json.load(f)

with open(os.path.join(base, 'train_3.json'), 'r') as f:
    data = json.load(f)

with open(os.path.join(base, 'train_4.json'), 'r') as f:
    data = json.load(f)
print()