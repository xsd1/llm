import argparse

# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description='这是一个演示根据参数构建params的示例')

# 添加命令行参数
parser.add_argument('--name', type=str, help='姓名')
parser.add_argument('--age', type=int, help='年龄')
parser.add_argument('--output_dir', type=str, help='年龄')

# 解析命令行参数
args = parser.parse_args()

# 获取params字典
params = args.__dict__
import os, yaml
os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(params, f)
print("params:", params)
