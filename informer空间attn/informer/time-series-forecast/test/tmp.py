import argparse
parser = argparse.ArgumentParser(description='test')
parser.add_argument("--model_class", type=str, required=True, help='模型类型')
parser.add_argument("--model_name", type=str, required=True)