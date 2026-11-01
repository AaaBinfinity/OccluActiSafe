"""在数据所在电脑打包N17服务器输入/已提交代码；不训练，不包含测试集。"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from occluactisafe.experiments.detector_portable import export_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, help='N17节点目录；默认原冻结配置')
    parser.add_argument('--destination-parent', type=Path, help='默认移动硬盘项目/服务器输入/N17')
    args = parser.parse_args()
    print(export_input(args.source_root, args.destination_parent), flush=True)


if __name__ == '__main__':
    main()
