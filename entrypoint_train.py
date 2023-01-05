import argparse
#argsparse是python的命令行解析的标准模块，内置于python，不需要安装。
#这个库可以让我们直接在命令行中就可以向程序中传入参数并让程序运行。
import sys

from functions.trainer import Trainer
from options import update_options, options, reset_options


def parse_args():
    #实例化
    #创建一个ArgumentParser对象
    #它包含将命令行解析成python数据类型所需的全部信息
    #description简要描述这个程序做什么以及怎么做
    parser = argparse.ArgumentParser(description='Pixel2Mesh Training Entrypoint')
    #指定ArgumentParser如何获取命令行字符串将其转换为对象
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    #只解析已知的argument 作用和parse_args()一样
    #返回已知Namespace()+ 其他乱七八糟[...,...]

    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    #首先导入--option path/to/yaml 中配置options
    
    #然后导入训练需要的参数 如batch size，checkpoint...

    #training
    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument('--num-epochs', help='number of epochs', type=int)
    parser.add_argument('--version', help='version of task (timestamp by default)', type=str)
    parser.add_argument('--name', required=True, type=str)

    #解析（动词）参数
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    #options含有一系列原始的+yaml文件规定的训练参数
    #args里面含有的是 命令行中额外输入的参数

    logger, writer = reset_options(options, args)
    #logger和writer还没看  估计是记录训练数据的？？
    
    trainer = Trainer(options, logger, writer)
    trainer.train()


if __name__ == "__main__":
    main()
