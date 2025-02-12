import argparse
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser(description='Example script with YAML configuration')

    # 添加需要的命令行参数
    parser.add_argument('--foo', type=int,default=321,  help='An integer parameter')
    parser.add_argument('--bar', type=str, help='A string parameter')
    parser.add_argument('--baz', type=bool, help='A boolean parameter')
    parser.add_argument('--lr_sched', type=lambda x: (str(x).lower() == 'true'), default=False)

    # 解析命令行参数
    args = parser.parse_args()

    # 如果没有命令行参数，尝试从配置文件中读取
    
    dict = {}
    dict['lr_sched'] = False

    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
        # 将配置文件中的参数添加到命令行参数中
        for key, value in config.items():
            setattr(args, key, value)
    
    # setattr(args, 'lr_sched',  dict['lr_sched'])

    return args

def main():
    args = parse_arguments()

    if args.lr_sched:
        print('nihO+++++')

    # 输出参数值
    print("Foo:", args.foo)
    print("Bar:", args.bar)
    print("Baz:", args.baz)

if __name__ == "__main__":
    main()
