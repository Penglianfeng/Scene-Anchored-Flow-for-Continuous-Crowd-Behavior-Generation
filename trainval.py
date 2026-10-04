import argparse
from utils.config import get_config, print_arguments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_config',
        type=str,
        default='./configs/model/SAFE_eth.yaml',
        help='Path to a model config file',
    )
    parser.add_argument('--dataset_config', type=str, default=None, help='Path to a trainer config file (optional). If not provided, the default config will be used.')
    parser.add_argument('--trainer_config', type=str, default=None, help='Path to a trainer config file (optional). If not provided, the default config will be used.')
    parser.add_argument('--model_train', type=str, default='emitter', help='Stage of the experiment', choices=['emitter_pre', 'emitter', 'simulator'])
    parser.add_argument('--test', default=False, action='store_true', help='Evaluation mode.')
    parser.add_argument('--export', default=False, action='store_true', help='Visualization mode.')
    parser.add_argument('--synthetic', default=False, action='store_true', help='Use synthetic dataset for inference.')
    args = parser.parse_args()

    # 合并模型、数据集、训练器的三份配置文件
    config = get_config(args.model_config, args.dataset_config, args.trainer_config)
    
    # Print the arguments and configs
    print('===== Arguments =====')
    print_arguments(vars(args))

    print('===== Configs =====')
    print_arguments(config)

    # Import the appropriate pipeline
    if args.test:
        from SAFM.evaluate import *
    elif args.export:
        from SAFM.evaluate_export_generated_traj import *
    elif args.synthetic:
        from SAFM.evaluate_synthetic_dataset import *
    elif args.model_train == 'emitter_pre':
        from SAFM.emitter.emitter_pre_trainer import *
    elif args.model_train == 'emitter':
        from SAFM.emitter.emitter_trainer import *
    elif args.model_train == 'simulator':
        from SAFM.simulator.simulator_trainer import *
        
    main(config)
