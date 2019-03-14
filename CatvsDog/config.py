# -*- coding: utf-8 -*-
"""
@ project: CatvsDog
@ author: lzx
@ file: config.py
@ time: 2019/3/13 20:12
"""
class Config():
    model = 'SqueezeNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_root = 'F:/刘子绪/数据/数据image/kaggle/train/train/'
    test_root = 'F:/刘子绪/数据/数据image/kaggle/test1/test1/'
    load_model_path = 'checkpoints/squeezenet_0313_21_53_57.pth'  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 32  # batch size
    use_gpu = True  # use GPU or not
    num_workers = 4  # how many workers for loading data

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数