# -*- coding: utf-8 -*-
"""
@ project: CatvsDog
@ author: lzx
@ file: main.py
@ time: 2019/3/13 19:59
"""
# from torchnet import meter
import config
from dataset import CatDog
from torch.utils.data import DataLoader
import torch
import model
from torchnet import meter
from tqdm import tqdm
import torch.nn.functional as F
# print(config.Config)
opt= config.Config()
# print(opt.lr)
'''将结果存入csv文件'''
def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)
'''验证集测试'''
def val(model,dataloader):
    with torch.no_grad():
        model.eval()
        confusion_matrix = meter.ConfusionMeter(2)
        for ii, (val_input, label) in (enumerate(dataloader)):
            if torch.cuda.is_available():
                val_input = val_input.cuda()
            label = label
            score = model(val_input)
            confusion_matrix.add(score.detach().squeeze(), label.type(torch.LongTensor))

        model.train()
        cm_value = confusion_matrix.value()
        accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
        return cm_value, accuracy
'''训练'''
def train():
    models = getattr(model, opt.model)()
    print(models)
    models.cuda()
    train_dataset = CatDog(root=opt.train_root)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    valid_dataset = CatDog(root=opt.train_root,train=False,test=False)
    # print(len(valid_dataset))
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = models.get_optimizer(lr,opt.weight_decay)
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)

    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(train_dataloader)):

            # train model
            if torch.cuda.is_available():
                input = data.cuda()
                target = label.cuda()

            optimizer.zero_grad()
            score = models(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.item())
            # detach 一下更安全保险
            confusion_matrix.add(score.detach(), target.detach())
        print('epoch:{};loss:{};confusion_matrix:{}'.format(epoch,loss_meter.value()[0],str(confusion_matrix.value())))
        val_cm, val_accuracy = val(models, valid_dataloader)
        print('valid confusion：{}；val_accuracy:{}'.format(val_cm,val_accuracy))


    models.save()

def test():
    '''
    测试
    :return:
    '''
    models = getattr(model, opt.model)().eval()
    print(models)
    if opt.load_model_path:
        models.load(opt.load_model_path)
    if torch.cuda.is_available():
        models.cuda()
    test_data = CatDog(root=opt.test_root, train=False, test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    for i,(data,name) in tqdm(enumerate(test_dataloader)):
        if torch.cuda.is_available():
            data = data.cuda()
            output = models(data)
            prob = F.softmax(output,dim=1)[:,0].detach().tolist()

            batch_reslut = [(path_.item(),prob_) for (path_,prob_) in zip(name,prob)]
            results += batch_reslut
    write_csv(results, opt.result_file)
    return results
            # print(prob)
        # print(name)


if __name__ == '__main__':
    # train()
    test()