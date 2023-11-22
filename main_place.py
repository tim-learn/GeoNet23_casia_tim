import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx, make_dataset
import random, pdb, math, copy
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from torchvision.transforms.functional import InterpolationMode

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=1.0):  # power=0.75
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay

    return optimizer

def image_transform(args):
    net = args.net
    size_dic = {
        'swinb': [238, 224],
        'swins': [246, 224]
    }

    if net in ['resnet50']:
        resize = 256
        crop = 224
        interpolation = InterpolationMode.BILINEAR
    elif net in ['swinb']:
        resize, crop = size_dic[net]
        interpolation = InterpolationMode.BICUBIC

    train_transform = transforms.Compose([
                transforms.Resize((resize, resize), interpolation=interpolation),
                transforms.RandomCrop(crop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
    test_transform = transforms.Compose([
                transforms.Resize((resize, resize), interpolation=interpolation),
                transforms.CenterCrop(crop),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
    return train_transform, test_transform

def data_load(args): 
    image_train, image_test = image_transform(args)

    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test1 = open(args.test_dset_path).readlines()
    txt_test2 = open(args.test2_dset_path).readlines()

    dsets["source_tr"] = ImageList(txt_src, transform=image_train)
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    if args.sampler == 'weight':
        source_tr_images = make_dataset(txt_src, None)
        source_tr_weights = make_weights_for_balanced_classes(source_tr_images, args.class_num)
        count = [0] * args.class_num                                                      
        for item in source_tr_images:                                                         
            count[item[1]] += 1 

        source_tr_weights = torch.DoubleTensor(source_tr_weights)                                       
        sampler = torch.utils.data.sampler.WeightedRandomSampler(source_tr_weights, len(source_tr_weights))   
        dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False, sampler=sampler)

    else:
        source_tr_images = make_dataset(txt_src, None)
        source_tr_weights = make_weights_for_balanced_classes(source_tr_images, args.class_num)
        count = [0] * args.class_num                                                      
        for item in source_tr_images:                                                         
            count[item[1]] += 1 

    dsets["target_tr"] = ImageList_idx(txt_tar, transform=image_train)
    dset_loaders["target_tr"] = DataLoader(dsets["target_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    dsets["target_tr_t"] = ImageList(txt_tar, transform=image_test)
    dset_loaders["target_tr_t"] = DataLoader(dsets["target_tr_t"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    dsets["target_te"] = ImageList(txt_test1, transform=image_test)
    dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    dsets["test"] = ImageList(txt_test2, transform=image_test)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders, count

def make_weights_for_balanced_classes(images, nclasses): 
    # from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703                       
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N / (float(count[i]))                                
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight

def cal_acc(loader, net):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = net(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)

    _, pred = all_output.topk(3, 1, True, True)
    pred = pred.t()
    correct = pred.eq(all_label[None])
    res = []
    
    acc_1 = correct[:1].flatten().sum(dtype=torch.float32)/float(all_label.size()[0])*100
    acc_3 = correct[:3].flatten().sum(dtype=torch.float32)/float(all_label.size()[0])*100

    return acc_1, acc_3


def train_source(args):
    dset_loaders, per_class_num = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(net_name=args.net).cuda()
    elif args.net[0:3] == 'swi':
        netF = network.swin(net_name=args.net).cuda()  

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]  

    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = len(dset_loaders["source_tr"])
    iter_num = 0

    model = nn.Sequential(netF, netB, netC)
    model.train()

    if args.net[0:3] == 'res':
        netF2 = network.ResBase(net_name=args.net).cuda()
    elif args.net[0:3] == 'swi':
        netF2 = network.swin(net_name=args.net).cuda()  
    netB2 = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC2 = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    ema_model = nn.Sequential(netF2, netB2, netC2)
    ema_model.eval()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1

        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = model(inputs_source)
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)  

        if iter_num > int(0.4 * max_iter):
            with torch.no_grad():
                ema_model_out = ema_model(inputs_source)
                ema_model_softout = nn.Softmax(dim=1)(ema_model_out)

            softmax_out = nn.Softmax(dim=1)(outputs_source)
            ema_loss = - torch.sum(ema_model_softout * torch.log(softmax_out)) / ema_model_softout.size(0)
            classifier_loss += 1.0 * ema_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter or iter_num == 100000:
            model.eval()
            acc1, acc3 = cal_acc(dset_loaders['target_te'], model)
            log_str = 'Iter:{}/{}; Accuracy = {:.2f}/ {:.2f}'.format(iter_num, max_iter, acc1, acc3)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc3 >= acc_init:
                acc_init = acc3
                best_net = model.state_dict()
                torch.save(best_net, osp.join(args.output_dir_src, "best_source.pt"))

            model.train()

        if iter_num == int(0.4 * max_iter):
            ema_model.load_state_dict(model.state_dict())

        if iter_num > int(0.4 * max_iter) and iter_num % interval_iter == 0:
            alpha = 0.95
            for mean_param, param in zip(ema_model.parameters(), model.parameters()):
                mean_param.data = alpha * mean_param.data + (1.0 - alpha) * param.data

    torch.save(model.state_dict(), osp.join(args.output_dir_src, "source.pt"))

def test_target(args):
    dset_loaders, _ = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(net_name=args.net).cuda()
    elif args.net[0:3] == 'swi':
        netF = network.swin(net_name=args.net).cuda()   

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    model = nn.Sequential(netF, netB, netC)
    
    model.load_state_dict(torch.load(args.modelpath))
    model.eval()

    start_test = True
    with torch.no_grad():
        iter_test = iter(dset_loaders['test'])
        for i in range(len(dset_loaders['test'])):
            data = iter_test.next()
            inputs = data[0].cuda()
            outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    for i in range(len(dset_loaders['test'].dataset)):
        log_str = 'image_{:06d}.jpg {:d}'.format(i, predict[i])
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        # print(log_str+'\n')

    print('test finished!')


def train_target(args):
    dset_loaders, _ = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(net_name=args.net).cuda()
    elif args.net[0:3] == 'swi':
        netF = network.swin(net_name=args.net).cuda()   

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    model = nn.Sequential(netF, netB, netC)
    model.load_state_dict(torch.load(args.modelpath))

    param_group = []
    for k, v in model[0].named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.1}]

    for k, v in model[1].named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    for k, v in model[2].named_parameters():
        v.requires_grad = False

    model[0].train()
    model[1].train()
    model[2].eval()

    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
    optimizer = op_copy(optimizer)

    max_iter = 1 * len(dset_loaders["target_tr"])
    interval_iter = len(dset_loaders["target_tr"])
    iter_num = 0
    acc_init = 0

    model[0].eval()
    model[1].eval()
    acc1, acc3 = cal_acc(dset_loaders['target_te'], model)
    log_str = 'Iter:{}/{}; Accuracy = {:.2f}/ {:.2f}'.format(0, max_iter, acc1, acc3)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    model[0].eval()
    model[1].eval()
    mem_label_all = obtain_label(dset_loaders['target_tr_t'], model, args)
    mem_label = mem_label_all[0].cuda()
    mem_label2 = mem_label_all[1].cuda()

    model[0].train()
    model[1].train()

    while iter_num < max_iter + 1:

        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target_tr"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        outputs_test = model(inputs_test)

        pred = mem_label[tar_idx]
        classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
        classifier_loss *= 0.3

        pred = mem_label2[tar_idx]
        classifier_loss2 = nn.CrossEntropyLoss()(outputs_test, pred)
        classifier_loss2 *= 0.1
        classifier_loss += classifier_loss2

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

    model[0].eval()
    model[1].eval()
    acc1, acc3 = cal_acc(dset_loaders['target_te'], model)
    log_str = 'Iter:{}/{}; Accuracy = {:.2f}/ {:.2f}'.format(iter_num, max_iter, acc1, acc3)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')
    model[0].train()
    model[1].train()
    
    torch.save(model.state_dict(), osp.join(args.output_dir_src, "target.pt"))

    return model

def obtain_label(loader, model, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            inputs = inputs.cuda()
            outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    mem_label_all = torch.topk(all_output, 3)[1].T

    return mem_label_all

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=10, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='PLACE', choices=['PLACE'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='swinb', help="swinb, resnet50")
    parser.add_argument('--seed', type=int, default=3407, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="linear", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='ckps')
    parser.add_argument('--sampler', type=str, default="weight", choices=["random", "weight"])

    args = parser.parse_args()

    names = ['usa_train', 'asia_train', 'asia_test', 'test']
    if args.dset == 'OBJ':
        args.class_num = 600
    if args.dset == 'PLACE':
        args.class_num = 204
    if args.dset == 'UNIDA':
        args.class_num = 200

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = 'data_list/'
    args.s_dset_path = folder + args.dset + '/usa_train.txt'
    args.t_dset_path = folder + args.dset + '/asia_train.txt' 
    args.test_dset_path = folder + args.dset + '/asia_test.txt' 
    args.test2_dset_path = folder + args.dset + '/test.txt'     

    args.output_dir_src = osp.join(args.output, args.dset)
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log_source.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_source(args)

    args.out_file = open(osp.join(args.output_dir_src, 'source_test.txt'), 'w')
    args.modelpath = args.output_dir_src + '/source.pt'
    test_target(args)

    args.out_file = open(osp.join(args.output_dir_src, 'log_target.txt'), 'w')
    args.modelpath = args.output_dir_src + '/source.pt'
    train_target(args)

    args.out_file = open(osp.join(args.output_dir_src, 'target_test.txt'), 'w')
    args.modelpath = args.output_dir_src + '/target.pt'  
    test_target(args)
