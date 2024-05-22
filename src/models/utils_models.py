from torch import nn
import torch.nn.init as init
import torch

import torchvision
import timm
import copy
import sys
#sys.path.append('/home/srip23/FedMH/src')   
from src.models import *

def get_models_fedmh(num_users, model, dataset, args):

    users_model = []

    for i in range(-1, num_users):
        if model == "lenet5":
            if dataset in ("cifar10"):
                net = LeNet5(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif dataset in ("cifar100"):
                net = LeNet5(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=100)
        elif model == "resnet20":
            if dataset in ("cifar10"):
                net = resnet20(num_classes=10)
            elif dataset in ("cifar100"):
                net = resnet20(num_classes=100)
        elif model == "resnet32":
            if dataset in ("cifar10"):
                net = resnet32(num_classes=10)
            elif dataset in ("cifar100"):
                net = resnet32(num_classes=100)
        elif model == "resnet4":
            if dataset in ("cifar10"):
                net = ResNet4(BasicBlock, [1], scaling=1.0, num_classes=10)
            elif dataset in ("cifar100"):
                net = ResNet4(BasicBlock, [1], scaling=1.0, num_classes=100)
        elif model == "resnet6":
            if dataset in ("cifar10"):
                net = ResNet6(BasicBlock, [1,1], scaling=1.0, num_classes=10)
            elif dataset in ("cifar100"):
                net = ResNet6(BasicBlock, [1,1], scaling=1.0, num_classes=100)
        elif model == "resnet8":
            if dataset in ("cifar10"):
                net = ResNet8(BasicBlock, [1,1,1], scaling=1.0, num_classes=10)
            elif dataset in ("cifar100"):
                net = ResNet8(BasicBlock, [1,1,1], scaling=1.0, num_classes=100)
            elif dataset in ("tinyimagenet"):
                net = ResNet8(BasicBlock, [1,1,1], scaling=1.0, num_classes=200)
        elif model == "resnet14-0.75":
            if dataset in ("cifar10"):
                net = ResNet(BasicBlock, [1,1,2,2], scaling=0.75, num_classes=10)
        elif model == "resnet14":
            if dataset in ("cifar10"):
                net = ResNet(BasicBlock, [1,2,2,1], scaling=1.0, num_classes=10)
            elif dataset in ("cifar100"):
                net = ResNet(BasicBlock, [1,2,2,1], scaling=1.0, num_classes=100)
            elif dataset in ("tinyimagenet"):
                net = ResNet(BasicBlock, [1,2,2,1], scaling=1.0, num_classes=200)
        elif model == "resnet18":
            if dataset in ("cifar10"):
                net = ResNet(BasicBlock, [2,2,2,2], scaling=1.0, num_classes=10)
            elif dataset in ("cifar100"):
                net = ResNet(BasicBlock, [2,2,2,2], scaling=1.0, num_classes=100)
            elif dataset in ("cinic10"):
                net = ResNet(BasicBlock, [2,2,2,2], scaling=1.0, num_classes=10)
            elif dataset in ("tinyimagenet"):
                net = ResNet(BasicBlock, [2,2,2,2], scaling=1.0, num_classes=200)
        elif model == "resnet34":
            if dataset in ("cifar10"):
                net = ResNet(BasicBlock, [3,4,6,3], num_classes=10)
            elif dataset in ("cifar100"):
                net = ResNet(BasicBlock, [3,4,6,3], num_classes=100)
            elif dataset in ("cinic10"):
                net = ResNet(BasicBlock, [3,4,6,3], num_classes=10)
        elif model == "resnet50":
            if dataset in ("cifar10"):
                net = ResNet(BottleneckBlock, [3,4,6,3], num_classes=10)
            elif dataset in ("cifar100"):
                net = ResNet(BottleneckBlock, [3,4,6,3], num_classes=100)
            elif dataset in ("cinic10"):
                net = ResNet(BottleneckBlock, [3,4,6,3], num_classes=10)
        elif model == "resnet101":
            if dataset in ("cifar10"):
                net = ResNet(BottleneckBlock, [3,4,23,3], num_classes=10)
            elif dataset in ("cifar100"):
                net = ResNet(BottleneckBlock, [3,4,23,3], num_classes=100)
        elif model == "vgg7":
            if dataset in ("cifar10"):
                net = vgg7_bn(num_classes=10)
            elif dataset in ("cifar100"):
                net = vgg7_bn(num_classes=100)
        elif model == "vgg7_256":
            if dataset in ("cifar10"):
                net = vgg7_bn_256(num_classes=10)
            elif dataset in ("cifar100"):
                net = vgg7_bn_256(num_classes=100)
        elif model == "vgg12":
            if dataset in ("cifar10"):
                net = vgg12_bn(num_classes=10)
            elif dataset in ("cifar100"):
                net = vgg12_bn(num_classes=100)
        elif model == "vgg11":
            if dataset in ("cifar10"):
                net = vgg11_bn(num_classes=10)
            elif dataset in ("cifar100"):
                net = vgg11_bn(num_classes=100)
        elif model == "vgg16":
            if dataset in ("cifar10"):
                net = vgg16_bn(num_classes=10)
            elif dataset in ("cifar100"):
                net = vgg16_bn(num_classes=100)
            elif dataset in ("tinyimagenet"):
                net = vgg16_bn(num_classes=200)
        elif model == "vit-s":
            if dataset in ("cifar10"):
                net = ViT(
                        image_size = 32,
                        patch_size = 4,
                        num_classes = 10,
                        dim = 64,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 256,
                        dropout = 0.1,
                        emb_dropout = 0.1
                        )
            elif dataset in ("cifar100"):
                net = ViT(
                        image_size = 32,
                        patch_size = 4,
                        num_classes = 100,
                        dim = 64,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 256,
                        dropout = 0.1,
                        emb_dropout = 0.1
                        )
            elif dataset in ("tinyimagenet"):
                net = ViT(
                        image_size = 32,
                        patch_size = 4,
                        num_classes = 200,
                        dim = 64,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 256,
                        dropout = 0.1,
                        emb_dropout = 0.1
                        )
        elif model == "vit-l":
            if dataset in ("cifar10"):
                net = ViT(
                        image_size = 32,
                        patch_size = 4,
                        num_classes = 10,
                        dim = 512,
                        depth = 6,
                        heads = 12,
                        mlp_dim = 1024,
                        dropout = 0.1,
                        emb_dropout = 0.1
                        )
        elif model == "squeezenet1_0":
            if dataset in ("cifar10"):
                net = torchvision.models.squeezenet1_0(pretrained=False, num_classes=10)
        elif model == "regnetx_002":
            if dataset in ("cifar10"):
                net = timm.create_model('regnetx_002', pretrained=False, num_classes=10)
        elif model == "shufflenetv2":
            if dataset == "cifar10":
                net = torchvision.models.shufflenet_v2_x0_5(weights=None, num_classes=10)
                num_ftrs = net.fc.in_features
                net.fc = torch.nn.Linear(num_ftrs, 10)
                torch.nn.init.xavier_uniform_(net.fc.weight)
                torch.nn.init.zeros_(net.fc.bias)               
            elif dataset == "cifar100":
                net = torchvision.models.shufflenet_v2_x0_5(weights=None, num_classes=100)
                num_ftrs = net.fc.in_features
                net.fc = torch.nn.Linear(num_ftrs, 100)
                torch.nn.init.xavier_uniform_(net.fc.weight)
                torch.nn.init.zeros_(net.fc.bias)
        elif model == "shufflenet_v2_x1_0":
            if dataset in ("cifar10"):
                net = torchvision.models.shufflenet_v2_x1_0(pretrained=False, num_classes=10)
        elif model == "densenet121":
            if dataset in ("cifar10"):
                net = torchvision.models.densenet121(pretrained=False, num_classes=10)
        elif model == "efficientnet-b3":
            if dataset in ("cifar10"):
                net = torchvision.models.efficientnet_b3(pretrained=False, num_classes=10)
        elif model == "mobilenet_v2":
            if dataset in ("cifar10"):
                net = torchvision.models.mobilenet_v2(pretrained=False, num_classes=10)
        elif model == "edgenext_xx_small":
            if dataset in ("cifar10"):
                net = timm.create_model('edgenext_xx_small', pretrained=False, num_classes=10)
        elif model == "edgenext_x_small":
            if dataset in ("cifar10"):
                net = timm.create_model('edgenext_x_small', pretrained=False, num_classes=10)
        elif model == "lenet5":
            if dataset in ("cifar10"):
                net = LeNet5(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
        elif model == "resnet10_s":
            if dataset in ("cifar10"):
                net = ResNet10_s(num_classes=10, adaptive_pool=True)
            elif dataset in ("cifar100"):
                net = ResNet10_s(num_classes=100, adaptive_pool=True)
            elif dataset in ("cinic10"):
                net = ResNet10_s(num_classes=10, adaptive_pool=True)
        elif model == "resnet10_xs":
            if dataset in ("cifar10"):
                net = ResNet10_xs(num_classes=10, adaptive_pool=True)
            elif dataset in ("cifar100"):
                net = ResNet10_xs(num_classes=100, adaptive_pool=True)
            elif dataset in ("cinic10"):
                net = ResNet10_xs(num_classes=10, adaptive_pool=True)
        elif model == "resnet10_xxs":
            if dataset in ("cifar10"):
                net = ResNet10_xxs(num_classes=10, adaptive_pool=True)
            elif dataset in ("cifar100"):
                net = ResNet10_xxs(num_classes=100, adaptive_pool=True)
            elif dataset in ("cinic10"):
                net = ResNet10_xxs(num_classes=10, adaptive_pool=True)
        elif model == "resnet10_m":
            if dataset in ("cifar10"):
                net = ResNet10_m(num_classes=10, adaptive_pool=True)
            elif dataset in ("cifar100"):
                net = ResNet10_m(num_classes=100, adaptive_pool=True)
            elif dataset in ("cinic10"):
                net = ResNet10_m(num_classes=10, adaptive_pool=True)
        elif model == "resnet10_l":
            if dataset in ("cifar10"):
                net = ResNet10_l(num_classes=10, adaptive_pool=True)
            elif dataset in ("cifar100"):
                net = ResNet10_l(num_classes=100, adaptive_pool=True)
            elif dataset in ("cinic10"):
                net = ResNet10_l(num_classes=10, adaptive_pool=True)
        elif model == "resnet10":
            if dataset in ("cifar10"):
                net = ResNet10(num_classes=10, adaptive_pool=True)
            elif dataset in ("cifar100"):
                net = ResNet10(num_classes=100, adaptive_pool=True)
            elif dataset in ("cinic10"):
                net = ResNet10(num_classes=10, adaptive_pool=True)
        elif model == "mobilenetv3_small_050":
            if dataset in ("tinyimagenet"):
                net = timm.create_model(model, pretrained=True, num_classes=200)
        elif model == "mobilevit_s":
            if dataset in ("tinyimagenet"):
                net = timm.create_model(model, pretrained=True, num_classes=200)
        elif model == "mobilenetv3_large_100":
            if dataset in ("tinyimagenet"):
                net = timm.create_model(model, pretrained=True, num_classes=200)
        elif model == "mobilevitv2_175":
            if dataset in ("tinyimagenet"):
                net = timm.create_model(model, pretrained=True, num_classes=200)
        elif model == "resnet18":
            if dataset in ("tinyimagenet"):
                net = timm.create_model(model, pretrained=True, num_classes=200)
        elif model == "resnet34_p":
            if dataset in ("tinyimagenet"):
                net = timm.create_model('resnet34', pretrained=True, num_classes=200)
        elif model == "mobilenet_v3_large":
            if dataset in ("tinyimagenet"):
                net = torchvision.models.mobilenet_v3_large(pretrained=True)#, num_classes=200)
                final_layer_name = "classifier"  # Hypothetical name, replace it with the correct name
                num_ftrs = getattr(net, final_layer_name).in_features
                setattr(net, final_layer_name, torch.nn.Linear(num_ftrs, 200))
                torch.nn.init.xavier_uniform_(getattr(net, final_layer_name).weight)
                torch.nn.init.zeros_(getattr(net, final_layer_name).bias)
        else:
            print("not supported yet")
            sys.exit()
        
        if i == -1:
            net_glob = copy.deepcopy(net)
            net_glob.apply(weight_init)
            initial_state_dict = copy.deepcopy(net_glob.state_dict())
            if args.load_initial:
                initial_state_dict = torch.load(args.load_initial)
                net_glob.load_state_dict(initial_state_dict)
        else:
            users_model.append(copy.deepcopy(net))
            users_model[i].load_state_dict(initial_state_dict)

    return users_model, net_glob, initial_state_dict


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias != None:
            init.normal_(m.bias.data)
    
    return 

def create_vit():
    pass

if __name__ == '__main__': 

    class Args: 
        def __init__(self): 
            self.load_initial = False
    
    args = Args()
    model = get_models_fedmh(1,'resnet50','cinic10',args)[0][0]

    num_params = 0
    for p in model.parameters(): 
        if p.requires_grad:
            num_params += p.numel()
    
    print(num_params)