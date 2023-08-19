import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import datasets
from enum import Enum
import math

## ImageNet folder: Please replace with your own imagenet validation path
val_dir = "/usr/data/imagenet/raw/val"

## Hyper parameters
# Testbed noise features. We calibrate the system to make the noise mean to be 0.
weight_testbed_data_mean = 0
weight_testbed_data_stdev = 1.6569279459762971
inter_layer_testbed_data_mean = 0
inter_layer_testbed_data_stdev = 1.6569279459762971
input_testbed_data_mean = 0
input_testbed_data_stdev = 1.6569279459762971

# Inference parameters
experiment_time = 10
device = torch.device("cuda")
batch_size = 512
num_to_run = 10000

# Model definition of AlexNet 32-bit
class AlexNet_32bit(torch.nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super(AlexNet_32bit, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv1_nonlinear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv2_nonlinear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)           
        )
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv3_nonlinear = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv4_nonlinear = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5_nonlinear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc1_nonlinear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        self.fc2 = nn.Linear(4096, 4096)
        self.fc2_nonlinear = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_nonlinear(x)
        x = self.conv2(x)
        x = self.conv2_nonlinear(x)
        x = self.conv3(x)
        x = self.conv3_nonlinear(x)
        x = self.conv4(x)
        x = self.conv4_nonlinear(x)
        x = self.conv5(x)
        x = self.conv5_nonlinear(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.fc1_nonlinear(x)
        x = self.fc2(x)
        x = self.fc2_nonlinear(x)
        x = self.fc3(x)

        return x

# Model definition of AlexNet 8-bit
class AlexNet_8bit(torch.nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super(AlexNet_8bit, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv1_nonlinear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv2_nonlinear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)           
        )
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv3_nonlinear = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv4_nonlinear = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5_nonlinear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc1_nonlinear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        self.fc2 = nn.Linear(4096, 4096)
        self.fc2_nonlinear = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, num_classes)
    
    def adjust_conv_bias(self, x, scaling_factor, bias):
        expanded_bias = torch.reshape(bias, (1, bias.size(dim=0), 1, 1))
        output = x + (scaling_factor - 1) * expanded_bias
        return output

    def adjust_fc_bias(self, x, scaling_factor, bias):
        expanded_bias = torch.reshape(bias, (1, bias.size(dim=0)))
        output = x + (scaling_factor - 1) * expanded_bias
        return output

    def convert_float8_input(self, x):
        scaling_factor = 255/torch.max(torch.abs(x))
        output = torch.round(x * scaling_factor)
        return scaling_factor, output
    
    def convert_float8_inter_layer(self, x):
        scaling_factor = 255/torch.max(torch.abs(x))
        output = torch.round(x * scaling_factor)
        return scaling_factor, output
    
    def forward(self, x):
        scaling_factor, x = self.convert_float8_input(x)
        x = self.conv1(x)
        x = self.adjust_conv_bias(x, scaling_factor, self.conv1.bias)
        scaling_factor, x = self.convert_float8_inter_layer(x)
        x = self.conv1_nonlinear(x)
        x = self.conv2(x)
        x = self.adjust_conv_bias(x, scaling_factor, self.conv2.bias)
        scaling_factor, x = self.convert_float8_inter_layer(x)
        x = self.conv2_nonlinear(x)
        x = self.conv3(x)
        x = self.adjust_conv_bias(x, scaling_factor, self.conv3.bias)
        scaling_factor, x = self.convert_float8_inter_layer(x)
        x = self.conv3_nonlinear(x)
        x = self.conv4(x)
        x = self.adjust_conv_bias(x, scaling_factor, self.conv4.bias)
        scaling_factor, x = self.convert_float8_inter_layer(x)
        x = self.conv4_nonlinear(x)
        x = self.conv5(x)
        x = self.adjust_conv_bias(x, scaling_factor, self.conv5.bias)
        _, x = self.convert_float8_inter_layer(x)
        x = self.conv5_nonlinear(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        scaling_factor, x = self.convert_float8_inter_layer(x)

        x = self.fc1(x)
        x = self.adjust_fc_bias(x, scaling_factor, self.fc1.bias)
        scaling_factor, x = self.convert_float8_inter_layer(x)
        x = self.fc1_nonlinear(x)
        x = self.fc2(x)
        x = self.adjust_fc_bias(x, scaling_factor, self.fc2.bias)
        scaling_factor, x = self.convert_float8_inter_layer(x)
        x = self.fc2_nonlinear(x)
        x = self.fc3(x)
        x = self.adjust_fc_bias(x, scaling_factor, self.fc3.bias)
        scaling_factor, x = self.convert_float8_inter_layer(x)

        return x

# Model definition of AlexNet 8-bit with noise
class AlexNet_8bit_noise(torch.nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super(AlexNet_8bit_noise, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv1_nonlinear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv2_nonlinear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)           
        )
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv3_nonlinear = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv4_nonlinear = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5_nonlinear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc1_nonlinear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        self.fc2 = nn.Linear(4096, 4096)
        self.fc2_nonlinear = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, num_classes)
    
    def adjust_conv_bias(self, x, scaling_factor, bias):
        expanded_bias = torch.reshape(bias, (1, bias.size(dim=0), 1, 1))
        output = x + (scaling_factor - 1) * expanded_bias
        return output

    def adjust_fc_bias(self, x, scaling_factor, bias):
        expanded_bias = torch.reshape(bias, (1, bias.size(dim=0)))
        output = x + (scaling_factor - 1) * expanded_bias
        return output
    
    def add_noise(self, x):
        x_sign = torch.sign(x)
        noise = torch.normal(inter_layer_testbed_data_mean, inter_layer_testbed_data_stdev, size = x.size()).to(device)
        x_noise = torch.abs(torch.abs(x) + noise)
        output = torch.mul(x_sign, x_noise)
        return output

    def convert_float8_input(self, x):
        scaling_factor = 255/torch.max(torch.abs(x))
        output = torch.round(x * scaling_factor)
        return scaling_factor, output
    
    def convert_float8_inter_layer(self, x):
        scaling_factor = 255/torch.max(torch.abs(x))
        output = torch.round(x * scaling_factor)
        return scaling_factor, output
    
    def forward(self, x):
        scaling_factor, x = self.convert_float8_input(x)
        x = self.conv1(x)
        x = self.adjust_conv_bias(x, scaling_factor, self.conv1.bias)
        scaling_factor, x = self.convert_float8_inter_layer(x)
        x = self.add_noise(x)
        x = self.conv1_nonlinear(x)
        x = torch.round(x)
        x = self.conv2(x)
        x = self.adjust_conv_bias(x, scaling_factor, self.conv2.bias)
        scaling_factor, x = self.convert_float8_inter_layer(x)
        x = self.add_noise(x)
        x = self.conv2_nonlinear(x)
        x = torch.round(x)
        x = self.conv3(x)
        x = self.adjust_conv_bias(x, scaling_factor, self.conv3.bias)
        scaling_factor, x = self.convert_float8_inter_layer(x)
        x = self.add_noise(x)
        x = self.conv3_nonlinear(x)
        x = torch.round(x)
        x = self.conv4(x)
        x = self.adjust_conv_bias(x, scaling_factor, self.conv4.bias)
        scaling_factor, x = self.convert_float8_inter_layer(x)
        x = self.add_noise(x)
        x = self.conv4_nonlinear(x)
        x = torch.round(x)
        x = self.conv5(x)
        x = self.adjust_conv_bias(x, scaling_factor, self.conv5.bias)
        _, x = self.convert_float8_inter_layer(x)
        x = self.add_noise(x)
        x = self.conv5_nonlinear(x)
        x = torch.round(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        scaling_factor, x = self.convert_float8_inter_layer(x)

        x = self.fc1(x)
        x = self.adjust_fc_bias(x, scaling_factor, self.fc1.bias)
        scaling_factor, x = self.convert_float8_inter_layer(x)
        x = self.add_noise(x)
        x = self.fc1_nonlinear(x)
        x = torch.round(x)
        x = self.fc2(x)
        x = self.adjust_fc_bias(x, scaling_factor, self.fc2.bias)
        scaling_factor, x = self.convert_float8_inter_layer(x)
        x = self.add_noise(x)
        x = self.fc2_nonlinear(x)
        x = torch.round(x)
        x = self.fc3(x)
        x = self.adjust_fc_bias(x, scaling_factor, self.fc3.bias)
        scaling_factor, x = self.convert_float8_inter_layer(x)
        x = self.add_noise(x)
        x = torch.round(x)

        return x

# Load model parameter for AlexNet 32-bit model
def load_alexnet_model_32bit():
    model = AlexNet_32bit()
    original_model = torch.load("models_weight/alexnet.pth")
    model_32bit = models.alexnet()
    model_32bit.load_state_dict(original_model)
    
    model.conv1.weight = model_32bit.features[0].weight
    model.conv1.bias = model_32bit.features[0].bias

    model.conv2.weight = model_32bit.features[3].weight
    model.conv2.bias = model_32bit.features[3].bias

    model.conv3.weight = model_32bit.features[6].weight
    model.conv3.bias = model_32bit.features[6].bias

    model.conv4.weight = model_32bit.features[8].weight
    model.conv4.bias = model_32bit.features[8].bias

    model.conv5.weight = model_32bit.features[10].weight
    model.conv5.bias = model_32bit.features[10].bias

    model.fc1.weight = model_32bit.classifier[1].weight
    model.fc1.bias = model_32bit.classifier[1].bias

    model.fc2.weight = model_32bit.classifier[4].weight
    model.fc2.bias = model_32bit.classifier[4].bias

    model.fc3.weight = model_32bit.classifier[6].weight
    model.fc3.bias = model_32bit.classifier[6].bias

    return model

# helper function for loading 8-bit model weights
def convert_float8_weight(input_tensor, scaling_factor):
    output_tensor = torch.round(input_tensor * scaling_factor)
    return nn.Parameter(output_tensor)

# helper function for loading 8-bit model biases
def convert_float8_bias(input_tensor, scaling_factor):
    output_tensor = torch.round(input_tensor * scaling_factor)
    return nn.Parameter(output_tensor)

# helper function for finding the scaling factor of 8-bit model        
def find_weight_scaling_factor(model_path, model_32bit):
    original_model = torch.load(model_path)
    model_32bit.load_state_dict(original_model)
    max = 0
    for _, v in model_32bit.state_dict().items():
        cur_max = torch.max(torch.abs(v)).numpy()
        if cur_max > max:
            max = cur_max
    scaling_factor = 255/max
    return scaling_factor

# Load model parameter for AlexNet 8-bit model
def load_alexnet_model_8bit():
    model_path = "models_weight/alexnet.pth"
    model = AlexNet_8bit()
    original_model = torch.load(model_path)
    model_32bit = models.alexnet()
    model_32bit.load_state_dict(original_model)
    scaling_factor = find_weight_scaling_factor(model_path, model_32bit)

    model.conv1.weight = convert_float8_weight(model_32bit.features[0].weight, scaling_factor)
    model.conv1.bias = convert_float8_bias(model_32bit.features[0].bias, scaling_factor)

    model.conv2.weight = convert_float8_weight(model_32bit.features[3].weight, scaling_factor)
    model.conv2.bias = convert_float8_bias(model_32bit.features[3].bias, scaling_factor)

    model.conv3.weight = convert_float8_weight(model_32bit.features[6].weight, scaling_factor)
    model.conv3.bias = convert_float8_bias(model_32bit.features[6].bias, scaling_factor)

    model.conv4.weight = convert_float8_weight(model_32bit.features[8].weight, scaling_factor)
    model.conv4.bias = convert_float8_bias(model_32bit.features[8].bias, scaling_factor)

    model.conv5.weight = convert_float8_weight(model_32bit.features[10].weight, scaling_factor)
    model.conv5.bias = convert_float8_bias(model_32bit.features[10].bias, scaling_factor)

    model.fc1.weight = convert_float8_weight(model_32bit.classifier[1].weight, scaling_factor)
    model.fc1.bias = convert_float8_bias(model_32bit.classifier[1].bias, scaling_factor)

    model.fc2.weight = convert_float8_weight(model_32bit.classifier[4].weight, scaling_factor)
    model.fc2.bias = convert_float8_bias(model_32bit.classifier[4].bias, scaling_factor)

    model.fc3.weight = convert_float8_weight(model_32bit.classifier[6].weight, scaling_factor)
    model.fc3.bias = convert_float8_bias(model_32bit.classifier[6].bias, scaling_factor)

    return model

# Load model parameter for AlexNet 8-bit model with noise
def load_alexnet_model_8bit_noise():
    model_path = "models_weight/alexnet.pth"
    model = AlexNet_8bit_noise()
    original_model = torch.load(model_path)
    model_32bit = models.alexnet()
    model_32bit.load_state_dict(original_model)
    scaling_factor = find_weight_scaling_factor(model_path, model_32bit)

    model.conv1.weight = convert_float8_weight(model_32bit.features[0].weight, scaling_factor)
    model.conv1.bias = convert_float8_bias(model_32bit.features[0].bias, scaling_factor)

    model.conv2.weight = convert_float8_weight(model_32bit.features[3].weight, scaling_factor)
    model.conv2.bias = convert_float8_bias(model_32bit.features[3].bias, scaling_factor)

    model.conv3.weight = convert_float8_weight(model_32bit.features[6].weight, scaling_factor)
    model.conv3.bias = convert_float8_bias(model_32bit.features[6].bias, scaling_factor)

    model.conv4.weight = convert_float8_weight(model_32bit.features[8].weight, scaling_factor)
    model.conv4.bias = convert_float8_bias(model_32bit.features[8].bias, scaling_factor)

    model.conv5.weight = convert_float8_weight(model_32bit.features[10].weight, scaling_factor)
    model.conv5.bias = convert_float8_bias(model_32bit.features[10].bias, scaling_factor)

    model.fc1.weight = convert_float8_weight(model_32bit.classifier[1].weight, scaling_factor)
    model.fc1.bias = convert_float8_bias(model_32bit.classifier[1].bias, scaling_factor)

    model.fc2.weight = convert_float8_weight(model_32bit.classifier[4].weight, scaling_factor)
    model.fc2.bias = convert_float8_bias(model_32bit.classifier[4].bias, scaling_factor)

    model.fc3.weight = convert_float8_weight(model_32bit.classifier[6].weight, scaling_factor)
    model.fc3.bias = convert_float8_bias(model_32bit.classifier[6].bias, scaling_factor)

    return model

# Load imagenet dataset
def load_imagenet():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(val_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False)
    return val_loader

# helper class for reporting validation progress
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

# helper class for reporting validation progress
class AverageMeter(object):
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

# helper class for reporting validation progress
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# helper function for collecting validation accuracy
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
# Function for running inference on ImageNet validation sets
def validate(val_loader, model, device, batch_to_run):
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    # switch to evaluate mode
    model.eval()

    print("start validate.")
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if i >= batch_to_run:
                break
            images, target = images.to(device), target.to(device)
            # compute output
            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            print(i, "/", batch_to_run, " top1_avg: ", top1.avg, " top5_avg: ", top5.avg)
    
    return top1.avg, top5.avg

# function for creating the result file
def create_file(file_name):
    f = open(file_name, "w")
    f.close()

if __name__ == '__main__':
    print("Start device warmup...")
    for i in range(10):
        torch.normal(weight_testbed_data_mean, weight_testbed_data_stdev, size = [64, 3, 11, 11])

    print("Start model loading...")

    print("Emulate AlexNet 32-bit...")
    file_name = "results/alexnet_digital_32bit.txt"
    create_file(file_name)
    for i in range(experiment_time):
        model = load_alexnet_model_32bit().to(device)

        print("Start validating...", i)
        val_loader = load_imagenet()
        top1_acc, top5_acc = validate(val_loader, model, device, math.ceil(num_to_run/batch_size))
        print("Final result - top1: ", top1_acc, " top5: ", top5_acc)
    
        f = open(file_name, "a")
        line = str(top1_acc) + " " + str(top5_acc) + " \n"
        f.write(line)
        f.close()

    print("Emulate AlexNet 8-bit...")
    file_name = "results/alexnet_digital_8bit.txt"
    create_file(file_name)
    for i in range(experiment_time):
        model = load_alexnet_model_8bit().to(device)

        print("Start validating...", i)
        val_loader = load_imagenet()
        top1_acc, top5_acc = validate(val_loader, model, device, math.ceil(num_to_run/batch_size))
        print("Final result - top1: ", top1_acc, " top5: ", top5_acc)
    
        f = open(file_name, "a")
        line = str(top1_acc) + " " + str(top5_acc) + " \n"
        f.write(line)
        f.close()

    print("Emulate AlexNet 8-bit with photonic noise...")
    file_name = "results/alexnet_lightning.txt"
    create_file(file_name)
    for i in range(experiment_time):
        model = load_alexnet_model_8bit_noise().to(device)

        print("Start validating...", i)
        val_loader = load_imagenet()
        top1_acc, top5_acc = validate(val_loader, model, device, math.ceil(num_to_run/batch_size))
        print("Final result - top1: ", top1_acc, " top5: ", top5_acc)
    
        f = open(file_name, "a")
        line = str(top1_acc) + " " + str(top5_acc) + " \n"
        f.write(line)
        f.close()