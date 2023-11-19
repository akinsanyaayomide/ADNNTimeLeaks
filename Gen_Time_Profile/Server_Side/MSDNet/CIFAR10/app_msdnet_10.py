from flask import Flask, render_template, request
import torch
import numpy as np
from Msdnet import MSDNet

from torchvision.transforms import ToTensor
#from models.Ranet import RANet
#from PIL import Image
import argparse
import os
import sys

host = sys.argv[1]
app = Flask(__name__)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
args = argparse.ArgumentParser(description="Early Exit CLI")
args.nBlocks = 3
args.base = 4
args.step = 2
args.stepmode ='even'
args.compress_factor = 0.25
args.nChannels = 16
args.data = 'cifar10'
args.growthRate = 6
args.block_step = 2

args.prune = 'max'
args.bottleneck =True

args.grFactor = '1-2-4'
args.bnFactor = '1-2-4'
#args.scale_list = '1-2-3'

args.reduction = 0.5

args.use_valid = True

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
#args.scale_list = list(map(int, args.scale_list.split('-')))
args.nScales = len(args.grFactor)
# print(args.grFactor)
if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

if args.data == 'cifar10':
    args.num_classes = 10
elif args.data == 'cifar100':
    args.num_classes = 100
else:
    args.num_classes = 2

model = MSDNet(args)

path = os.getcwd()
path = path.split('/')
root_dir_index = path.index('ADNNTimeLeaks')

data_path = path[:root_dir_index+1]+['CheckPoints','MSDNet','CIFAR10']
data_path = '/'.join(data_path)

model.to(device)
model_path = os.path.join(data_path,'msdnet_cifar10.pth.tar')

try:
    checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
except:
    new_keys = []
    for key in checkpoint['state_dict'].keys():
        new_key = key.split(".")
        if new_key[0] == 'module':
            new_key = ".".join(new_key[1:])
        else:
            new_key = ".".join(new_key[0:])
        new_keys.append(new_key)

    checkpoint['state_dict'] = dict(zip(new_keys, checkpoint['state_dict'].values()))
threshold =  [ 9.9792e-01,  8.7337e-01, -1.0000e+08]


checkpoint = checkpoint
model.load_state_dict(checkpoint['state_dict'])
model.exit_threshold =torch.tensor([threshold], dtype=torch.float32)

model.eval()

#routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Please subscribe  Artificial Intelligence Hub..!!!"


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        x = request.files['x']
        x = torch.load(x)
        x = torch.unsqueeze(x, dim=0)
        x = x.to(device)
        pred,exit_ = model(x)
        p = pred.argmax(dim=1).item()
        label = classes[p]

        #img_path = "static/" + img.filename	
        #img.save(img_path)

        #p = predict_label(img_path)
        new_page = render_template("index.html", prediction = label) + str(exit_)
    return new_page

if __name__ =='__main__':
    #app.debug = True
    #app.run(debug = True)
    app.run(debug = True,host=host )
