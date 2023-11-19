from flask import Flask, render_template, request

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
import argparse
import models
from data import *
import sys

host = sys.argv[1]

app = Flask(__name__)

def convert(masks):
    res =0
    for i in range(53):
        res += masks[i].item()
        
    
    return res

args = argparse.ArgumentParser()

path = os.getcwd()
path = path.split('/')
root_dir_index = path.index('ADNNTimeLeaks')

data_path = path[:root_dir_index+1]+['CheckPoints','SkipNet','CANCER']
data_path = '/'.join(data_path)

model_path = os.path.join(data_path,'skipnet-cancer.pth.tar')

args.model = 'cifar10_rnn_gate_110'
args.dataset = 'cifar10'
args.resume = model_path
args.cmd = 'test'
args.arch = 'cifar10_rnn_gate_110'
args.batch_size = 1
args.pretrained = ('pretrained','store_true')

model = models.__dict__[args.arch](args.pretrained)
model = torch.nn.DataParallel(model)

if args.resume:
    if os.path.isfile(args.resume):
        print('=> loading checkpoint `{}`'.format(args.resume))
        checkpoint = torch.load(args.resume,map_location=torch.device('cpu'))
        args.start_iter = checkpoint['iter']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print('=> loaded checkpoint `{}` (iter: {})'.format(
            args.resume, checkpoint['iter']
        ))
    else:
        print('=> no checkpoint found at `{}`'.format(args.resume))


classes = ['Benign','Malignant']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
        x = x.to(device)
        #x = torch.unsqueeze(x, dim=0)
        with torch.no_grad():
            x = Variable(x)
            pred,masks,_ = model(x)
        exit_ = int(convert(masks))

        p = pred.argmax(dim=1).item()
        label = classes[p]

        #img_path = "static/" + img.filename	
        #img.save(img_path)

        #p = predict_label(img_path)
        new_page = render_template("index.html", prediction = label) + str(exit_)
    return new_page




if __name__ =='__main__':
    #app.debug = True
   # app.run(debug = True)
    app.run(debug = True,host=host )
    #app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
