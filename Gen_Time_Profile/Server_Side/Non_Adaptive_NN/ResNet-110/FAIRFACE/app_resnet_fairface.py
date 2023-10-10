from flask import Flask, render_template, request
import resnet
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

app = Flask(__name__)

host = sys.argv[1]


classes = ['0-19','20-49','more than 50']



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = resnet.ResNet(resnet.BasicBlock, [18, 18, 18])
model.to(device)

path = os.getcwd()
path = path.split('/')
root_dir_index = path.index('ADNNTimeLeaks')

data_path = path[:root_dir_index+1]+['CheckPoints','Non_Adaptive_NN','ResNet-110','FAIRFACE']
data_path = '/'.join(data_path)

model_path = os.path.join(data_path,'resnet_110_fairface.th')

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
    #     print(key)

    checkpoint['state_dict'] = dict(zip(new_keys, checkpoint['state_dict'].values()))

model.load_state_dict(checkpoint['state_dict'])


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
        #x = torch.unsqueeze(x, dim=0)
        pred = model(x)
        p = pred.argmax(dim=1).item()
        label = classes[p]
        exit_ = 1

        #img_path = "static/" + img.filename	
        #img.save(img_path)

        #p = predict_label(img_path)
        new_page = render_template("index.html", prediction = label) + str(exit_)
    return new_page




if __name__ =='__main__':
    #app.debug = True
    #app.run(debug = True)
    app.run(debug = True,host=host )
