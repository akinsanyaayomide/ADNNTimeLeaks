from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torchvision.transforms import ToTensor
import os
import Branchynet
import sys

#from models.Branchynet import ConvPoolAc,B_Lenet,B_Lenet_fcn,B_AlexNet

from PIL import Image

host = sys.argv[1]

app = Flask(__name__)

threshold = [1.0e-02, 5.0e-02, 1.0000e+04]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Branchynet.B_AlexNet()
model.to(device)
path = os.getcwd()
path = path.split('/')
root_dir_index = path.index('ADNNTimeLeaks')

data_path = path[:root_dir_index+1]+['CheckPoints','Non_Adaptive_NN','AlexNet','FAIRFACE']
data_path = '/'.join(data_path)
model_path = os.path.join(data_path,'alexnet_fairface.pth')
checkpoint_2 = torch.load(model_path,map_location=torch.device('cpu'))
model.load_state_dict(checkpoint_2['model_state_dict'])
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
        x = x.to(device)
        #x = torch.unsqueeze(x, dim=0)
        pred,exit_ = model(x)
        p = pred.argmax(dim=1).item()
        #label = classes[p]

        #img_path = "static/" + img.filename	
        #img.save(img_path)

        #p = predict_label(img_path)
        new_page = render_template("index.html", prediction = p) + str(exit_)
    return new_page




if __name__ =='__main__':
    #app.debug = True
    #app.run(debug = True)
    app.run(debug = True,host=host )
