import torch


import network_architectures as arcs

import os
from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import sys

#from models.Branchynet import ConvPoolAc,B_Lenet,B_Lenet_fcn,B_AlexNet

from PIL import Image
host = sys.argv[1]
app = Flask(__name__)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

path = os.getcwd()
path = path.split('/')
root_dir_index = path.index('ADNNTimeLeaks')

data_path = path[:root_dir_index+1]+['CheckPoints','SDNet','CANCER']
data_path = '/'.join(data_path)
sdn_model, sdn_params = arcs.load_model(
                        data_path,'test',epoch=-1)

confidence_thresholds = [0.9] # set for the confidence threshold for early exits
sdn_model.forward = sdn_model.early_exit
sdn_model.confidence_threshold = confidence_thresholds[0]

sdn_model.eval()

classes = ['Benign','Malignant']


tfs_1 = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def predict_label(img_path):
    i = Image.open(img_path)
    i = tfs_1(i).float()
    i = torch.unsqueeze(i, dim=0)
    p = sdn_model(i)
    p = p.argmax(dim=1).item()
    return dic[p]

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
            pred,exit_ ,_= sdn_model(x)
        p = pred.argmax(dim=1).item()
        label = classes[p]

        #img_path = "static/" + img.filename	
        #img.save(img_path)

        #p = predict_label(img_path)
        new_page = render_template("index.html", prediction = p) + str(exit_)
    return new_page




if __name__ =='__main__':
    #app.debug = True
    #app.run(debug = True)
    app.run(debug = True,host=host )
    #app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))