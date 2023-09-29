import torch
import torch.nn as nn
import os
import sys

import network_architectures as arcs


from flask import Flask, render_template, request


from PIL import Image
host = sys.argv[1]
app = Flask(__name__)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

path = os.getcwd()
path = path.split('/')
root_dir_index = path.index('ADNNTimeLeaks')

data_path = path[:root_dir_index+1]+['CheckPoints','SDNet','FAIRFACE']
data_path = '/'.join(data_path)

sdn_model, sdn_params = arcs.load_model(
                        data_path,'test',epoch=-1)

confidence_thresholds = [0.99] # set for the confidence threshold for early exits
sdn_model.forward = sdn_model.early_exit
sdn_model.confidence_threshold = confidence_thresholds[0]

sdn_model.eval()

classes = ['0-19', '20-49', 'more than 50']


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

