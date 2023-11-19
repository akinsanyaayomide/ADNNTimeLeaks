from flask import Flask, render_template, request
import torch
import torch.nn as nn
import argparse
from collections import Counter
import utils
import os
import sys
#from PIL import Image

host = sys.argv[1]
app = Flask(__name__)

args = argparse.ArgumentParser()
args.model = 'R110_C10'

path = os.getcwd()
path = path.split('/')
root_dir_index = path.index('ADNNTimeLeaks')

data_path = path[:root_dir_index+1]+['CheckPoints','BlockDrop','CIFAR10']
data_path = '/'.join(data_path)

model_path = os.path.join(data_path,'ckpt_E_2000_A_0.936_R_1.95E-01_S_16.93_#_469.t7')
args.load = model_path

rnet, agent = utils.get_model(args.model)
classes =  ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


dic = {x:y for x,y in zip(range(10),classes)}

if args.load is not None:
    utils.load_checkpoint(rnet, agent, args.load)

rnet.eval()
agent.eval()



storage = {x:[] for x in classes}

def Convert(string):
    list1 = []
    list1[:0] = string
    list1 = [int(x) for x in list1]
    return list1

def count_no_blocks(policy):
    a = policy.tolist()
    #print(a)
    a = [str(int(x)) for x in a[0]]
    a = ''.join(a)
    x = Convert(a)
    b = Counter(x)
    no = b[1]
    return no,a


def predict_label(x):
    x = torch.unsqueeze(x, dim=0)
    probs, _ = agent(x)
    policy = probs.clone()
    policy[policy<0.5] = int(0)
    policy[policy>=0.5] = int(1)


    preds = rnet.forward_single(x, policy.data.squeeze(0))
    _ , pred_idx = preds.max(1)
    y_pred = pred_idx.item()
    label = classes[y]
    return label

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
        probs, _ = agent(x)
        policy = probs.clone()
        policy[policy<0.5] = int(0)
        policy[policy>=0.5] = int(1)


        preds = rnet.forward_single(x, policy.data.squeeze(0))
        _ , pred_idx = preds.max(1)
        y_pred = pred_idx.item()
        label = classes[y_pred]
        exit_,_ = count_no_blocks(policy)


        #img_path = "static/" + img.filename	
        #img.save(img_path)

        #p = predict_label(img_path)
        new_page = render_template("index.html", prediction = label) + " "+ label + " "+ str(exit_)
    return new_page




if __name__ =='__main__':
    #app.debug = True
    #app.run(debug = True)
    app.run(debug = True,host=host )
    #app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))