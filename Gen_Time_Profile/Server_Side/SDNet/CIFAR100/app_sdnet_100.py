import torch


import network_architectures as arcs

from profiler import profile_sdn, profile
from flask import Flask, render_template, request
import torchvision
import os
import sys
from torchvision.transforms import ToTensor

host = sys.argv[1]
app = Flask(__name__)

data_test = torchvision.datasets.CIFAR100('./data_',
        download=True,train=False,transform=ToTensor())

new_dict = {'aquatic_mammals':['beaver', 'dolphin', 'otter', 'seal', 'whale'],
           'fish':['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
           'flowers':['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
           'food_containers':['bottle', 'bowl', 'can', 'cup', 'plate'],
           'fruit_and_vegetables':['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
           'household_electrical_devices':['clock', 'keyboard', 'lamp', 'telephone', 'television'],
           'household_furniture':['bed', 'chair', 'couch', 'table', 'wardrobe'],
           'insects':['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
           'large_carnivores':['bear', 'leopard', 'lion', 'tiger', 'wolf'],
           'large_man_made_outdoor_things':['bridge', 'castle', 'house', 'road', 'skyscraper'],
           'large_natural_outdoor_scenes':['cloud', 'forest', 'mountain', 'plain', 'sea'],
            'large_omnivores_and_herbivores':['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
            'medium_sized_mammals':['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
            'non_insect_invertebrates':['crab', 'lobster', 'snail', 'spider', 'worm'],
            'people':['baby', 'boy', 'girl', 'man', 'woman'],
           'reptiles':['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
            'small_mammals':['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
            'trees':['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
            'vehicles_1':['bicycle', 'bus', 'motorcycle','pickup_truck', 'train'],
            'vehicles_2':['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
           }

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path = os.getcwd()
path = path.split('/')
root_dir_index = path.index('ADNNTimeLeaks')

data_path = path[:root_dir_index+1]+['CheckPoints','SDNet','CIFAR100']
data_path = '/'.join(data_path)

sdn_model, sdn_params = arcs.load_model(data_path,'test',epoch=-1)

confidence_thresholds = [0.9] # set for the confidence threshold for early exits
sdn_model.forward = sdn_model.early_exit
sdn_model.confidence_threshold = confidence_thresholds[0]

sdn_model.eval()
classes =  (data_test.classes)




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
        with torch.no_grad():
            pred,exit_,_ = sdn_model(x)
        p = pred.argmax(dim=1).item()
        label = classes[p]
        new_page = render_template("index.html", prediction = label) + str(exit_)
    return new_page




if __name__ =='__main__':
    #app.debug = True
   # app.run(debug = True)
    app.run(debug = True,host=host)
