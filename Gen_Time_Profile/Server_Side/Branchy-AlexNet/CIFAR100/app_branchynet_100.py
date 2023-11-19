import Branchynet
from flask import Flask, render_template, request
import torchvision
import os
import torch
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

classes =  (data_test.classes)


threshold = [1.0e-04, 5.0e-02, 1.0000e+04]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Branchynet.B_AlexNet()

model.to(device)

path = os.getcwd()
path = path.split('/')
root_dir_index = path.index('ADNNTimeLeaks')

data_path = path[:root_dir_index+1]+['CheckPoints','Branchy-AlexNet','CIFAR100']
data_path = '/'.join(data_path)

model_path = os.path.join(data_path,'branch_100.pth')

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
        x = torch.unsqueeze(x, dim=0)
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
    app.run(debug = True,host=host)
