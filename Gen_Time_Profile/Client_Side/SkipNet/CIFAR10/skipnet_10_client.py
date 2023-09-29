import requests
import csv
import os
import io
import torch
import sys
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from torchvision.transforms import ToTensor
import numpy as np

host = sys.argv[1]
test_url = "http://"+host+":5000/submit"
data = {'upload': ''}

transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
data_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


classes =  ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

m =1

def remove(string):
    return "".join(string.split())

for i in range(10000):
    time_list =[]
    for j in range(m):
        x = data_test[i][0]
        y = data_test[i][1]
        label = classes[y]
        torch.save(x, 'x.pt')
        test_file = open('x.pt', "rb")
        

        test_response = requests.post(test_url,data=data, 
                                            files={'x':test_file})
        time_list.append(test_response.elapsed.total_seconds())
   
    b = test_response.text
    b = b[-9:]
    block = remove(b[-2:])
    time = sum(time_list)/m
    time = round(time*int(block)/18,4)

    
    notes_path = os.path.join('.','lan_client_server_timing_skipnet_10.csv')
    if os.path.exists(notes_path) == True :    
        with open (notes_path,"a") as csvfile:
            fieldnames = ['No of Blocks','Label','Time']
            filewriter = csv.DictWriter(csvfile,fieldnames=fieldnames)
            
            filewriter.writerow({fieldnames[0]: block,fieldnames[1]:label,
            fieldnames[2]:time})
            csvfile.close()


    else:
        with open (notes_path,"w") as csvfile:
            fieldnames = ['No of Blocks','Label','Time']
            filewriter = csv.DictWriter(csvfile,fieldnames=fieldnames)
            filewriter.writeheader()
            #print(result.stdout)
            #filewriter.writerow(["X_new_2"])
            filewriter.writerow({fieldnames[0]: block,fieldnames[1]:label,
            fieldnames[2]:time})
            csvfile.close()
