import requests
import csv
import os
import io
import torch
import sys
import torchvision
from torchvision.transforms import ToTensor
import numpy as np


test_url = "http://127.0.0.1:5000/submit"
data = {'upload': ''}

data_test = torchvision.datasets.CIFAR10('./data_',
        download=True,train=False,transform=ToTensor())
classes =  ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
#storage = {x:[] for x in classes}

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
        #buff.seek(0)
        #test_file = buff.read()
        #test_file =torch.Tensor.byte(x)

        test_response = requests.post(test_url,data=data, 
                                            files={'x':test_file})
        time_list.append(test_response.elapsed.total_seconds())
    time = sum(time_list)/m
    time = round(time,4)
    #storage[label].append(time)

    b = test_response.text
    b = b[-9:]
    #print(b)
    block = remove(b[-2:])
    #print(block)

    #label = remove(b[:7])
    #print(remove(block))
    #print(remove(b[:7]))
    #sys.exit()
    notes_path = os.path.join('.','lan_client_server_timing_branch_10.csv')
    if os.path.exists(notes_path) == True :    
        with open (notes_path,"a") as csvfile:
            fieldnames = ['No of Blocks','Label','Time']
            filewriter = csv.DictWriter(csvfile,fieldnames=fieldnames)
            #print(result.stdout)
            #filewriter.writerow(["X_new_2"])
            #filewriter.writeheader()
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
            
