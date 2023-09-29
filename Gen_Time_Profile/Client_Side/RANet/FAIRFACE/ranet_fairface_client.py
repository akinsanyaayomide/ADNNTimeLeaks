import requests
import data_collector
import csv
import os
import io
import torch
import sys
import torchvision
from torchvision.transforms import ToTensor
import numpy as np


host = sys.argv[1]
test_url = "http://"+host+":5000/submit"
data = {'upload': ''}

batch_size_train = 500 #training bs in branchynet
validation_split = 0.2
batch_size_test =1
normalise=False 

datacoll = data_collector.TRAITDataColl(batch_size_train=batch_size_train,
                batch_size_test=batch_size_test,normalise=normalise,v_split=validation_split)
test_dl = datacoll.get_test_dl()

classes =  ['0-19','20-49','more than 50']

storage = {x:[] for x in classes}


m =1

def remove(string):
    return "".join(string.split())

for x,y in test_dl:
    time_list =[]
    for j in range(m):
        x = x
        y = y
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
    storage[label].append(time)

    b = test_response.text
    b = b[-9:]
    #print(b)
    block = remove(b[-2:])

    #label = remove(b[:7])
    #print(remove(block))
    #print(remove(b[:7]))
    #sys.exit()
    notes_path = os.path.join('.','lan_client_server_timing_ranet_Age.csv')
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
            

