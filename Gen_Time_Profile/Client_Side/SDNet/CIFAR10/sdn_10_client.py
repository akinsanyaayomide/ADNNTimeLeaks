import torch




import requests
import csv
import os
import sys
from torchvision import datasets, transforms, utils
from torchvision.transforms import ToTensor
import numpy as np

host = sys.argv[1]

test_url = "http://"+host+":5000/submit"
data = {'upload': ''}

device='cpu'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalized = transforms.Compose([transforms.ToTensor(), normalize])


data_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=normalized)
classes =  ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

#loader = dataset.test_loader
#print("called")

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

    b = test_response.text
    b = b[-9:]
    #print(b)
    block = remove(b[-2:])

    #label = remove(b[:7])
    #print(remove(block))
    #print(remove(b[:7]))
    #sys.exit()
    notes_path = os.path.join('.','lan_client_server_timing_sdn_10.csv')
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
            
