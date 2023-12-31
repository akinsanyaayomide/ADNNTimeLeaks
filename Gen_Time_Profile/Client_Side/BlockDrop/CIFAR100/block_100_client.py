import requests
import csv
import os
import torch
import sys
import torchvision
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np

host = sys.argv[1]
test_url = "http://"+host+":5000/submit"
data = {'upload': ''}

data_test = torchvision.datasets.CIFAR100('./data_',
        download=True,train=False,transform=ToTensor())
classes =  (data_test.classes)

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


m =1

def remove_outliers(file_path):
    df = pd.read_csv(file_path)
    result = pd.DataFrame()
    n=1.0
    for i in df["No of Blocks"].unique():
        #print(i)
        df_1 = df[df['No of Blocks'] == i]
        df_1.reset_index(drop=True, inplace=True)
        Q1 = np.percentile(df_1['Time'], 25, method='midpoint')
        Q3 = np.percentile(df_1['Time'], 75, method='midpoint')
        IQR = Q3 - Q1
        upper = np.where(df_1['Time'] >= (Q3 + n * IQR))
        lower = np.where(df_1['Time'] <= (Q1 - n * IQR))

        ''' Replacing the Outliers with Mean '''
        non_outlier_values = df_1[(df_1['Time'] >= (Q1 - n * IQR)) & (df_1['Time'] <= (Q3 + n * IQR))]['Time']
        mean_value = round(np.mean(non_outlier_values),4)
        df_1.loc[upper[0], 'Time'] = mean_value
        df_1.loc[lower[0], 'Time'] = mean_value

        result = pd.concat([result, df_1])

    result = result.reset_index(drop=True, inplace=False)
    result = result.drop(["No of Blocks"], axis=1)
    df = result
    
    return df

def remove(string):
    return "".join(string.split())

def get_key(my_dict,val):
    for key, value in my_dict.items():
        if val in value:
            return key
 
    return "key doesn't exist"

for i in range(10000):
    time_list =[]
    for j in range(m):
        x = data_test[i][0]
        y = data_test[i][1]
        label = get_key(new_dict,classes[y])
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
    #print(block)
    #label = remove(b[:7])
    #print(remove(block))
    #print(remove(b[:7]))
    #sys.exit()
    notes_path = os.path.join('.','lan_client_server_timing_block_100.csv')
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
            
df = remove_outliers(notes_path)
df.to_csv(notes_path)