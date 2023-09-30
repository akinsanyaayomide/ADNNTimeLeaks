import requests
import data_collector
import csv
import os
import io
import torch
import sys
import torchvision
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
host = sys.argv[1]
test_url = "http://"+host+":5000/submit"
data = {'upload': ''}

batch_size_train = 500 #training bs in branchynet
validation_split = 0.2
batch_size_test =1
normalise=False 

datacoll = data_collector.CANCERDataColl(batch_size_train=batch_size_train,
                batch_size_test=batch_size_test,normalise=normalise,v_split=validation_split)
test_dl = datacoll.get_test_dl()

classes = ['Benign','Malignant']



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
for j in range(4):
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
        
        b = test_response.text
        b = b[-9:]
        #print(b)
        block = remove(b[-2:])

        #label = remove(b[:7])
        #print(remove(block))
        #print(remove(b[:7]))
        #sys.exit()
        notes_path = os.path.join('.','lan_client_server_timing_ranet_cancer.csv')
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
