#Set Tensorflow Logging Parameter
import os 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import all Necesary Libaries

import sys 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statistics
sns.set_theme(style="whitegrid")
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
#import keras.backend


# Function to remove Outliers
def remove_outliers(file_path):
    
    df = pd.read_csv(file_path)
    
    
    return df

# Function to find Cluster Ranges

def find_cluster_ranges(x,y):
    lowest_points = [x[0]]
    for i in range(1, len(y) - 1):
        if y[i] < y[i-1] and y[i] < y[i+1]:
            lowest_points.append(x[i])
    lowest_points.append(x[-1])
    return lowest_points

def output_histogram(X,Y,filename):
    fig = plt.figure(figsize = (10, 6))

    n, bins, patches = plt.hist([X[Y==i] for i in (classes)],edgecolor='black',
            stacked=False, label=classes)
    plt.xlabel('Response Time in Seconds')
    plt.ylabel('Frequency')
    #plt.xticks(rotation=90)
    plt.title('Histogram of Data by Class')
    plt.legend()
    plt.savefig(filename, dpi=300)

def get_cluster_accuracy(n_clusters,model,X_test,Y_test):
    clusters_str = ["Cluster_"+ str(i+1) for i in range(n_clusters)]
    ratio = {x:0 for x in clusters_str}
    Exit_Buckets = {x:{y:0 for y in classes} for x in clusters_str}

    for i in range(len(Y_test)):
        input_ =np.expand_dims(X_test[i], axis=0) 
        cluster = int(input_[0][1])
        ratio[clusters_str[cluster]] +=1
        label = classes[Y_test[i].argmax()]
        pred = model.predict(input_)
        #print(Y_test[i])
        #sys.exit()

        if pred[0]== Y_test[i]:
            Exit_Buckets[clusters_str[cluster]][label]+=1
    total = []
    for cluster in Exit_Buckets:
        total.append(sum(Exit_Buckets[cluster].values()))
        results = []
        for x,y in zip(total,list(ratio.values())):
            try:
                accu = round((x/y)*100,2)
                results.append(accu)
            except:
                accu = 0
                results.append(0)
    #results = [round((x/y)*100,2) for x,y in zip(total,list(ratio.values()))]
    return results

def ent_leakage(n_clusters,classes):
    clusters_str = ["Cluster_"+ str(i+1) for i in range(n_clusters)]
    Exit_Buckets = {x:{y:0 for y in classes} for x in clusters_str}
    for i in range(len(df)):
        index = df.iloc[i]["Cluster"]
        label = df.iloc[i]["Label"]
        #print(index)
        Exit_Buckets[clusters_str[index]][label]+=1
    a = {i:{x:round((j/sum(Exit_Buckets[i].values()))*100,2) for x,j in 
        zip(classes,Exit_Buckets[i].values())} for i in clusters_str}
    b = {i:{x:round((j/sum(Exit_Buckets[i].values()))*1,2) for x,j in 
        zip(classes,Exit_Buckets[i].values())} for i in clusters_str}
    
    
    max_probs = [1/len(classes) for x in range(len(classes))]
    max_entropy = entropy(max_probs,base=2)

    clusters_entropy = ["Cluster_"+ str(i+1) for i in range(n_clusters)]
    clusters_entropy = [x +"_Entropy" for x in clusters_entropy]

    c = {x:round(entropy(list(b[cluster].values()),base=2),2) for x,cluster in 
     zip(clusters_entropy,clusters_str)}
    
    list_c = list(c.values())

    min_c = min(list_c)

    index = list_c.index(min_c)

    for i in range(len(list_c)):
        print("\n")
        print("Leakage of Cluster ", str(i+1), " is", round(((max_entropy-list_c[i])/max_entropy)*100,2))
    
    print("\n")
    print("Cluster ",str(index+1), "has the highest Information" 
          "leakage with a leakage percentage of ",round(((max_entropy-list_c[index])/max_entropy)*100,2))

    return Exit_Buckets,b

def train_model(df,filename,outputSize):
    encoder = OneHotEncoder()
    X = df[['Time','Cluster']].to_numpy()
    Y = df[['Label_Encoded']].to_numpy()
    Y = Y.ravel()
    #encoder.fit(Y)
    #Y = encoder.transform(Y).toarray()

    # Split X and Y into Train and Test

    X_train, X_test_, Y_train, Y_test_ = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Further Split Test into Test and Val

    X_test,X_val,Y_test,Y_val =  train_test_split(X_test_, Y_test_, test_size=0.4, random_state=42)

    #model_1 = LogisticRegression()
    model_1 = DecisionTreeClassifier()
    #model_1 = SVC(kernel='linear', C=1.5)
    #model_1 = RandomForestClassifier(n_estimators=100, random_state=42)
    #model_1 = KNeighborsClassifier(n_neighbors=n_clusters)
    
    model_1.fit(X_train,Y_train)


    """# Define Model
    # Define the input size, hidden size, and output size of the neural network
    input_size = 2  # Assumes one-dimensional input data
    hidden_size = 64
    output_size = outputSize  # Number of classes in the classification task

    # Instantiate the neural network
    keras.backend.clear_session()
    model_1 = Sequential()
    model_1.add(Dense(hidden_size, input_dim=input_size, activation='relu'))
    model_1.add(Dense(32, input_dim=64, activation='relu'))
    model_1.add(Dense(16, input_dim=32, activation='relu'))
    model_1.add(Dense(output_size, activation='softmax'))

    model_1.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model_1.fit(X_train, Y_train,
                    batch_size=64, epochs =100,
                   validation_data=(X_val,Y_val),verbose=0)
    
    epochs = range(100)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.clf()
    plt.plot(epochs,train_loss,'b',label= 'Training Loss')
    plt.plot(epochs,val_loss,'r',label= 'Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename, dpi=300)"""

    return model_1, X_test, Y_test


def msr(df,filename,outputsize):
    # Train Model
    resultpath = filename[:-3]+"txt"
    print("\n")
    print("Model Training Started")
    model,X_test,Y_test = train_model(df,filename,outputsize)
    print("\n")
    print("Model Training Complete")

    # Evaluate Model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)

    result = [accuracy]
    #result = model.evaluate(X_test,Y_test,verbose=0)
    print("\n")
    random = (1/outputsize) *100
    print("Overall Model Accuracy on Target Secret is ",round((result[0]*100),2),"% ","Compared to ",random,"%"," of Random Guessing")


    # Get Accuracy of Individual Cluster
    cluster_accuracy = get_cluster_accuracy(n_clusters,model,X_test,Y_test)
    print("\n")
    print("Model Accuracy on all the different Clusters is: ", cluster_accuracy)
    for i in range(len(cluster_accuracy)):
        print("\n")
        print('Model Accuracy of all Samples that fall into Cluster', i+1, 'is', cluster_accuracy[i])
    with open(resultpath,'w') as file:
        a = "Total Number of Time Clusters Found is "+str(len(cluster_accuracy))
        file.write(a)
        file.write("\n")
        file.write("Model Training Started")
        file.write("\n")
        file.write("Model Training Complete")
        file.write("\n")
        a = "Overall Model Accuracy on Target Secret is "+ str(round((result[0]*100),2))+"% "+"Compared to "+str(random)+"%"+" of Random Guessing"
        file.write(a)
        file.write("\n")
        a = "Model Accuracy on all the different Clusters is: "+ str(cluster_accuracy)
        file.write(a)
        file.write("\n")
        for i in range(len(cluster_accuracy)):
            file.write("\n")
            a = 'Model Accuracy of all Samples that fall into Cluster'+ str(i+1)+ ' is '+str(cluster_accuracy[i])
            file.write(a)





# Load Timing Measurements from a File and Remove Outliers 
file_path = sys.argv[1]

df = remove_outliers(file_path)


# Use Kernel Density Esitimate (KDE) Algorithm to find Optimal Partitions (Clusters)

# Visualize KDE's Output
filename = 'output_'
filename = filename + '_kde_clusters.png'
ax = sns.kdeplot(df.Time,bw_adjust=1)
kde_fig = ax.get_figure()
kde_fig.savefig(filename,dpi=300)

kde_data = sns.kdeplot(df.Time,bw_adjust=1).get_lines()[0].get_data()
x = kde_data[0]
y = kde_data[1]

lowest_points = find_cluster_ranges(x,y)
lowest_points = sorted(lowest_points)
#print(len(lowest_points))

n_clusters = len(lowest_points)-1
#print("\n")
#print("Total Number of Time Clusters Found is ",n_clusters)

# Define Class labels

classes = list(df["Label"].unique())
label_encoder = LabelEncoder()

# Add a Bins, Cluster, Label_Encoded Column to the Dataframe
df['Bins'] = pd.cut(x=df['Time'], bins=lowest_points,include_lowest=False )
df['Cluster'] = label_encoder.fit_transform(df['Bins'])
df['Label_Encoded'] = label_encoder.fit_transform(df['Label'])

# Calculate the percentage distribution of values in the 'Cluster' column
percentage_distribution = df['Cluster'].value_counts(normalize=True) * 100

# Convert the Series to a dataframe
df_percentage = percentage_distribution.reset_index()
df_percentage.columns = ['Cluster', 'Percentage']
b = 'output'+'_cluster_input_distribution.csv'
df_percentage.to_csv(b)

# Calculate the percentage distribution of values in the 'Cluster' column
percentage_distribution_ = df['Bins'].value_counts(normalize=True) * 100

# Convert the Series to a dataframe
df_percentage_ = percentage_distribution_.reset_index()
df_percentage_.columns = ['Bins', 'Percentage']
b_ ='output'+'_bins_input_distribution.csv'
df_percentage_.to_csv(b_)

# Create Histogram Profile from the KDE cluster Ranges
df['Bins'] = df['Bins'].astype(str)
X = df[['Bins',]].to_numpy()
Y = df[['Label',]].to_numpy()


filename = 'output'
filename = filename+'_hist.png'
output_histogram(X,Y,filename)



leakage_method = 'Model'

if leakage_method == "Model":
    filename ='output_'
    filename=filename+"model_training.png"
    file_path_list = file_path.split('_')
    if file_path_list[-1][:-4] == '10':
        outputsize = 10
    elif file_path_list[-1][:-4] == '100':
        outputsize = 20
    elif file_path_list[-1][:-4] == 'cancer':
        outputsize = 2
    elif file_path_list[-1][:-4] == 'adv':
        outputsize = 2
    elif file_path_list[-1][:-4] == 'fairface':
        outputsize = 3
    else:
        print('Invalid Output Size')
        sys.exit()
    

    
    msr(df,filename,outputsize)
elif leakage_method == "Entropy":
    buckets,b = ent_leakage(n_clusters,classes)
    #print(buckets)
    #print(b)
else:
    print("Kindly select Leakage Method")






















