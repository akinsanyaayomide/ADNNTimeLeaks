import gdown

# Get Cancer Data, FairFace Data and Adversarial CIFAR10 Data
url = 'https://drive.google.com/uc?id=1vOS2eyVzquL6YOAgVsIzHioxC0uHmmn0'
output = 'Data.zip'
gdown.download(url, output, quiet=False)

print('Got CANCER and FAIRFACE Data')

#Get Model Checkpoints

url = 'https://drive.google.com/uc?id=1RIhMYZV84t_oN2aoi7yzf2kEZh8jSUi-'
output = 'CheckPoints.zip'
gdown.download(url, output, quiet=False)

print('Got Model CheckPoints')


