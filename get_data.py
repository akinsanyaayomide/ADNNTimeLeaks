import gdown

# Get Cancer Data and FairFace Data
url = 'https://drive.google.com/uc?id=1pPxcQFtesXGT-yjYrawWMtgdLA1YLSPd'
output = 'Data.zip'
gdown.download(url, output, quiet=False)

print('Got CANCER and FAIRFACE Data')

#Get Model Checkpoints

url = 'https://drive.google.com/uc?id=1My2R8vEHRftJqWVZSNu5yh5yGxr1lsKQ'
output = 'CheckPoints.zip'
gdown.download(url, output, quiet=False)

print('Got Model CheckPoints')


