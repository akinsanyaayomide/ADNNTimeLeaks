# Timing Channels in Adaptive Neural Networks

This repository contains source code, scripts and saved model objects that can be used to reproduce similar results to the lan results table in our paper.


## Prerequisites
The code was tested using Python (3.8)

**Packages**: Install using `pip install -r requirements.txt`


**model checkpoints, cancer and fairface data**: Our model checkpoints, cancer and fairface data can be gotten by running 

```bash
chmod 777 get_data_checkpoints.sh
./get_data_checkpoints.sh
```

The downloaded checkpoints and Data will be unpacked to `./CheckPoints` and `./Data` for further use. The folder also contains various checkpoints from each stage of training.

## Running the Experiment
The Experiment describes how the timing channels of adaptive neural networks can be exploited over a LAN, the experiment can also be extended to the public internet as well, with the only difference being that the server side is hosted on some public cloud server.

The experiment comprises of two main stages
1. Profiling Stage - This stage describes how the client(adversary) obtains the timing profile of the target model hosted on the server side.
2. Evaluation Stage - This stage describes how the client(adversary) evaluates the target model timing profile obtained over the LAN in the profiling stage for information leakage.

## Profiling Stage
For experiment over a LAN, ideally two machines would be needed to represent the client and the server side respectively and connected together via some network device such as switch or a router. If this setup proves to be an hassle the experiment can still be simulated using a single machine with two terminals to simulate the server and the client machines respectively.

## Server Side 
While in the root directory `cd` into the `Gen_Time_Profile/Server_Side` directory 
`cd` into the ADNN arcitecture of choice e.g `Branchy-AlexNet`
`cd` into the Dataset of choice e.g `CANCER`
from here run
```bash
python3 app_alexnet_cancer.py 127.0.0.1
```
where 127.0.0.1 represents the ip address the server is hosted at. This value can be changed depending on what ip address the server is hosted at.


## Client Side 
While in the root directory `cd` into the `Gen_Time_Profile/Client_Side` directory 
`cd` into the ADNN arcitecture directory that you selected for the server side e.g `Branchy-AlexNet`
`cd` into the Dataset directory that you selected for the server side  e.g `CANCER`
from here run
```bash
python3 branch_client_cancer.py 127.0.0.1
```
where 127.0.0.1 represents the ip address the server is hosted at. This value must be the same as whatever value the server is hosted on.

Once the client side script is done running a timing profile csv file would be generated e.g `lan_client_server_timing_branch_cancer.csv`

This completes the Profiling stage

## Evaluation Stage

To evaluate the timing profile just generated, we do the following
While in the root directory `cd` into the `Eval_Time_Profile` directory 
from here run 
```bash
python3 evaluate.py Example_Lan_Results/lan_client_server_timing_branch_cancer.csv
```
where `Example_Lan_Results/lan_client_server_timing_branch_cancer.csv` is the path of the timing profile csv file just generated from the profiling stage.
This would further generate a bunch of output files which further describe the result such as the input distribution per cluster, kde graphs of clusters, histograms showing input distribution per cluster etc.

