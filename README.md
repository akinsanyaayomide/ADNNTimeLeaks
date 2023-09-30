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
