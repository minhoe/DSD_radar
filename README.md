## Neural Network Prediction Model

### Set the working envorinment
```bash
# Clone the source codes
$ git clone https://github.com/minhoe/DSD_radar.git
$ cd DSD_radar

# Install necessary libraries
$ pip install -r requirements.txt
```

### How to train
```bash
$ python -m main --exp={EXP1|EXP2|EXP3} {--log} 

# EXP1 : Z, ZDR --> Dm, W
# EXP2 : Z, ZDR --> R
# EXP3 : Z, ZDR, KDP --> R

```

### How to test the model
```bash
$ python -m test --exp={EXP1|EXP2|EXP3} --timestamp={YYYYmmDDhhMMss} --model_file={*.pth}

# timestamp : Subfolder name in the results folder
# model_file : saved model file which is in the subfolder above

```