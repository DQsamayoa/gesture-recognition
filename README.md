<img src="https://raw.githubusercontent.com/DQsamayoa/personal-webpage/master/imgs/logo_vs_b.png" alt="logo" align="right" height="200">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Gesture Recognition
--------

This is a project created during my research stay for my M.Sc. Data Science at ITAM. The idea is to build a deep learning architecture using transfer learning in the CNN layers for a pattern detection in the images, then pass to a LSTM network to identify the principal sequence and then for a dense layer for classification purposes.

Dependencies
--------

- tensorflow
- keras-video-generators

The repository has an `environmet.yml` file to install all the dependencies using conda:

```
conda env create -f environment.yml
```

the conda environment is called `cnn-lstm`.

Clean tensorflow log
--------

The code used to clean the tensorflow logo in the `report/data` folder is:

```bash
awk 'BEGIN {OFS=","; print "cnn_layer,loss,acc,val_loss,val_acc"}
  /Starting/ || /step/ {
  if ($1 == "Starting" && $3 != "tunning") {cnn_layer=$3}; print cnn_layer,$8,$11,$14,$17;
}' training.log | awk '{FS=OFS=","; gsub(/\./,"",$1)}1' > train.csv
```

To Do
--------

- Modify `run_experiment.py` script to accept arguments to be more flexible.
