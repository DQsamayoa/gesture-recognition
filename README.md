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
