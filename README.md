# DigitRecogniser
A neural network that learn to recognize hand written digits

### Installation
This project uses mainly two libraries: numpy and matplotlib. cPickle and gzip are also used to load the mnist database. The easiest way to get started is to install [Anaconda](https://www.continuum.io/why-anaconda), informations can be found [here](https://www.continuum.io/downloads) for every OS.

If your goal is a simple presentation click this [link](https://nbviewer.jupyter.org/github/niccolomarcon/DigitRecogniser/blob/master/notebook.ipynb) to open the jupyter notebook in your browser so you do not have to install anything.

### Run the script
Simply run
```
$ python main.py
```
You should see an output like this
```
Loading the dataset...
Dataset Loaded

Learning...
Learned

Accuracy:
  train -> 98.60%
  test  -> 77.60%

Random test:
  guess   9
  correct 9
```
and a window with the image of a hand written nine should open.

### Run the notebook
If you have installed Anaconda you should be able to run
```
$ jupyter notebook
```
so you can play with the notebook in the browser.
