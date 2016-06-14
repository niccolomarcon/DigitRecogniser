# DigitRecogniser
A neural network that learns to recognize hand written digits

### Installation
This project uses mainly two libraries: numpy and matplotlib. cPickle and gzip are also used to load the mnist database. The easiest way to get started is to install [Anaconda](https://www.continuum.io/why-anaconda), information can be found [here](https://www.continuum.io/downloads) for every OS.

If your goal is a simple presentation click this [link](https://nbviewer.jupyter.org/github/niccolomarcon/DigitRecogniser/blob/master/notebook.ipynb) to open the jupyter notebook in your browser so you do not have to install anything. Instead if you want to interact with the notebook without installing anything click this tag [![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/niccolomarcon/DigitRecogniser) and then open the notebook (:warning: it is really slow :warning:)

### Run the script
Simply run
```
$ python main.py
```
You should see an output like this
```
Loading the dataset...
Dataset loaded

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

### Help
Currently the best score is 98.89% on training and 83.41% on test. If you get a better score please submit a pull request with the settings you used. Thank you! 🚀
