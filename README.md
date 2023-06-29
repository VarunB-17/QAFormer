<h1 align="center">
  <br>
  QAFormer
  <br>
</h1>

<h4 align="center"> An extractive question-answering model for SQuAD 1.1</h4>




![screenshot](https://github.com/VarunFuego17/thesisqat/blob/main/qaf.png)

## Features

* This model was built for my bachelor's thesis for the study Artificial Intelligence at Vrije Universiteit Amsterdam.
* The QAFormer is based on QAnet which was published by Google in 2018.
* Training about around ~9 hours for 3 epochs.
* Check out the config.py files to modify hyperparameters to what your machine can handle.

## How To Use
##### Create a weights and biases account before running all files

```bash
# Clone this repository
$ git clone https://github.com/VarunFuego17/thesisqat.git

# Install dependencies (latest versions)
$ pip install pytorch 
$ pip install pyarrow
$ pip install wandb
$ pip install torchtext
$ pip install datasets
$ pip install spacy
$ pip install pandas
$ pip install numpy

# Go into the repository
$ cd dataloader
# Run the following file
$ python3 dataloader.py

# This should create the following files in the dataloader folder:
```
<img width="419" alt="image" src="https://github.com/VarunFuego17/thesisqat/assets/45126763/997d5ccd-c820-415c-a911-495923ca2404">

```bash
# Go into the repository
$ cd model
# Run the following command for creating the model:
$ python3 train.py --debug=1
# This should create the file -> "qaf_128_8_4.pt"
# Run the following command for testing the model on the created dataset:
$ python3 train.py --debug=2
# Run the following command if you want to see if any errors appear:
$ python3 train.py --debug=0
```
## License

MIT

---
