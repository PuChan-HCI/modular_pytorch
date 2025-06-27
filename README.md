# modular_pytorch

The files in this project break down as follows:

**`data_get.py`** - a file to download data if needed.<br/>
**`data_setup.py`** - a file to prepare data.<br/>
**`engine.py`** - a file containing various training functions.<br/>
**`model_builder.py`** - a file to create a PyTorch TinyVGG model.<br/>
**`train_classification.py`** - a file to leverage all other files and train a target PyTorch model.<br/>
**`utils.py`** - a file dedicated to helpful utility functions.

To test this modular pytorch project, start by running data_get.py to set up the image files. This downloads a dataset of pictures of pizza, steak, and sushi. The files are then spit into folders for each category and are presplit for training and test.<br/>
You can perform this by running the following line in your command line,

python data_get.py

After that train.py will perform a basic image classification program using a tiny vgg model.<br/>
You can perform this by running the folllowing line in your command line,

python train_classification.py

Batch size, learning rate, and number of epochs can be edited via the command line. These can be input in any order and any can be ommited. The values default to,<br/>
Batch_size = 32<br/>
learning_rate = 0.001<br/>
epochs = 10

python train.py --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS

For example,

python train.py --lr 0.001 --num_epochs 100 --batch_size 16
