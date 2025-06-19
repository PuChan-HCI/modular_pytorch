# modular_pytorch

The Python scripts in this directory were generated using the notebook 05. Going Modular Part 2 (script mode).

They breakdown as follows:

**`data_setup.py`** - a file to prepare and download data if needed.<br/>
**engine.py** - a file containing various training functions.<br/>
**model_builder.py** - a file to create a PyTorch TinyVGG model.<br/>
**train.py** - a file to leverage all other files and train a target PyTorch model.<br/>
**utils.py** - a file dedicated to helpful utility functions.

To test this modular pytorch project, start by running data_get.py to set up the image files.<br/>
You can perform this by running the following line in your command line,

python data_get.py

After that train.py will perform a basic image classification program using a tiny vgg model.<br/>
You can perform this by running the folllowing line in your command line,

python train.py

If you want to adjust some of the settings, they can be adjusted via the arguments passed to the program using the following line as a guide.

python train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS

For example,

python train.py --model tinyvgg --batch_size 32 --lr 0.001 --num_epochs 10

Currently the model cannot be changed, but the other settings can be changed. Feel free to try out different settings and see what results you get.
