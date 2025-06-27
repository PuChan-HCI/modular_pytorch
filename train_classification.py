"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import sys
import torch
import data_setup, engine, model_builder, utils
import matplotlib.pyplot as plt

from torchvision import transforms


def main():
    # Setup hyperparameters
    if len(sys.argv) > 1:
        NUM_EPOCHS = 0
        BATCH_SIZE = 0
        HIDDEN_UNITS = 10
        LEARNING_RATE = 0.0
        for ix in range(len(sys.argv)):
            if sys.argv[ix] == "--batch_size":
                BATCH_SIZE = int(sys.argv[ix + 1])
            if sys.argv[ix] == "--num_epochs":
                NUM_EPOCHS = int(sys.argv[ix + 1])
            if sys.argv[ix] == "--learning_rate":
                LEARNING_RATE = float(sys.argv[ix + 1])
        if NUM_EPOCHS == 0:
            NUM_EPOCHS = 10
        if BATCH_SIZE == 0:
            BATCH_SIZE = 32
        if LEARNING_RATE == 0.0:
            LEARNING_RATE = 0.001
            
    else:
        NUM_EPOCHS = 10
        BATCH_SIZE = 32
        HIDDEN_UNITS = 10
        LEARNING_RATE = 0.001

    # Setup directories
    train_dir = "data/pizza_steak_sushi/train"
    test_dir = "data/pizza_steak_sushi/test"

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create transforms
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # Create model with help from model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

    # Start training with help from engine.py
    history = engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS,
                 device=device)

    acc = history['train_acc']
    val_acc = history['test_acc']
    loss = history['train_loss']
    val_loss = history['test_loss']
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(NUM_EPOCHS), acc, label='Training Accuracy')
    plt.plot(range(NUM_EPOCHS), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(NUM_EPOCHS), loss, label='Training Loss')
    plt.plot(range(NUM_EPOCHS), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    

    # Save the model with help from utils.py
    utils.save_model(model=model,
                     target_dir="models",
                     model_name="05_going_modular_script_mode_tinyvgg_model.pth")


if __name__ == '__main__':
    main()