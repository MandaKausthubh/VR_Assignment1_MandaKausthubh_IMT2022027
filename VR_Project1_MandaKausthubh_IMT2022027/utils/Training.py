import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def Train(
        model:nn.Module,
        TrainDataLoader:DataLoader,
        ValidationLoader:DataLoader,
        criterion:nn.Module,
        optimizer:Optimizer,
        device:torch.device,
        epochs:int):

    model.train()
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    training_accuracies, training_losses = [], []
    testing_accuracies, testing_losses = [], []

    for _ in range(epochs):
        test_total_accuracy, test_total_loss, test_samples = 0.0, 0.0, 0
        train_total_accuracy, train_total_loss, train_samples = 0.0, 0.0, 0
        for image, label in tqdm(TrainDataLoader, total=len(TrainDataLoader)):
            image = image.to(device)
            label = label.to(device)

            pred = model(image)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total_accuracy += (pred.argmax(1) == label.argmax(1)).sum().item()
            train_total_loss += loss.item() * image.shape[0]
            train_samples += image.shape[0]
        training_accuracies.append(train_total_accuracy / train_samples)
        training_losses.append(train_total_loss / train_samples)

        model.eval()

        with torch.no_grad():
            for image, label in tqdm(ValidationLoader, total=len(ValidationLoader)):
                image = image.to(device)
                label = label.to(device)

                pred = model(image)
                loss = criterion(pred, label)
                test_total_accuracy += (pred.argmax(1) == label.argmax(1)).sum().item()
                test_total_loss += loss.item() * image.shape[0]
                test_samples += image.shape[0]
            testing_accuracies.append(test_total_accuracy / test_samples)
            testing_losses.append(test_total_loss / test_samples)

    ax[0].plot(training_accuracies, label='Training Accuracy')
    ax[0].plot(testing_accuracies, label='Testing Accuracy')
    ax[0].legend()
    ax[0].set_title('Accuracy')
    
    ax[1].plot(training_losses, label='Training Loss')
    ax[1].plot(testing_losses, label='Testing Loss')
    ax[1].legend()
    ax[1].set_title('Loss')
    plt.show()

    return training_accuracies, training_losses, testing_accuracies, testing_losses



