"""
Maximilian Otto, 2022, maxotto45@gmail.com
Utility functions for training and evaluating models.
"""

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import time
import copy 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
import torch.nn as nn
from packages.invertible_resnet_master.models.conv_iResNet import conv_iResNet as iResNet

def try_make_dir(d):
    '''
    Create Directory-path, if it doesn't exist yet.
    '''
    import os
    if not os.path.isdir(d):
        os.makedirs(d)
    return


def get_mean_and_std(data_dir):
    '''
    Acquire the mean and std color values of all images (RGB-values) in the training set.
    inpupt: "data_dir" string
    output: mean and std Tensors of size 3 (RGB)
    '''
    # Load the training set
    train_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transforms.ToTensor()) for x in ["train"]}
    train_loader = {x: torch.utils.data.DataLoader(dataset=train_dataset[x], batch_size=1, num_workers=0) for x in ["train"]}
    # Calculate the mean and std of the training set
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(train_loader["train"], desc="Calculating mean and std of all RGB-values"):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    #var[x] = E[x**2] - E[X]**2
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    print("Mean: ", mean, ", Std: ", std)
    return mean, std


def get_init_batch(dataloader, batch_size):
    """
    gets a batch to use for init / warm-up epoch.
    I-ResNet-Style
    """
    batches = []
    seen = 0
    for x, _ in dataloader:
        batches.append(x)
        seen += x.size(0)
        if seen >= batch_size:
            break
    batch = torch.cat(batches)
    return batch


def get_model_type(model_type, num_classes = 2, 
                    num_channels = [32,64,128,128,128], num_blocks = [7,7,7,7,7], num_strides = [1,2,2,2,2], 
                    init_ds = 2, inj_pad=13, in_shape = (3,224,224), coeff=0.8,
                    num_trace_samples=1, num_series_terms=5, num_power_iter_spectral_norm=15,
                    density_estimation=False, no_actnorm=False, fixed_prior=True, nonlin="elu"):
    '''
    Get the model type. Either a ResNext50 with a modified classifier, or the iResNet with given parameters.
    The ResNext50 is meant as the baseline model for comparing different constellations of iResNets.
    input: model specifications. See below for details.
    output: model object
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    if model_type == "ResNext50":
        # Load pretrained model & modify it
        model = torchvision.models.resnext50_32x4d(pretrained=False)
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 256),
                                nn.Dropout(p=0.4),
                                nn.Linear(256, 100),
                                nn.ReLU(inplace=True),
                                nn.Linear(100, num_classes))

    if model_type == "ResNet50":
        # Load pretrained model & modify it
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 256),
                                nn.Dropout(p=0.4),
                                nn.Linear(256, 100),
                                nn.ReLU(inplace=True),
                                nn.Linear(100, num_classes))

    if model_type == "iResNet":
        # Build the invertible ResNet with given parameters
        model = iResNet(nBlocks=num_blocks, nStrides=num_strides,
                        nChannels=num_channels, nClasses=num_classes,
                        init_ds=init_ds,
                        inj_pad=inj_pad,
                        in_shape=in_shape,
                        coeff=coeff,
                        numTraceSamples=num_trace_samples,
                        numSeriesTerms=num_series_terms,
                        n_power_iter=num_power_iter_spectral_norm,
                        density_estimation=density_estimation,
                        actnorm=(not no_actnorm),
                        learn_prior=(not fixed_prior),
                        nonlin=nonlin)
    return model


# Save the whole model
def save_model(model, dir, model_name):
    '''Save a whole model, not only its weights, to a file.'''
    torch.save(model, f"{dir}/{model_name}.pt")


def train_iResNet(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, output_model_path, output_model_name, num_epochs=200):
    '''
    This training is without a density estimation. It focusses on the training of the classifier.
    Train the iResNet model like it is done in the iResNet-paper, but without the viz-server-connection and with all epochs. No need to call the training function in a loop.
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    try_make_dir(output_model_path)
    since = time.time()
    # Keep track of accuracy and loss
    best_model_wts = copy.deepcopy(model.state_dict())
    train_losses = []
    train_accus = []
    test_losses = []
    test_accus = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range (1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, _ = model(inputs, ignore_logdet=True)
                    pred_scores, pred_labels = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)  # Loss
                    if phase == "train":
                        loss.backward()   # Backward Propagation
                        optimizer.step()  # Optimizer update
                        scheduler.step()  # maybe use another learning_rate_scheduler
                    # stats
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(pred_labels == labels.data)

            # print stats
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase] * 100.
            if phase == "train":
                train_losses.append(epoch_loss)
                train_accus.append(epoch_acc.item())
            else: 
                test_losses.append(epoch_loss)
                test_accus.append(epoch_acc.item())
            print('{} Loss: {:.4f} Acc: {:.3f}%'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            # save depending on accuracy and logit mean would be nice. -> better Z-Prime
            if phase == 'test' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_model(model, output_model_path, output_model_name)
                print(">> Current model saved as: ", output_model_name)
                print(">> Current model saved in: ", output_model_path)
        print()
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed //60, time_elapsed % 60))
    print("Best accuracy: {:3f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    print("----DONE----")

    # Plot training curves and save the model
    plot_loss(train_losses, test_losses, output_model_name, output_model_path)
    plot_accuracy(train_accus, test_accus, output_model_name, output_model_path)
    save_model(model, output_model_path, output_model_name)
    print(">> Best model saved as: ", output_model_name)
    print(">> Best model saved in: ", output_model_path)
    return model


def imshow(img):
    '''Utility function to display an image returned of the Dataloader()''' 
    plt.imshow(np.transpose(img, (1, 2, 0)))

def reconstruct_and_print_sample_images(model, image_datasets, class_names):
    '''Reconstruct and Visualize 5 sample images'''    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get a Batch of samples
    dataloaders = torch.utils.data.DataLoader(image_datasets["test"], batch_size=5, shuffle=True, num_workers=0)
    images, labels = next(iter(dataloaders))
    images = images.to(device)
    labels = labels.to(device)
    # Sample NN-outputs
    model.eval()
    output, z_space = model(images)
    images = images.detach().cpu().numpy()

    # Invert the z_space back to an image
    reconstructions = model.inverse(z_space)
    print("-----Reconstruct and Visualize 5 sample images-----")
    reconstructions = reconstructions.view(5, images.shape[1], images.shape[2], images.shape[3])
    reconstructions = reconstructions.detach().cpu().numpy()

    # Original Images
    print("Original Images:")
    fig, _ = plt.subplots(
        nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12, 6))
    for idx in np.arange(5):
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(class_names[labels[idx]])
    plt.show()

    # Reconstructed Images
    print("Reconstructed Images")
    fig, _ = plt.subplots(
        nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12, 4))
    for idx in np.arange(5):
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        imshow(reconstructions[idx])
        ax.set_title(class_names[labels[idx]])
    plt.show()
    return

# ----------------------------------Plots-------------------------------
def plot_loss(train_losses, test_losses, output_model_name, output_model_path):
    '''Plot the loss per epoch of the training and test set'''
    plt.clf()
    plt.plot(train_losses, "-o")
    plt.plot(test_losses, "-o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Test"])
    plt.title("Train vs. Test Loss")
    plt.savefig(
        f"{output_model_path}/train_test_loss_{output_model_name}.png")
    return

def plot_accuracy(train_accus, test_accus, output_model_name, output_model_path):
    '''Plot the accuracy per epoch of the training and test set'''
    plt.clf()
    plt.plot(train_accus, "-o")
    plt.plot(test_accus, "-o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(["Train", "Test"])
    plt.title("Train vs. Test Accuracy")
    plt.savefig(
        f"{output_model_path}/train_test_accuracy_{output_model_name}.png")
    return
