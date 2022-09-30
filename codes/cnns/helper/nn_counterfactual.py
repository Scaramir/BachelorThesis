"""
Maximilian Otto, 2022, maxotto45@gmail.com
The MAIN file for training a CNN, evaluating it, generating counterfactuals and interpolations.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
import os
from cnns.helper.nn_prepro_utils import try_make_dir

def extract_z_space(model, image_datasets, class_names, data_set="test", avgPooled_latent_space=True):
    '''Extract features of a given data set from an iResNet model'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    class_names = image_datasets["test"].classes

    # Just get each image, no need to shuffle, no need to batch, no need for multiple workers
    # because it's only a forward pass through the model
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=False, num_workers=0)
                    for x in [data_set]}
    encoded_samples = []
    encoded_samples_label_ori = []
    encoded_samples_label_pred = []

    model.eval()
    for input, label_ori in tqdm(dataloaders[data_set], desc="Extracting latent space embeddings from {} classes...".format(data_set)):
        input = input.to(device)
        label_ori = label_ori.to(device)
        # Get latent space encoding of the image, aka vars before the classifier is applied
        with torch.no_grad():
            output, z_space = model(input)
            _, label_pred = torch.max(output.data, 1)
            # Activate this part if you want to get the smaller latent space encoding
            if avgPooled_latent_space:
                z_space = F.relu(model.bn1(z_space))
                z_space = F.avg_pool2d(z_space, z_space.size(2))
                z_space = z_space.view(z_space.size(0), z_space.size(1))
        # Append to lists
        # Flatten it, so it's just a list containing the embedding of the image
        encoded_img = z_space.flatten().cpu().detach().numpy()
        encoded_samples.append(encoded_img)
        encoded_samples_label_ori.append(class_names[label_ori])
        encoded_samples_label_pred.append(class_names[label_pred])
    print("Done")
    return encoded_samples, encoded_samples_label_ori, encoded_samples_label_pred

def plot_z_space(encoded_samples, labels_ori, three_d_TSNE=False):
    '''Plot the previously obtained latent space encoding of the data set'''

    print("Plotting latent space embeddings...")
    print("PCA:")
    # PCA - Principal Component Analysis
    pca = PCA(n_components=2)
    components = pca.fit_transform(encoded_samples, labels_ori)
    # Plot the PCA results and color by label
    fig = px.scatter(components, x=0, y=1, color=labels_ori, labels={'0': 'PC 1', '1': 'PC 2'})
    fig.show()

    # TSNE for 2 dimensions
    print("TSNE 2D:")
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne_results = tsne.fit_transform(encoded_samples, labels_ori)
    # Plot the TSNE results and color by label
    fig = px.scatter(tsne_results, x=0, y=1,
                    color=labels_ori,
                    labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'})
    fig.show()

    # TSNE - t-Distributed Stochastic Neighbor Embedding for 3 dimensions
    if three_d_TSNE:
        print("TSNE 3D:")
        tsne = TSNE(n_components=3, init='pca', random_state=0)
        tsne_results = tsne.fit_transform(encoded_samples, labels_ori)
        fig = px.scatter_3d(tsne_results, x=0, y=1, z=2,
                            color=labels_ori)
        fig.show()

    print("Done")
    return

def lists_to_dataframe(encoded_samples, labels_ori, labels_pred):
    '''Convert lists to a pandas dataframe'''
    print("Converting lists to a pandas dataframe...")
    # convert to numpy array for speedboost
    encoded_samples = np.array(encoded_samples)
    labels_ori = np.array(labels_ori)
    labels_pred = np.array(labels_pred)
    # convert to dataframe
    encodings_df = pd.DataFrame(encoded_samples)
    encodings_df["label_ori"] = labels_ori
    encodings_df["label_pred"] = labels_pred
    print("Done")
    return encodings_df

def get_cluster_means(encoded_samples, by_pred=False):
    '''
    Get the mean embedding of the latent space, either by predicted or original label.
    Warning: For data sets with more than 1000 images, the DataFrame-creation is super slow and requires more than 32GB of RAM for a short amount af time.
    '''
    cluster_means = {}
    if by_pred:
        for label in tqdm(encoded_samples.label_pred.unique(), desc="Calculating mean latent space embedding by predicted labels..."):
            cluster_means[label] = encoded_samples[encoded_samples.label_pred == label].mean(axis=0, numeric_only=True).tolist()
    else:
        for label in tqdm(encoded_samples.label_ori.unique(), desc="Calculating mean latent space embedding by original labels..."):
            cluster_means[label] = encoded_samples[encoded_samples.label_ori == label].mean(axis=0, numeric_only=True).tolist()
    print("Done")
    return cluster_means

def compute_alpha(z_space, cluster_means_ori, cluster_means_pred, start_cluster_name, target_cluster_name):
    '''
    Compute the alpha-zero value for the counterfactual generation.
    This method and code is derived from:
    https://github.com/fhvilshoj/ECINN/blob/main/counterfactual/compute_and_store.py
    https://arxiv.org/pdf/2103.13701.pdf
    '''
    # Define y(α) = z + α∆p, q to be the line intersecting
    # z with direction ∆p, q
    # We wish to identify the intersec-tion between y(α) and the hyperplane that constitutes the
    # decision boundary between the two normal distributions
    # N(μp, 1) and N(μq, 1). 
    # Due to the simplicity of the co-variance matrices of the normal distributions, we can define
    # w = μq − μp and b = −((μp + μq) /2)ᵀw to form the decision boundary.
    # wᵀx + b = 0. (5)
    # To find the α-value which corresponds to the intersection, 
    # set x = z + α∆_(p,q) and solve for α in Equation(5):
    # wᵀ(z + α∆p, q) + b = 0 (6)
    # ⇒ αwᵀ∆p, q = −(wᵀz + b)(7)
    # We want this: 
    # ⇒ α = − ((wᵀz + b) / (wᵀ∆_(p,q))) (8)
    #   where w = μq − μp, b = −((μp + μq) /2)ᵀw
    #   and z is the latent space encoding of the image
    #   and p, q are the two classes
    #   and ∆_(p,q) is the direction vector of the line connecting the two clusters, e.g. mu_q - mu_p
    #   and μp, μq are the mean vectors of the two clusters
    #   and wᵀ is the transpose of the vector w
    with torch.no_grad():
        alpha_zero = 0.
        mu1 = torch.FloatTensor(cluster_means_ori[start_cluster_name])
        mu2 = torch.FloatTensor(cluster_means_ori[target_cluster_name])
        tilde_mu1 = torch.FloatTensor(cluster_means_pred[start_cluster_name])
        tilde_mu2 = torch.FloatTensor(cluster_means_pred[target_cluster_name])

        # The lenght of the vector w is the norm of the difference of the two means
        # Analyzing the latent space under L2-norm
        #w = mu2 - mu1 + according to paper
        # (not sure, why they do this is the ECINN-Code, but not in their paper)
        # NOTE: They do this for numerical stability (avoid inf/NaNs). 
        # The division by the norm of the difference of the two means is not necessary, but it is a good idea to keep the values in the same range.
        # The division gets cut out in the end anyway.
        w = (mu2 - mu1) / torch.norm(mu2 - mu1) 
        delta_pq = tilde_mu2 - tilde_mu1
        b = -((mu1 + mu2) / 2) @ w.t()
        alpha_zero = (-b - (w @ z_space.t())) / (w @ delta_pq.t())
        alpha_convincing = 0.8 + 0.5*alpha_zero
    return alpha_zero, alpha_convincing, delta_pq

def generate_counterfactual(model, input, label_ori, cluster_means_ori, cluster_means_pred, class_names, target_cluster_name=None):
    '''binary class counterfactual generation for one image of its full latent space.'''    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the output of the model
    model.eval()
    with torch.no_grad():
        output, latent_space = model(input.to(device))
        latent_space_shape = latent_space.size()
        _, label_pred = torch.max(output.data, 1)
    encoded_img = latent_space.flatten().cpu().detach() # FIXME: add .numpy() to get a numpy array and a speed boost when converting to a dataframe

    # compute alpha from predicted label to the other one
    alpha_zero, alpha_convincing, delta_pq = compute_alpha(encoded_img, cluster_means_ori, cluster_means_pred, class_names[label_pred], class_names[not label_pred])

    # Latent space correction for the counterfactual
    counterfactual_alpha_zero = encoded_img + (alpha_zero * delta_pq)
    counterfactual_alpha_convincing = encoded_img + (alpha_convincing * delta_pq)
    # Invert the z_space back to an image
    counterfactual_reconstructions_zero = model.inverse(counterfactual_alpha_zero.unflatten(0, latent_space_shape).to(device))
    counterfactual_reconstructions_convincing = model.inverse(counterfactual_alpha_convincing.unflatten(0, latent_space_shape).to(device))
    encoded_sample_reconstruction = model.inverse(encoded_img.unflatten(0, latent_space_shape).to(device))
    return counterfactual_reconstructions_zero, counterfactual_reconstructions_convincing, encoded_sample_reconstruction, alpha_zero, alpha_convincing, delta_pq

def generate_counterfactuals(model, image_datasets, cluster_means_ori, cluster_means_pred, class_names, data_set="test", amount=1, path="./counterfactuals", norm_mean=None, norm_std=None):
    '''
    Generate counterfactual images for the whole test set.
    When provided with means and stds of the dataset, the images will be de-normalized before saving. (tesnors with values for each channel)
    Warning: only works for binary class problems!
    Can be extended to Multi-Class-Problems with little effort, by providing a target_cluster_name, to generate counterfactuals for a specific cluster. 
    Apply these changes also to the generate_counterfactual function.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    loader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                shuffle=False, num_workers=0)
                for x in [data_set]}

    # for each image in the data set generate a counterfactual
    for i, (input, label_ori) in enumerate(tqdm(loader[data_set], desc="Generating counterfactuals")):
        if i == amount:
            break
        counterfactual_reconstructions_zero, counterfactual_reconstructions_convincing, encoded_sample, _, _, _ = generate_counterfactual(model, 
            input, label_ori, cluster_means_ori, cluster_means_pred, class_names)
        
        # if normalization_mean and normalization_std are provided, de-normalize the images
        if (norm_mean!=None) and (norm_std!=None):
            unnormalize = transforms.Normalize((-norm_mean / norm_std).tolist(), (1.0 / norm_std).tolist())
            counterfactual_reconstructions_zero = unnormalize(counterfactual_reconstructions_zero)
            counterfactual_reconstructions_convincing = unnormalize(counterfactual_reconstructions_convincing)
            encoded_sample = unnormalize(encoded_sample)

        # save images to disk 
        try_make_dir(path)
        save_image(counterfactual_reconstructions_zero, f"./{path}/{i}_counterfactual_reconstruction_zero.bmp")
        save_image(counterfactual_reconstructions_convincing, f"./{path}/{i}_counterfactual_reconstruction_convincing.bmp")
        save_image(encoded_sample, f"./{path}/{i}_encoded_sample_reconstruction.bmp")
    print("Done.")
    return 

def interpolate_encoded_samples(encoded_sample_1, encoded_sample_2, steps=48):
    '''
    Interpolate between two encoded_samples
    This works for total mean embeddings, so it's just a traversal from one mean embedding to the other.
    '''
    # Create empty dataframe
    encoded_samples_interpolated = []

    # interpolate between two encoded_samples
    # from the first to the second encoded_sample with steps
    for i in np.arange(0, steps):
        sample_interpolated_i = encoded_sample_1 * (1 - i / steps) + encoded_sample_2 * (i / steps)
        # other way around: (from the second to the first encoded_sample with steps)
        # encoded_sample_1 * (i / steps) + encoded_sample_2 * ((steps - i) / steps))
        encoded_samples_interpolated.append(sample_interpolated_i)    
    return encoded_samples_interpolated

def reconstruct_encoded_sample(encoded_samples_interpolated, model, save_unnormalized=True):
    '''
    Decode/reconstruct the interpolated encoded_sample
    This works for total mean embeddings, so it's just a traversal from one mean embedding to the other.
    encoded_samples_interpolated: list of encoded, interpolated samples to be reconstructed

    '''
    # Make directory to save the interpolated images
    if not os.path.exists('interpolated_images'):
        os.makedirs('interpolated_images')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Decode/reconstruct the interpolated encoded_sample
    with torch.no_grad():
        for i, encoded_sample_interpolated in enumerate(encoded_samples_interpolated):
            #encoded_sample_interpolated = torch.from_numpy(encoded_sample_interpolated).float().to(device)
            output = model.inverse(encoded_sample_interpolated.unflatten(0, [1, 2048, 14, 14]).to(device))
            s = '0'+str(i) if i < 10 else str(i)
            # Save the interpolated images
            if save_unnormalized:
                save_image(output, './interpolated_images/interpolated_reconstruction_' + s + '.bmp', format='BMP')
            else:
                save_image(output, './interpolated_images/interpolated_reconstruction_' + s + '.bmp', format='BMP', normalize=True)
    return output

def generate_interpolations(model, image_datasets, class_names, z_space_1, z_space_2, to_gif=False):
    '''Generate interpolations between two latent-/z-spaces. Save them as a gif.'''
    return

def make_gif(filepath):
    '''Read the interpolated images within one folder and combine them to a .gif-file'''
    import os, imageio
    # Save the interpolated images as a gif
    images = []
    for filename in os.listdir(filepath):
        images.append(imageio.imread(filepath + filename))
    imageio.mimsave(filepath + 'interpolated_reconstruction.gif', images, duration=0.045)
    return
#make_gif('interpolated_images/')