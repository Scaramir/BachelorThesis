"""
Maximilian Otto, 2022, maxotto45@gmail.com
Gradient and occlusion based attribute for example images.
This relies on the Captum-AI-interpretability library and tutorial.
Extract the values for normalization beforehand and insert them in this script. 
"""

from __future__ import print_function, division
if __name__ == '__main__':
    import torch
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from captum.attr import IntegratedGradients
    from captum.attr import Occlusion
    from captum.attr import NoiseTunnel
    from captum.attr import visualization as viz
    import os
    import random

    # Set parent folder to be the root folder:
    pic_folder_path = "S:\mdc_work\mdc_leigh\images\\04-11-2021"
    # the actual folder name:
    pic_resolution_beginning = "/224" 

    # model to use
    fully_trained_model_path = "S:/mdc_work/mdc_leigh/models/DS_vs_TT_MMP_only/ResNext50_10Epochs_on_224px.pt"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_classes = 2
    batch_size = 1     # 1 for testing

    # (reproducability)
    seed = 1129142087
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
    data_dir = pic_folder_path + pic_resolution_beginning
    os.chdir(data_dir)
    # ------------Data Augmentation---------------
    # Apply ``get_meand_and_std()`` from ``nn_prepro_util.py`` to your train-dataset 
    # and insert the resulting values into the "transforms.Normalize()"
    data_transforms = {
        "test": transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.0062, 0.0047, 0.0104], [0.0068, 0.0037, 0.0094], inplace=True)
        ]),
    }
    # --------------------------------------------

    # ---------------Data Loader------------------
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                                data_transforms[x])
                        for x in ["test"]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size,
                        shuffle=True, num_workers=4)
                        for x in ["test"]}

    dataset_sizes = {x: len(image_datasets[x]) for x in ["test"]}
    class_names = image_datasets["test"].classes
    # --------------------------------------------

    # ------------------CNN setup-----------------
    torch.cuda.empty_cache()

    # set model here
    model = torch.load(fully_trained_model_path)
    model.to(device)
    model.eval()
    # --------------------------------------------

    # --------------------------------------------
    # for visualization
    default_cmap = LinearSegmentedColormap.from_list("custom blue", 
                                                    [(0, "#ffffff"),
                                                    (0.25, "#000000"), # imagine it as a "threshold"
                                                    (1, "#000000")], N=256)

    # Generate only 12 Sample-Saliency-Maps for randomly selected pictures of test set 
    for i in range(0, 12):
        # CaptumAI-Style (https://captum.ai/tutorials/Resnet_TorchVision_Interpret)
        torch.cuda.empty_cache()
        # Load a picture and predict its class
        input, label = next(iter(dataloaders["test"]))
        input = input.to(device)
        label = label.to(device)
        output = F.sigmoid(model(input)) # for binary classification
        #output = F.softmax(model(input), dim=1)
        prediction_score, pred_label = torch.max(output, 1)
        print()
        print("-----"*10)
        title=f"Predicted: {class_names[pred_label[0]]} (Prediction score: {float(prediction_score.cpu()):.4f}) - Real: {class_names[label[0]]}"
        print(title)

        integrated_gradients = IntegratedGradients(model)
        # smoothened visualization through generating noise channels and smooth them out. clears the heatmap a bit 
        # increades nt_samples_batch_size (together calculated noise images) increases the GPU-Memory consumption
        noise_tunnel = NoiseTunnel(integrated_gradients)
        attributions_ig = noise_tunnel.attribute(input, nt_samples=8, nt_samples_batch_size=1, nt_type="smoothgrad_sq", target=pred_label)

        # Gradient based saliency maps:
        # insert overlay, not side-by-side
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                np.transpose(input.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                ["original_image", "heat_map"],
                                                ["all", "positive"],
                                                cmap=default_cmap,
                                                show_colorbar=True)

        # Occlusion-based
        occlusion = Occlusion(model)
        
        attributions_occ = occlusion.attribute(input,
                                                strides = (3, 6, 6),
                                                target=pred_label,
                                                sliding_window_shapes=(3, 15, 15),
                                                baselines=0)
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                np.transpose(input.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                ["original_image", "heat_map"],
                                                ["all", "all"],
                                                show_colorbar=True,
                                                titles=[title, "Occlusion-based attribution"],
                                                outlier_perc=2)

        # finer analysis-resolution:
        #attributions_occ = occlusion.attribute(input,
        #                                        trides = (3, 4, 4),
        #                                        arget=pred_label,
        #                                        liding_window_shapes=(3, 10, 10),
        #                                        aselines=0)
        #_ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
        #                                        np.transpose(input.squeeze().cpu().detach().numpy(), (1,2,0)),
        #                                        ["original_image", "blended_heat_map"],
        #                                        ["all", "all"],
        #                                        alpha_overlay=0.3,
        #                                        show_colorbar=True,
        #                                        titles=[title, "Occlusion-based attribution"],
        #                                        outlier_perc=2)
    # -------------------------------------------


