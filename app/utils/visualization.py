import nibabel as nib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
import os
from flask import current_app
import io
from datetime import datetime





def visualize_single_modality_prediction(image_tensor, output_tensor, filename="result_image.png"):
    # Convert tensors to NumPy arrays
    image_np = image_tensor[0].cpu().numpy()  # Convert input image to NumPy array
    output_np = output_tensor[0].cpu().numpy()  # Convert output to NumPy array

    # Determine the number of slices along the depth (D) dimension
    depth = image_np.shape[1]  # Assuming [C, D, H, W] format
    slice_idx = depth // 2  # Middle slice for visualization

    # Extract ET, TC, WT from the output tensor
    et_mask = output_np[0]  # Enhancing Tumor (ET)
    tc_mask = output_np[1]  # Tumor Core (TC)
    wt_mask = output_np[2]  # Whole Tumor (WT)

    # Create subplots to visualize input, output, and overlay side by side
    fig, axs = plt.subplots(4, 4, figsize=(24, 24))

    # Plot input image channels
    for i in range(4):
        if i < image_np.shape[0]:  # Ensure within bounds
            axs[0, i].imshow(image_np[i, slice_idx, :, :], cmap="gray")
            axs[0, i].set_title(f"Input Channel {i+1}")
        else:
            axs[0, i].axis('off')  # Hide unused subplots

    # Plot output segmentation channels (ET, TC, WT)
    segmentation_titles = ["Enhancing Tumor (ET)", "Tumor Core (TC)", "Whole Tumor (WT)"]
    for i in range(3):
        if i < output_np.shape[0]:  # Ensure within bounds
            axs[1, i].imshow(output_np[i, slice_idx, :, :], cmap="jet")
            axs[1, i].set_title(segmentation_titles[i])
        else:
            axs[1, i].axis('off')  # Hide unused subplots

    # Plot overlays for each tumor class (ET, TC, WT)
    for i in range(3):
        if i < image_np.shape[0] and i < output_np.shape[0]:  # Ensure within bounds
            overlay = np.copy(image_np[0, slice_idx, :, :])
            axs[2, i].imshow(overlay, cmap="gray", alpha=0.7)  # Input in grayscale
            axs[2, i].imshow(output_np[i, slice_idx, :, :], cmap="jet", alpha=0.3)  # Output in color
            axs[2, i].set_title(f"{segmentation_titles[i]} Overlay")
        else:
            axs[2, i].axis('off')  # Hide unused subplots

    # Plot dummy images for unused subplots in the last row
    for i in range(4):
        axs[3, i].axis('off')  # Hide all unused subplots in the last row

    # Define the path for the uploads directory
    uploads_dir = os.path.join(current_app.root_path, 'static/uploads')  # 'static/uploads'
    os.makedirs(uploads_dir, exist_ok=True)  # Ensure the uploads directory exists

     # Generate a unique filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., "20240916_121530"
    filename = f"result_image_{timestamp}.png"

    # Save the figure to the uploads folder
    filepath = os.path.join(uploads_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)

    return 'static/uploads/' + filename

def visualize_multiple_modality_prediction(inputs, prediction, selected_slices=None):
    """
    Visualize the input modalities and the model's prediction.

    Parameters:
    - inputs: A 4D tensor or numpy array of shape (4, H, W, D), where each channel is a different modality.
    - prediction: The model's output, assumed to be a 4D tensor or numpy array of shape (C, H, W, D) 
                  where C is the number of classes.
    - selected_slices: A list of slice indices to visualize. If None, all slices are shown.
    """

    # Convert tensors to numpy arrays if necessary
    if torch.is_tensor(inputs):
        inputs = inputs.cpu().numpy()
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()

    # Remove batch dimension if present
    inputs = inputs[0] if inputs.ndim == 5 else inputs
    prediction = prediction[0] if prediction.ndim == 5 else prediction

    selected_slices = [10, 30, 50, 70]

    # If selected slices are not provided, visualize all slices
    if selected_slices is None:
        selected_slices = range(inputs.shape[3])

    # Adjust the number of rows to accommodate all inputs and class predictions + overlay
    total_rows = 4 + prediction.shape[0] + 1  # 4 for input modalities, C for predictions, 1 for overlay
    fig, axes = plt.subplots(total_rows, len(selected_slices), figsize=(15, 8))

    # Plot input modalities (FLAIR, T1CE, T1, T2)
    for i, modality in enumerate(["FLAIR", "T1CE", "T1", "T2"]):
        for j, slice_idx in enumerate(selected_slices):
            axes[i, j].imshow(inputs[i, :, :, slice_idx], cmap='gray')
            axes[i, j].set_title(f"{modality} Slice {slice_idx}", fontsize=8)
            axes[i, j].axis('off')

    # Plot model predictions for each class (ET, WT, TC)
    classes = ["ET", "WT", "TC"]
    for class_idx in range(prediction.shape[0]):
        for j, slice_idx in enumerate(selected_slices):
            axes[4 + class_idx, j].imshow(prediction[class_idx, :, :, slice_idx], cmap='hot', alpha=0.5)
            axes[4 + class_idx, j].set_title(f"Prediction {classes[class_idx]} Slice {slice_idx}", fontsize=8)
            axes[4 + class_idx, j].axis('off')

    # Plot overlayed input and prediction
    for j, slice_idx in enumerate(selected_slices):
        overlay = np.zeros_like(inputs[0, :, :, slice_idx])
        for class_idx in range(prediction.shape[0]):
            overlay += prediction[class_idx, :, :, slice_idx] * (class_idx + 1)

        axes[4 + prediction.shape[0], j].imshow(inputs[0, :, :, slice_idx], cmap='gray')
        axes[4 + prediction.shape[0], j].imshow(overlay, cmap='jet', alpha=0.3)
        axes[4 + prediction.shape[0], j].set_title(f"Overlay Slice {slice_idx}", fontsize=8)
        axes[4 + prediction.shape[0], j].axis('off')

    plt.tight_layout(pad=2.0)

    # Define the path for the uploads directory
    uploads_dir = os.path.join(current_app.root_path, 'static/uploads')  
    os.makedirs(uploads_dir, exist_ok=True)  


     # Generate a unique filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., "20240916_121530"
    filename = f"result_image_{timestamp}.png"


    filepath = os.path.join(uploads_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)

    return 'static/uploads/' + filename


