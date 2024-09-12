import nibabel as nib
import numpy as np 
import torch 
import tempfile
import os
from flask import current_app 


def preprocess_single_image(file):
    # Determine the file extension
    filename = file.filename
    ext = os.path.splitext(filename)[-1]  # Get the file extension (e.g., .nii, .nii.gz)

    # Define the path to the tmp directory
    tmp_dir = os.path.join(current_app.root_path, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)  # Ensure the tmp directory exists

    # Create a temporary file with the correct extension in the tmp directory
    with tempfile.NamedTemporaryFile(suffix=ext, dir=tmp_dir, delete=False) as tmp:
        file.save(tmp.name)
        temp_file_path = tmp.name

    try:
        # Load the image using nibabel
        image = nib.load(temp_file_path)
        image_data = image.get_fdata()

        # Check if the image has only one channel
        if image_data.ndim == 3:  # [D, H, W] -> Convert to [C, D, H, W]
            image_data = np.expand_dims(image_data, axis=0)  # Add channel dimension

        # Repeat the single channel to match the expected input channels (4) and add a dummy fifth channel
        input_data = np.repeat(image_data, 4, axis=0)  # Repeat along the channel axis
        dummy_channel = np.zeros_like(input_data[0])  # Create a zero-filled channel
        input_data = np.concatenate([input_data, np.expand_dims(dummy_channel, axis=0)], axis=0)  # Add dummy channel

        # Convert the image data to a tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

    return input_tensor


def preprocess_multiple_images(zip_ref):
    tmp_dir = os.path.join(current_app.root_path, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    # Extract the zip file into the tmp directory
    zip_ref.extractall(tmp_dir)

    # List of required modalities
    required_modalities = ['flair', 't1ce', 't1', 't2']
    modality_files = {modality: None for modality in required_modalities}

    # Process each extracted file
    for file_name in os.listdir(tmp_dir):
        for modality in required_modalities:
            if file_name.endswith(f"_{modality}.nii") or file_name.endswith(f"_{modality}.nii.gz"):
                modality_files[modality] = os.path.join(tmp_dir, file_name)

    # Check if all required modalities are present
    if not all(modality_files.values()):
        raise ValueError('ZIP file must contain all four modalities: flair, t1ce, t1, t2')

    images = []
    for modality, file_path in modality_files.items():
        try:
            # Load the image using nibabel
            img = nib.load(file_path)
            img_data = img.get_fdata()

            # Normalize and convert to float32
            img_data = (img_data - np.mean(img_data)) / np.std(img_data)
            images.append(img_data.astype(np.float32))

        except nib.filebasedimages.ImageFileError as e:
            print(f"Error loading image for modality {modality}: {e}")
            raise ValueError(f"Failed to load NIfTI image for modality '{modality}'. Please check the file format.")

    # Ensure all modalities have the same dimensions
    shapes = [img.shape for img in images]
    if len(set(shapes)) != 1:
        raise ValueError("All modalities must have the same dimensions")

    # Stack modalities into a single tensor
    image_stack = np.stack(images, axis=0)  # Stack along channel axis

    # Add a fifth dummy channel
    dummy_channel = np.zeros_like(image_stack[0])  # Create a zero-filled channel
    image_stack = np.concatenate([image_stack, np.expand_dims(dummy_channel, axis=0)], axis=0)

    image_tensor = torch.tensor(image_stack, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    return image_tensor

