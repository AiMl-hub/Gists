import os

# Define the directories for labels and images
base_dir = "/medical_dataset"
labels_dir = os.path.join(base_dir, "labels")
images_dir = os.path.join(base_dir, "images")
new_labels_dir = os.path.join(base_dir, "newlabels")
os.makedirs(new_labels_dir, exist_ok=True)

# List all cases based on label files
cases = sorted(
    {file.split("_")[1] for file in os.listdir(labels_dir) if file.endswith(".nrrd")}
)

import SimpleITK as sitk
import numpy as np

for case in cases:
    # Filter label files for the current case
    label_files = [
        f
        for f in os.listdir(labels_dir)
        if f.startswith(f"case_{case}") and f.endswith(".nrrd")
    ]

    # Initialize combined label with zeros
    first_label = sitk.ReadImage(os.path.join(labels_dir, label_files[0]))
    combined_label = sitk.Image(first_label.GetSize(), sitk.sitkUInt8)
    combined_label.CopyInformation(first_label)

    # Assign unique intensity values to each label
    for i, label_file in enumerate(label_files):
        label_img = sitk.ReadImage(os.path.join(labels_dir, label_file))
        label_np = sitk.GetArrayFromImage(label_img)

        # Add to combined label with unique intensity (i+1)
        combined_label_np = sitk.GetArrayFromImage(combined_label)
        combined_label_np[label_np > 0] = i + 1

        combined_label = sitk.GetImageFromArray(combined_label_np)
        combined_label.CopyInformation(label_img)

    # Save the combined label as .nrrd or .nii file
    sitk.WriteImage(combined_label, f"{new_labels_dir}/case_{case}.nrrd")
