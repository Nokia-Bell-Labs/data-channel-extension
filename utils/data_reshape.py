import torch
import math
import torchvision

from PIL import Image
import numpy as np
import torch

def fractional_repeat(lst, N): # used for repeating MEAN and STDEV for normalization across 3> channels
    full_repeats = N // 3
    remainder = N % 3

    # Create the repeated list based on full repetitions
    result = lst * full_repeats

    # Append the remainder elements if needed
    if remainder > 0:
        result += lst[:remainder]

    return result



class DownsamplePIL:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        # Convert PIL image to a NumPy array
        img_array = np.array(img)
        num_channel = img_array.shape[2]  # Assuming img is in HWC format

        if (self.target_size == img_array.shape[0] and
            self.target_size == img_array.shape[1]):
            return img

        # Compute the new grid size
        height_ratio = img_array.shape[0] / self.target_size
        width_ratio = img_array.shape[1] / self.target_size

        # Create an array of indices for sampling
        row_indices = (np.arange(self.target_size) * height_ratio).astype(int)
        col_indices = (np.arange(self.target_size) * width_ratio).astype(int)

        # Use advanced indexing to sample the image
        reshaped_img = img_array[row_indices[:, None], col_indices, :]

        # Convert the reshaped image back to a PIL image
        reshaped_img = reshaped_img.astype(np.uint8)
        return Image.fromarray(reshaped_img)


class DownsampleTensor:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        num_channel = img.shape[0]
        if self.target_size == img.shape[1] and self.target_size == img.shape[2]:
            return img

        # Compute the new grid size
        height_ratio = img.shape[1] / self.target_size
        width_ratio = img.shape[2] / self.target_size

        # Create tensors of indices for sampling
        row_indices = torch.floor(torch.arange(self.target_size) * height_ratio).long()
        col_indices = torch.floor(torch.arange(self.target_size) * width_ratio).long()

        # Use advanced indexing to sample the image
        reshaped_img = img[:, row_indices[:, None], col_indices]

        return reshaped_img

class DownsampleTensorRepeat:
    def __init__(self, target_size, target_channel):
        self.target_size = target_size
        self.target_channel = target_channel
    def __call__(self, img):
        num_channel = img.shape[0]
        num_samples = math.ceil(self.target_channel / num_channel)
        if self.target_size == img.shape[1] and self.target_size == img.shape[2]:
            if self.target_channel == num_channel:
                return img
            else:
                return img.repeat(num_samples, 1, 1)[:self.target_channel, :, :]


        # Compute the new grid size
        height_ratio = img.shape[1] / self.target_size
        width_ratio = img.shape[2] / self.target_size

        # Create tensors of indices for sampling
        row_indices = torch.floor(torch.arange(self.target_size) * height_ratio).long()
        col_indices = torch.floor(torch.arange(self.target_size) * width_ratio).long()

        # Use advanced indexing to sample the image
        reshaped_img = img[:, row_indices[:, None], col_indices]

        # Repeat the reshaped image across the channel axis
        repeated_img = reshaped_img.repeat(num_samples, 1, 1)

        return repeated_img[:self.target_channel, :, :]

class DataReshape:
    def __init__(self, target_size, target_channel, method='dex'):
        self.target_size = target_size
        self.target_channel = target_channel
        self.method = method

    def __call__(self, img):
        if (self.target_channel == img.shape[0] and
            self.target_size == img.shape[1] and
            self.target_size == img.shape[2]):
            return img

        if self.method == 'dex' or self.method == 'skewed_sample':
            # Compute the new grid size and allocate output tensor
            height_ratio = img.shape[1] / self.target_size
            width_ratio = img.shape[2] / self.target_size
            reshaped_img = torch.zeros((self.target_channel, self.target_size, self.target_size))

            # Process each block using vectorized operations

            for i in range(self.target_size):
                for j in range(self.target_size):
                    start_row = int(i * height_ratio)
                    end_row = int(min((i + 1) * height_ratio, img.shape[1]))
                    start_col = int(j * width_ratio)
                    end_col = int(min((j + 1) * width_ratio, img.shape[2]))

                    block = img[:, start_row:end_row, start_col:end_col]
                    block = block.reshape(img.shape[0], -1)  # Flatten the block

                    # Adapt number of samples based on block size
                    num_samples = math.ceil(self.target_channel/img.shape[0])
                    num_samples = min(block.shape[1], num_samples)
                    if self.method == 'dex':
                        indices = torch.linspace(0, block.shape[1] - 1, steps=num_samples).long()
                    elif self.method == 'skewed_sample':
                        indices = torch.arange(num_samples).long()
                    sampled_data = block[:, indices].T.flatten()

                    # Pad if necessary to match target channels (when block.shape[1] < self.target_channels)
                    if sampled_data.size(0) < self.target_channel:
                        sampled_data = torch.cat((sampled_data, torch.zeros(self.target_channel - sampled_data.size(0))))

                    if self.method == 'dex' or self.method =='skewed_sample':
                        reshaped_img[:, i, j] = sampled_data[:self.target_channel]

        elif self.method == 'tile_per_channel':

            # Number of patches to create
            num_samples = math.ceil(self.target_channel / img.shape[0])

            # Calculate number of rows and columns for patches
            num_patches = math.ceil(math.sqrt(num_samples))
            num_rows = num_patches
            num_cols = num_patches

            img_height, img_width = img.shape[1], img.shape[2]

            patch_height = img_height // num_rows
            patch_width = img_width // num_cols

            patches = []

            for row in range(num_rows):
                for col in range(num_cols):
                    if len(patches) >= num_samples:
                        break
                    y = row * patch_height
                    x = col * patch_width

                    # Ensure the patch doesn't exceed image dimensions
                    if y + patch_height > img_height:
                        y = img_height - patch_height
                    if x + patch_width > img_width:
                        x = img_width - patch_width

                    patch = img[:, y:y + patch_height, x:x + patch_width]

                    # Downsample the patch to the target size
                    indices_y = torch.linspace(0, patch.shape[1] - 1, steps=self.target_size).long()
                    indices_x = torch.linspace(0, patch.shape[2] - 1, steps=self.target_size).long()
                    downsampled_patch = patch[:, indices_y][:, :, indices_x]

                    patches.append(downsampled_patch)

            # Concatenate patches along the channel dimension
            reshaped_img = torch.cat(patches, dim=0).view(-1, self.target_size, self.target_size)[:self.target_channel,:,:]
        else:
            raise NotImplementedError
        return reshaped_img

if __name__ == "__main__":
    from torchvision import transforms, datasets
    import matplotlib.pyplot as plt
    import numpy as np

    # Assuming data_reshape is already imported and defined elsewhere

    # Initialize the transformation and the Caltech101 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a common size
        transforms.ToTensor()  # Convert images to tensor
    ])

    # Load Caltech101 dataset
    dataset = torchvision.datasets.Caltech101(root='../data/Caltech101', download=True, transform=transform)

    # Load a single image and its label
    img, label = dataset[0]  # Get the first image and label from the dataset

    # Initialize your custom reshape class
    # reshaper = DataReshape(target_size=32, target_channel=64)  # Example: target to 64x64 image with 9 channels
    reshaper = DownsampleTensorRepeat(target_size=32, target_channel=64)  # Example: target to 64x64 image with 9 channels

    # Apply the reshaping to the image
    reshaped_img = reshaper(img)
    print(img.shape)
    print(reshaped_img.shape)


    # Function to display images
    def show_tensor_images(original, transformed):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # Original image
        ax[0].imshow(np.transpose(original.numpy(), (1, 2, 0)))
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        # Transformed image
        # Since reshaped_img may have more than 3 channels, just show the first 3 for visualization
        ax[1].imshow(np.transpose(transformed.numpy()[3:6], (1, 2, 0)))
        ax[1].set_title('Transformed Image')
        ax[1].axis('off')

        plt.show()


    # Display the images
    show_tensor_images(img, reshaped_img)
