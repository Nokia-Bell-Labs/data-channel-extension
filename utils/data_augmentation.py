import math


random_mirror = True

def scale_to_new_range(m, min, max):

    old_range = (1 - 0)
    new_range = (max - min)
    new_value = (((m - 0) * new_range) / old_range) + min

    return new_value

def tensor_to_pil(tensor):

    return transforms.ToPILImage()(tensor.squeeze_(0))

def pil_to_tensor(pil):

    return transforms.ToTensor()(pil)

def Rotate_deterministic(img, v):  # [-30, 30]
    v = scale_to_new_range(v, -30, 30)
    assert -30 <= v <= 30
    return img.rotate(v)

class DataRotation:
    def __init__(self, target_channel):
        self.target_channel = target_channel
    def __call__(self, img):
        img_array = np.array(img)
        num_channel = img_array.shape[2]  # Assuming img is in HWC format
        num_samples = math.ceil(self.target_channel / num_channel) - 1 # assume the first element is the original image
        mag_list = np.linspace(0, 1, num_samples)
        result_img = np.array(img)
        for i in range(num_samples):
            augmented = Rotate_deterministic(img, v=mag_list[i])
            result_img = np.concatenate((result_img, np.array(augmented)), axis=-1)
        return result_img[:, :, :self.target_channel]

if __name__ == "__main__":
    import torchvision
    from torchvision import transforms, datasets
    import matplotlib.pyplot as plt
    import numpy as np

    # Assuming data_reshape is already imported and defined elsewhere

    # Initialize the transformation and the Caltech101 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a common size
    ])

    # Load Caltech101 dataset
    dataset = torchvision.datasets.Caltech101(root='../data/Caltech101', download=True, transform=transform)

    # Load a single image and its label
    img, label = dataset[0]  # Get the first image and label from the dataset

    # Initialize your custom reshape class
    # augmenter = DataAugmentation("000000000000001")  # Example: target to 64x64 image with 9 channels
    augmenter = DataRotation(target_channel=64)

    # Apply the reshaping to the image
    augmented_img = augmenter(img)

    # Plotting the original and augmented images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(np.array(img))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(np.array(augmented_img[:,:,3:6]))
    axes[1].set_title('Augmented Image')
    axes[1].axis('off')

    plt.show()