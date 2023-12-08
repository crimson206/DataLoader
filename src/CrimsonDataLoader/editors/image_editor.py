import numpy as np
import torch

def give_coler_to_image(image_rgb, idx_to_take):
    #image_rgb = copy.deepcopy(image_rgb)
    black_mask = image_rgb == -1
    # Convert to numpy and create an RGB version
    full_indexes = [0, 1, 2]

    remained_indexes = [idx for idx in full_indexes if idx != idx_to_take]

    image_rgb[:, remained_indexes, :, :] = 0
    image_rgb[black_mask] = -1
    return image_rgb

def add_striped_pattern_to_image(image_rgb, angle):
    #image_rgb = copy.deepcopy(image_rgb)
    # Convert to numpy and repeat along the color channel to make it RGB
    black_mask = image_rgb == -1
    # Add stripes
    for i in range(0, image_rgb.shape[2], 4):
        if angle == 0:
            image_rgb[:, :,i:i+2,:] = 0.4 * image_rgb[:, :,i:i+2,:] # Gray stripes
        elif angle== 1:
            image_rgb[:, :, :, i:i+2] = 0.4 * image_rgb[:, :, :, i:i+2] # Gray stripes
        else:
            pass

    image_rgb[black_mask] = -1
    return image_rgb

def expand_image(image_grey:torch.Tensor):
    image = image_grey.repeat(1, 3, 1, 1)
    return image

def add_diversity_to_grey_image(dataloader):
    images = []
    labels = []

    for image, label in dataloader:
        
        batch_size = image.shape[0]

        random_color = np.random.randint(0, 4)
        random_striped = np.random.randint(0, 3)

        image = expand_image(image)
        if random_color!=3:
            image = give_coler_to_image(image_rgb=image, idx_to_take=random_color)
        if random_striped!=2:
            image = add_striped_pattern_to_image(image_rgb=image, angle=random_striped)

        color_label = random_color*torch.ones(batch_size, 1)
        striped_label = random_striped*torch.ones(batch_size, 1)

        tot_label = torch.cat([label.unsqueeze(-1), color_label, striped_label], dim=-1)

        images.append(image)
        labels.append(tot_label)
    images_cat = torch.cat(images, dim=0)
    labels_cat = torch.cat(labels, dim=0)

    tensor_dict = {
        "image":images_cat,
        "labels":labels_cat
    }

    return tensor_dict