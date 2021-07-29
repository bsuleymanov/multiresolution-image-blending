import torch
from torch.nn import functional as F
import jpeg4py as jpeg
from pathlib import Path
import math
from matplotlib import pyplot as plt
import imageio


def image_to_tensor(image, device='cpu'):
    tensor = torch.tensor(image, dtype=torch.float32, device=device)\
                        .permute(2, 0, 1).unsqueeze(0) / 255.
    return F.interpolate(tensor, (256, 256), mode='bilinear')

def tensor_to_image(tensor):
    return tensor.squeeze().permute(1, 2, 0).cpu().numpy()

def get_gaussian_kernel(device=torch.device('cpu'), n_channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(n_channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def gaussian_blur(image, kernel_weight=1):
    kernel = get_gaussian_kernel(image.device, image.size(1)) * kernel_weight
    return F.conv2d(image, weight=kernel, groups=image.size(1), padding="same")

def upsample(image, scale_factor):
    image_upscaled = F.upsample_nearest(image, scale_factor=scale_factor)
    image_upscaled[:, :, :, 1::2] = 0.
    image_upscaled[:, :, 1::2, :] = 0.
    return gaussian_blur(image_upscaled, kernel_weight=scale_factor**2)

def downsample(image):
    image_downscaled = image[:, :, ::2, ::2]
    return gaussian_blur(image_downscaled)

def build_pyramid(image):
    g_pyramid = []
    l_pyramid = []
    pyramid_height = int(math.floor(math.log2(image.size(2))) - 1)
    for level in range(pyramid_height):
        g_pyramid.append(image)
        filtered_image = gaussian_blur(g_pyramid[level])
        downsampled_image = downsample(filtered_image)
        l_pyramid.append(g_pyramid[level] - upsample(downsampled_image, 2))
        image = downsampled_image
    return g_pyramid, l_pyramid

def collapse_pyramid(g_pyramid, l_pyramid):
    up_pyramid = []
    image = g_pyramid[len(g_pyramid) - 1]
    for j in range(len(l_pyramid) - 1):
        g_exp = upsample(image, 2)
        l = l_pyramid[len(l_pyramid) - j - 2]
        image = g_exp + l
        up_pyramid.append(image)
    return up_pyramid

def blend_images(image1, image2, device="cpu"):
    image1 = image_to_tensor(image1, device)
    image2 = image_to_tensor(image2, device)

    mask = torch.zeros((image1.shape[2], image1.shape[3], 3), dtype=torch.float32)\
                       .permute(2, 0, 1).unsqueeze(0)
    mask[:, :, :, mask.shape[3] // 2:] = 1

    g_pyramid_mask, l_pyramid_mask = build_pyramid(mask)
    g_pyramid1, l_pyramid1 = build_pyramid(image1)
    g_pyramid2, l_pyramid2 = build_pyramid(image2)


    pyramid_height = len(g_pyramid_mask)
    ls_pyr = []
    for i in range(pyramid_height):
        ls = (1 - g_pyramid_mask[i]) * l_pyramid1[i] + g_pyramid_mask[i] * l_pyramid2[i]
        ls_pyr.append(ls)
    image = (1 - g_pyramid_mask[-1]) * g_pyramid1[-1] + \
            g_pyramid_mask[-1] * g_pyramid2[-1]
    l = len(ls_pyr)
    gs_pyr = [image]
    pyramid_blended = collapse_pyramid(gs_pyr, ls_pyr)
    image_blended = pyramid_blended[-1]

    return image_blended

def main():
    image_dir = Path("./images")
    out_dir = Path("./output")
    image1 = jpeg.JPEG(image_dir / "orange.jpeg").decode()
    image2 = jpeg.JPEG(image_dir / "apple.jpeg").decode()
    composition = blend_images(image1, image2)
    composition = tensor_to_image(composition)
    plt.imshow(composition)
    plt.show()
    imageio.imsave(out_dir / "result.jpeg", composition)


if __name__ == "__main__":
    main()