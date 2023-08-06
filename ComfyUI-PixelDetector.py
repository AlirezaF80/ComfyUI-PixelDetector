import numpy as np
from itertools import product
import torchvision.transforms as T
from PIL import Image
from scipy import signal
from torchvision.transforms import transforms


class PixelDetector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "palette": (["enable", "disable"],),
                "max_colors": ("INT", {"default": 128, "min": 2, "max": 4096, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pixelize"
    CATEGORY = "Pixel Art"

    def pixelize(self, image, palette, max_colors=128):
        downscale, hf, vf = self._pixel_detect(self._tensor_to_pil(image))
        output = downscale
        if palette == "enable":
            best_k = self._determine_best_k(downscale, max_colors)
            output = downscale.quantize(colors=best_k, method=1, kmeans=best_k, dither=0).convert('RGB')
        return (self._pil_to_tensor(output),)

    @staticmethod
    def _tensor_to_pil(tensor):
        tensor = tensor.squeeze(0)
        tensor = tensor.permute(2, 0, 1)
        transform = T.ToPILImage()
        # convert the tensor to PIL image using above transform
        img = transform(tensor)
        return img

    @staticmethod
    def _pil_to_tensor(image):
        convert_tensor = transforms.ToTensor()
        print(image)
        print(type(image))
        print(image.size)
        tensor = convert_tensor(image)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.permute(0, 2, 3, 1)
        return tensor

    def _kCentroid(self, image: Image, width: int, height: int, centroids: int):
        image = image.convert("RGB")

        # Create an empty array for the downscaled image
        downscaled = np.zeros((height, width, 3), dtype=np.uint8)

        # Calculate the scaling factors
        wFactor = image.width / width
        hFactor = image.height / height

        # Iterate over each tile in the downscaled image
        for x, y in product(range(width), range(height)):
            # Crop the tile from the original image
            tile = image.crop((x * wFactor, y * hFactor, (x * wFactor) + wFactor, (y * hFactor) + hFactor))

            # Quantize the colors of the tile using k-means clustering
            tile = tile.quantize(colors=centroids, method=1, kmeans=centroids).convert("RGB")

            # Get the color counts and find the most common color
            color_counts = tile.getcolors()
            most_common_color = max(color_counts, key=lambda x: x[0])[1]

            # Assign the most common color to the corresponding pixel in the downscaled image
            downscaled[y, x, :] = most_common_color

        return Image.fromarray(downscaled, mode='RGB')

    def _pixel_detect(self, image: Image):
        # Convert the image to a NumPy array
        npim = np.array(image)[..., :3]

        # Compute horizontal differences between pixels
        hdiff = np.sqrt(np.sum((npim[:, :-1, :] - npim[:, 1:, :]) ** 2, axis=2))
        hsum = np.sum(hdiff, 0)

        # Compute vertical differences between pixels
        vdiff = np.sqrt(np.sum((npim[:-1, :, :] - npim[1:, :, :]) ** 2, axis=2))
        vsum = np.sum(vdiff, 1)

        # Find peaks in the horizontal and vertical sums
        hpeaks, _ = signal.find_peaks(hsum, distance=1, height=0.0)
        vpeaks, _ = signal.find_peaks(vsum, distance=1, height=0.0)

        # Compute spacing between the peaks
        hspacing = np.diff(hpeaks)
        vspacing = np.diff(vpeaks)

        # Resize input image using kCentroid with the calculated horizontal and vertical factors
        return self._kCentroid(image, round(image.width / np.median(hspacing)),
                               round(image.height / np.median(vspacing)), 2), np.median(hspacing), np.median(vspacing)

    def _determine_best_k(self, image: Image, max_k: int):
        # Convert the image to RGB mode
        image = image.convert("RGB")

        # Prepare arrays for distortion calculation
        pixels = np.array(image)
        pixel_indices = np.reshape(pixels, (-1, 3))

        # Calculate distortion for different values of k
        distortions = []
        for k in range(1, max_k + 1):
            quantized_image = image.quantize(colors=k, method=0, kmeans=k, dither=0)
            centroids = np.array(quantized_image.getpalette()[:k * 3]).reshape(-1, 3)

            # Calculate distortions
            distances = np.linalg.norm(pixel_indices[:, np.newaxis] - centroids, axis=2)
            min_distances = np.min(distances, axis=1)
            distortions.append(np.sum(min_distances ** 2))

        # Calculate the rate of change of distortions
        rate_of_change = np.diff(distortions) / np.array(distortions[:-1])

        # Find the elbow point (best k value)
        if len(rate_of_change) == 0:
            best_k = 2
        else:
            elbow_index = np.argmax(rate_of_change) + 1
            best_k = elbow_index + 2

        return best_k


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "PixelDetector": PixelDetector
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelDetector": "PixelDetector"
}
