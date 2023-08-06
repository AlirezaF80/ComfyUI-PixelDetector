# ComfyUI - Pixel Detector
Downscale and restore pixel art images that have been resized or corrupted.

This is primarily created to fix compression issues from saving pixel art in jpg format.

## Exaggerated example

![Example](https://github.com/Astropulse/pixeldetector/assets/61034487/f8ae2802-42c1-4dba-af56-fe849ac8915c)

# Installation
Download the `ComfyUI-PixelDetector.py` into `custom_nodes` folder.

# Usage
Palette: If enabled, automatically reduces the image to predicted color palette.
Max Colors: Max colors for computation, more = slower, default = 128

# Credits
Thanks to @Astropulse and @paultron for the downscaling logic. I just wrapped it as is into a ComfyUI custom node.
Big thanks to https://github.com/paultron for numpy-ifying the downscale calculation and making it tons faster.

Test image by Skeddles https://lospec.com/gallery/skeddles/rock-and-grass
