import torch
import cv2
import numpy as np


class DepthBackgroundBlur:
    CATEGORY = "background-edit"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "depth_maps": ("IMAGE",),
                "blur_strength": ("INT", {"default": 51}),
                "threshold": ("INT", {"default": 125}),
            }
        }

    def execute(
        self,
        images: torch.Tensor,
        depth_maps: torch.Tensor,
        blur_strength: int,
        threshold: int,
    ):
        if images.shape[0] != depth_maps.shape[0]:
            raise Exception("mismatch number of images and depth maps")

        images_np = (images * 255.0).clamp(0, 255).to(dtype=torch.uint8).cpu().numpy()
        depth_maps_np = (
            (depth_maps * 255.0).clamp(0, 255).to(dtype=torch.uint8).cpu().numpy()
        )

        results = []
        for image, depth_map in zip(images_np, depth_maps_np):
            # Convert to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            binary_mask = (depth_map > threshold).astype(np.uint8) * 255
            blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

            binary_mask_3ch = binary_mask / 255.0
            inverse_mask_3ch = 1.0 - binary_mask_3ch

            foreground = image * binary_mask_3ch
            background = blurred * inverse_mask_3ch
            result = cv2.add(foreground, background)

            # Convert to RGB for torch
            result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)

            results.append(result)

        return (torch.from_numpy(np.stack(results, axis=0).astype(np.float32) / 255.0),)
