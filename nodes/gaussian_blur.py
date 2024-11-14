import torch
import cvcuda
import cv2
import numpy as np


class GaussianBlur:
    CATEGORY = "background-edit"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "kernel_size": ("INT", {"default": 61}),
                "sigma": ("INT", {"default": 5}),
                "mode": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    def execute(
        self,
        images: torch.Tensor,
        kernel_size: int = 61,
        sigma: int = 5,
        mode: str = "cuda",
    ):
        if mode == "cpu":
            images_np = (
                (images * 255.0).clamp(0, 255).to(dtype=torch.uint8).cpu().numpy()
            )

            results = []
            for image in images_np:
                result = cv2.GaussianBlur(
                    image, (kernel_size, kernel_size), sigma, sigma
                )
                results.append(result)

            return (
                torch.from_numpy(np.stack(results, axis=0).astype(np.float32) / 255.0),
            )
        elif mode == "cuda":
            images = images.to("cuda")

            result = cvcuda.gaussian(
                cvcuda.as_tensor(images, "NHWC"),
                (kernel_size, kernel_size),
                (sigma, sigma),
            )
            return (torch.as_tensor(result.cuda()),)
        else:
            raise Exception("invalid mode")
