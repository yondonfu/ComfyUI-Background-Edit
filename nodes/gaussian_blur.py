import torch

from torchvision.transforms import v2


class GaussianBlur:
    CATEGORY = "background-edit"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "kernel_size": ("INT", {"default": 61, "min": 1, "step": 2}),
                "sigma": ("INT", {"default": 5, "min": 1, "step": 2}),
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
        if mode != "cuda" and mode != "cpu":
            raise Exception("invalid mode")

        if mode == "cuda":
            images = images.to("cuda")

        gaussian_blur = v2.GaussianBlur((kernel_size, kernel_size), (sigma, sigma))
        # torchvision expects a BCHW tensor
        # Convert input BHWC -> BCHW
        blurred = gaussian_blur(images.permute(0, 3, 1, 2))

        # Comfy expects a BHWC tensor
        # Convert output BCHW -> BHWC
        return (blurred.permute(0, 2, 3, 1),)
