import torch
import cvcuda
import cv2
import numpy as np


class Composite:
    CATEGORY = "background-edit"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "foregrounds": ("IMAGE",),
                "backgrounds": ("IMAGE",),
                "foreground_masks": ("MASK",),
                "mode": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    def execute(
        self,
        backgrounds: torch.Tensor,
        foregrounds: torch.Tensor,
        foreground_masks: torch.Tensor,
        mode: str = "cuda",
    ):
        if (
            backgrounds.shape[0] != foregrounds.shape[0]
            or backgrounds.shape[0] != foreground_masks.shape[0]
        ):
            raise Exception(
                "mismatch number of backgrounds, foregrounds and foreground masks"
            )

        if mode == "cpu":
            inverse_masks = 1.0 - foreground_masks

            fgs = foregrounds * foreground_masks.unsqueeze(-1)
            bgs = backgrounds * inverse_masks.unsqueeze(-1)

            fgs_np = (fgs * 255.0).clamp(0, 255).to(dtype=torch.uint8).cpu().numpy()
            bgs_np = (bgs * 255.0).clamp(0, 255).to(dtype=torch.uint8).cpu().numpy()

            results = []
            for fg, bg in zip(fgs_np, bgs_np):
                result = cv2.add(fg, bg)
                results.append(result)

            return (
                torch.from_numpy(np.stack(results, axis=0).astype(np.float32) / 255.0),
            )
        elif mode == "cuda":
            foregrounds = foregrounds.to("cuda")
            backgrounds = backgrounds.to("cuda")
            foreground_masks = foreground_masks.to("cuda")

            fgs = cvcuda.convertto(
                cvcuda.as_tensor(foregrounds, "NHWC"), np.uint8, scale=255
            )
            bgs = cvcuda.convertto(
                cvcuda.as_tensor(backgrounds, "NHWC"), np.uint8, scale=255
            )
            fgmasks = cvcuda.convertto(
                cvcuda.as_tensor(foreground_masks.unsqueeze(-1), "NHWC"),
                np.uint8,
                scale=255,
            )
            result = cvcuda.composite(fgs, bgs, fgmasks, 3)

            return (torch.as_tensor(result.cuda()) / 255.0,)
        else:
            raise Exception("invalid mode")
