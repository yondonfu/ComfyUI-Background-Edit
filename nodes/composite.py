import torch


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

        if mode != "cuda" and mode != "cpu":
            raise Exception("invalid mode")

        if mode == "cuda":
            foregrounds = foregrounds.to("cuda")
            backgrounds = backgrounds.to("cuda")
            foreground_masks = foreground_masks.to("cuda")

        inverse_masks = 1.0 - foreground_masks

        fgs = foregrounds * foreground_masks.unsqueeze(-1)
        bgs = backgrounds * inverse_masks.unsqueeze(-1)

        results = fgs + bgs

        return (results,)
