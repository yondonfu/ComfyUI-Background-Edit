import torch


class BackgroundColor:
    CATEGORY = "background-edit"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "color": (["black", "red", "blue", "green"], {"default": "black"}),
            }
        }

    def execute(
        self,
        images: torch.Tensor,
        color: str = "black",
    ):
        result = torch.zeros_like(images)

        if color == "red":
            result[..., 0] = 1.0
        elif color == "green":
            result[..., 1] = 1.0
        elif color == "blue":
            result[..., 2] = 1.0

        return (result,)
