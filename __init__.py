from .nodes.background_color import BackgroundColor
from .nodes.gaussian_blur import GaussianBlur
from .nodes.composite import Composite

NODE_CLASS_MAPPINGS = {
    "BackgroundColor": BackgroundColor,
    "GaussianBlur": GaussianBlur,
    "Composite": Composite,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
