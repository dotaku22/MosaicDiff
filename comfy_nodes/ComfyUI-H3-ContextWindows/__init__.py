from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

from .picture_video import MiniMaxH3PictureVideoReference
NODE_CLASS_MAPPINGS["MiniMaxH3PictureVideoReference"] = MiniMaxH3PictureVideoReference
