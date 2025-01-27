from typing import Any, Optional, Union, Tuple

class Image:
    @staticmethod
    def open(fp: Union[str, bytes, Any], mode: str = "r") -> "Image": ...
    def save(self, fp: Union[str, bytes, Any], format: Optional[str] = None) -> None: ...
    # Add other type hints as needed 