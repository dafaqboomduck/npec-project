from dataclasses import dataclass

@dataclass
class PlantResult:
    """Container for individual plant measurement results.
    
    Attributes:
        image_name: Base name of the source image (without extension).
        plant_id: Zero-indexed identifier for the plant within the image.
        length: Measured root length in pixels.
    """
    image_name: str
    plant_id: int
    length: int

    @property
    def plant_label(self) -> str:
        """Return human-readable plant identifier (1-indexed)."""
        return f"{self.image_name}_plant_{self.plant_id + 1}"