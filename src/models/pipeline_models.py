from dataclasses import dataclass


@dataclass
class PlantMeasurement:
    """A single plant measurement for CSV output."""

    image_name: str
    plant_id: int
    length: int

    @property
    def plant_label(self) -> str:
        """Generate label like 'image_name_plant_1'."""
        return f"{self.image_name}_plant_{self.plant_id}"
