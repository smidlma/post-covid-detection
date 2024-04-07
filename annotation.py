from dataclasses import dataclass
from serde import serde
from serde.json import from_json, from_dict
import json
import matplotlib.pyplot as plt


@serde
@dataclass
class Point:
    x: float | int
    y: float | int


@serde
@dataclass
class Annotation:
    description: str
    label: str
    markers: list[str]
    points: list[Point]
    type: int

    # def plot_annotations(self, plot:plt):
    #     x_values = [point.x for point in self.points]  
    #     y_values = [point.y for point in self.points]
    #     plot.plot(x_values, y_values, color='red')
    #     plot.show() 
    


@serde
@dataclass
class AnnotationFile:
    annotations: list[Annotation]
    sourceFile: str

    @staticmethod
    def from_json_file(file_path: str) -> "AnnotationFile":
        with open(file_path, "r") as json_file:
            data = json_file.read()
        # json_data = json.loads(data)
        # return from_dict(AnnotationFile, json_data)
            return from_json(AnnotationFile, data)
    
    # def plot_all_annotations(self, plot: plt):
    #     for ann in self.annotations:
    #         ann.plot_annotations(plot)
