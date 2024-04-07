import os
from annotation import AnnotationFile, Annotation, Point
import pydicom
import numpy as np
from PIL import Image


def dicomToImage(dicomPath: str) -> Image:
    dicom = pydicom.dcmread(dicomPath)
    image = dicom.pixel_array.astype(float)
    rescale_image = (np.maximum(image, 0) / image.max()) * 255
    image = np.uint8(rescale_image)
    return Image.fromarray(image)


def boundingBox(points: list[Point]) -> str:
    min_x = min(point.x for point in points)
    min_y = min(point.y for point in points)
    max_x = max(point.x for point in points)
    max_y = max(point.y for point in points)

    x_center = (min_x + max_x) / 2
    y_center = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y

    return f"{x_center} {y_center} {width} {height}"


def annToLabels(annotations: list[Annotation], classes: dict):
    labels = []
    for ann in annotations:
        if ann.markers and ann.markers[0] in classes.values():
            classId = [k for k, v in classes.items() if v == ann.markers[0]][0]
            box = boundingBox(ann.points)
            labels.append(f"{classId} {box}\n")
    return labels


def createConfig():
    pass


def prepareData(classes: dict):
    trainImagePath = "./train/images/"
    trainLabelsPath = "./train/labels/"
    os.makedirs(os.path.dirname(trainImagePath), exist_ok=True)
    os.makedirs(os.path.dirname(trainLabelsPath), exist_ok=True)

    directory = "brezen"
    fileNames = [
        f
        for f in os.listdir(directory)
        if f.endswith(".json") and not f.startswith("config")
    ]

    for name in fileNames:
        annFile = AnnotationFile.from_json_file(f"{directory}/{name}")

        labels = annToLabels(annFile.annotations, classes)
        if labels:
            with open(f"{trainLabelsPath}{name}.txt", "w") as f:
                f.writelines(labels)

            image = dicomToImage(f"{directory}/{annFile.sourceFile}")
            image.save(trainImagePath + name + ".png")


def main():
    prepareData({0: "P4a: patchy opacity", 1: "P5a: linear opacity"})


if __name__ == "__main__":
    main()
