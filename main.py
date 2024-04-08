import os
from annotation import AnnotationFile, Annotation, Point
import pydicom
import numpy as np
from PIL import Image
from ultralytics import YOLO


def dicomToImage(dicomPath: str) -> Image:
    dicom = pydicom.dcmread(dicomPath)
    image = dicom.pixel_array.astype(float)
    rescale_image = (np.maximum(image, 0) / image.max()) * 255
    image = np.uint8(rescale_image)
    return Image.fromarray(image)


def boundingBox(points: list[Point], imgWidth, imgHeight) -> str:
    min_x = min(point.x for point in points)
    min_y = min(point.y for point in points)
    max_x = max(point.x for point in points)
    max_y = max(point.y for point in points)

    x_center = (min_x + max_x) / 2
    y_center = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y

    return f"{x_center / imgWidth} {y_center / imgHeight} {width / imgWidth} {height / imgHeight}"


def annToLabels(
    annotations: list[Annotation], classes: list[str], imgWidth: float, imgHeight: float
):
    labels = []
    for ann in annotations:
        if ann.markers and ann.markers[0] in classes:
            classId = classes.index(ann.markers[0])
            box = boundingBox(ann.points, imgWidth, imgHeight)
            labels.append(f"{classId} {box}\n")
    return labels


def createConfig():
    pass


def prepareData(
    classes: list[str],
    trainImgDir: str,
    trainLblDir: str,
    valImgDir: str,
    valLblDir: str,
):
    os.makedirs(os.path.dirname(trainImgDir), exist_ok=True)
    os.makedirs(os.path.dirname(trainLblDir), exist_ok=True)
    os.makedirs(os.path.dirname(valImgDir), exist_ok=True)
    os.makedirs(os.path.dirname(valLblDir), exist_ok=True)

    directory = "brezen"
    fileNames = [
        f
        for f in os.listdir(directory)
        if f.endswith(".json") and not f.startswith("config")
    ]

    filesCount = 0
    for name in fileNames:
        annFile = AnnotationFile.from_json_file(f"{directory}/{name}")

        image = dicomToImage(f"{directory}/{annFile.sourceFile}")
        imgWidth, imgHeight = image.size
        labels = annToLabels(annFile.annotations, classes, imgWidth, imgHeight)
        if labels:
            with open(f"{trainLblDir}{name}.txt", "w") as f:
                f.writelines(labels)

            image.save(trainImgDir + name + ".png")

            if filesCount < 2:
                os.rename(f"{trainLblDir}{name}.txt", f"{valLblDir}{name}.txt")
                os.rename(f"{trainImgDir}{name}.png", f"{valImgDir}{name}.png")
            filesCount += 1


def main():
    datasetDir = "./dataset/"
    imagesTrain = f"{datasetDir}train/images/"
    labelsTrain = f"{datasetDir}train/labels/"
    imagesValidate = f"{datasetDir}val/images/"
    labelsValidate = f"{datasetDir}val/labels/"

    classes = ["P4a: patchy opacity", "P5a: linear opacity"]

    prepareData(classes, imagesTrain, labelsTrain, imagesValidate, labelsValidate)


if __name__ == "__main__":
    main()
