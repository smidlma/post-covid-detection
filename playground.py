from annotation import Annotation, AnnotationFile
import json
from serde.json import from_dict
import pydicom
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import io
import numpy as np

from preprocessing import Dataset

# def find_marker(dir_path):
#         self.filenames = [
#             f
#             for f in os.listdir(self.input_dir)
#             if f.endswith(".json") and not f.startswith("config")
#         ]


def show_image(path: str, labels: list[str]):
    annotationFile = AnnotationFile.from_json_file(path)
    dicom = pydicom.dcmread("./anotace/brezen/" + annotationFile.sourceFile)
    print(dicom.pixel_array.shape)
    image = dicom.pixel_array.astype(float)

    rescale_im = (np.maximum(image, 0) / image.max()) * 255
    image = np.uint8(rescale_im)
    print(image.shape)

    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 640, 640], outline=255, width=3)

    for annotation in annotationFile.annotations:
        for label in labels:
            if label in annotation.markers:
                point_tuples = [(point.x, point.y) for point in annotation.points]

                min_x = min(point[0] for point in point_tuples)
                min_y = min(point[1] for point in point_tuples)
                max_x = max(point[0] for point in point_tuples)
                max_y = max(point[1] for point in point_tuples)

                x_center = (min_x + max_x) / 2
                y_center = (min_y + max_y) / 2
                width = max_x - min_x
                height = max_y - min_y

                top_left = (x_center - width / 2, y_center - height / 2)
                bottom_right = (x_center + width / 2, y_center + height / 2)
                draw.rectangle([top_left, bottom_right], outline=255, width=3)

    image.show()

    # Display the DICOM image
    plt.imshow(dicom.pixel_array, cmap="gray")
    plt.title("X-ray Scan of Lungs")

    for annotation in annotationFile.annotations:
        for label in labels:
            if label in annotation.markers:
                x_values = [point.x for point in annotation.points]
                y_values = [point.y for point in annotation.points]
                #   plt.imshow(dicom_data.pixel_array, cmap='gray')
                plt.plot(x_values, y_values, color="red")
                plt.text(
                    x_values[0] + 10, y_values[0] - 10, f"TypeID: {annotation.type}"
                ),
    plt.show()


def main():
    file_path = "./anotace/brezen/142704_4012344_1.2.840.113564.10.1.28394376252560817261153186159151834210662.dcm"
    # show_image(file_path, ["P3: consolidation"])
    dataset = Dataset("./anotace/brezen")

    for data in dataset:
        if data.visualize(["P5d: reticulonodular pattern"], True): 
            # data.crop_area()
            pass
            

    # dicom = pydicom.dcmread(file_path)
    # print(dicom)

    # lines = dicom.__str__().count('\n')
    # print(lines)



if __name__ == "__main__":
    main()
