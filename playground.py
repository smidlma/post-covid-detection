from annotation import Annotation, AnnotationFile
import json
from serde.json import from_dict
import pydicom
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import io
import numpy as np


def extractClasses():
    with open('brezen/config.json', 'r') as f:
        config = json.load(f)
    markers = config['annotationMarkers']

    classes = {}
    for idx, marker in  enumerate(markers):
        classes[idx] = marker['name']
        # "".repla

    return classes

def prepare(annFile: AnnotationFile, filter:list[str]):
    pass

def main():
    print(extractClasses())
    print("Hello, World!")
    file_path = "brezen/142704_4012344_1.2.840.113564.10.1.28394376252560817261153186159151834210662.json"
    annotationFile = AnnotationFile.from_json_file(file_path)
    # print(annotationFile.sourceFile)

    im = pydicom.dcmread("brezen/" + annotationFile.sourceFile)
    print(im.pixel_array.shape)
    im = im.pixel_array.astype(float)

    rescale_im = (np.maximum(im, 0) / im.max()) * 255
    image = np.uint8(rescale_im)
    print(image.shape)

    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for annotation in annotationFile.annotations:
        if "P4a: patchy opacity" in annotation.markers:
            point_tuples = [(point.x, point.y) for point in annotation.points]
            # draw.line(xy=point_tuples, fill='black', width=2)
            
            min_x = min(point[0] for point in point_tuples)
            min_y = min(point[1] for point in point_tuples)
            max_x = max(point[0] for point in point_tuples)
            max_y = max(point[1] for point in point_tuples)
            # min_point = min(annotation.points, key=lambda point: (point.x, point.y))
            # max_point = max(annotation.points, key=lambda point: (point.x, point.y))

            # min_x, min_y = min_point.x, min_point.y
            # max_x, max_y = max_point.x, max_point.y

            x_center = (min_x + max_x) / 2
            y_center = (min_y + max_y) / 2
            width = max_x - min_x
            height = max_y - min_y
            
            top_left = (x_center - width / 2, y_center - height / 2)
            bottom_right = (x_center + width / 2, y_center + height / 2)

            # draw.polygon(xy=point_tuples, fill='black', width=2)
            draw.rectangle([top_left, bottom_right], outline=255, width=3)


            # plt.text(
            #     x_values[0] + 10,
            #     y_values[0] - 10,
            #     f"TypeID: {annotation.type}", {'size': 6})
    # ann1 = annotationFile.annotations[12]
    # ann1.points
    # point_tuples = [(point.x, point.y) for point in ann1.points]
    # draw.line(xy=point_tuples, fill='black', width=10)

    image.show()
    # dataset = pydicom.dcmread("brezen/" + annotationFile.sourceFile)
    # print(dataset)
    # original_size = (dataset.Columns, dataset.Rows)
    # dataset = dataset.pixel_array.astype(float)
    # rescale_im = (np.maximum(dataset, 0) / dataset.max()) * 255
    # im_uint8 = np.uint8(rescale_im)
    # image = Image.fromarray(im_uint8)
    # new_size = image.size
    # x_scale = new_size[0] / original_size[0]
    # y_scale = new_size[1] / original_size[1]

    # draw = ImageDraw.Draw(image)
    # ann1 = annotationFile.annotations[3]
    # point_tuples = [(point.x, point.y) for point in ann1.points]

    # draw.line(point_tuples, fill=128, width=10)
    # image.show()

    # Load the DICOM file
    dicom_file_path = "brezen/" + annotationFile.sourceFile
    dicom_data = pydicom.dcmread(dicom_file_path)

    # annotationFile.plot_all_annotations(plot)
    # x_values = [point.x for point in ann1.points]
    # y_values = [point.y for point in ann1.points]

    # Display the DICOM image
    plt.imshow(dicom_data.pixel_array, cmap="gray")
    plt.title("X-ray Scan of Lungs")

    print("asdf")
    for annotation in annotationFile.annotations:
        # if annotation.type == 2:
        if "P4a: patchy opacity" in annotation.markers:
            x_values = [point.x for point in annotation.points]
            y_values = [point.y for point in annotation.points]
            #   plt.imshow(dicom_data.pixel_array, cmap='gray')
            plt.plot(x_values, y_values, color="red")
            plt.text(
                x_values[0] + 10,
                y_values[0] - 10,
                f"TypeID: {annotation.type}",
                {"size": 6},
            )
    #   plt.show()

    # plt.plot(x_values, y_values, color='red')
    # for point in ann1.points:
    #     # print(point)
    #     plt.plot(point.x, point.y, color='red', linewidth=2)  # Assuming x, y are coordinates of points
    plt.show()
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)

    # img = Image.open(buf)
    # img.show()
    # buf.close()


if __name__ == "__main__":
    main()
