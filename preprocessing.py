import os
from annotation import AnnotationFile, Annotation, Point
import pydicom
import numpy as np
from PIL import Image, ImageDraw
import albumentations as A
import datetime


class BBox:
    def __init__(self, x_center, y_center, width, height, category_name) -> None:
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.category_name = category_name

    def __str__(self) -> str:
        return f"{self.category_name} {self.x_center} {self.y_center} {self.width} {self.height}"

    def get_bbox(self):
        return [self.x_center, self.y_center, self.width, self.height]

    @staticmethod
    def from_points(points: list[Point], im_width, im_height, category_name):
        min_x = min(point.x for point in points)
        min_y = min(point.y for point in points)
        max_x = max(point.x for point in points)
        max_y = max(point.y for point in points)

        x_center = (min_x + max_x) / 2
        y_center = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y

        return BBox(
            x_center / im_width,
            y_center / im_height,
            width / im_width,
            height / im_height,
            category_name,
        )


class AnnotatedImage:
    def __init__(
        self, image: np.uint8, bboxes: list[BBox], category_names: list, name: str
    ) -> None:
        self.image = image
        self.bboxes = bboxes
        self.category_names = category_names
        self.name = name

    @staticmethod
    def from_annotations(
        image: np.uint8, annotations: list[Annotation], image_name: str
    ):
        bboxes = []
        category_names = []
        im_height, im_width = image.shape
        for ann in annotations:
            for category in ann.markers:
                bboxes.append(
                    BBox.from_points(ann.points, im_width, im_height, category)
                )
                category_names.append(category)
        return AnnotatedImage(image, bboxes, category_names, image_name)

    def visualize(self, categories: list[str]):
        im = Image.fromarray(self.image)
        draw = ImageDraw.Draw(im)

        indices = [i for i, x in enumerate(self.category_names) if x in categories]
        if indices:
            for idx in indices:
                bbox = self.bboxes[idx]
                # Denormalize the bounding box coordinates
                x_center = bbox.x_center * im.width
                y_center = bbox.y_center * im.height
                width = bbox.width * im.width
                height = bbox.height * im.height
                # Calculate the top left and bottom right points of the rectangle
                top_left = (x_center - width / 2, y_center - height / 2)
                bottom_right = (x_center + width / 2, y_center + height / 2)
                # Draw the rectangle on the image
                draw.rectangle([top_left, bottom_right], outline="red", width=3)
                draw.text((x_center, y_center), self.category_names[idx], fill="white")

            im.show()
        elif not categories:
            raise NotImplementedError("Just show for all bboxes")

    def save(self, im_path: str, label_path: str, categories: list):
        labels = []
        for bbox in self.bboxes:
            if bbox.category_name in categories:
                id = categories.index(bbox.category_name)
                bbox_format = " ".join(map(str, bbox.get_bbox()))
                labels.append(f"{id} {bbox_format}\n")

        if labels:
            with open(os.path.join(label_path, self.name + ".txt"), "w") as f:
                f.writelines(labels)

            im = Image.fromarray(self.image)
            im.save(os.path.join(im_path, self.name + ".png"))
            return True
        return False


class Augmentation:
    def __init__(self) -> None:
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.ColorJitter(p=0.7),
                A.RandomCrop(p=0.3, width=640, height=640),
                A.Solarize(p=0.7)
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )
        pass

    def augment_image(
        self, image: AnnotatedImage, categories: list, image_id: int
    ) -> AnnotatedImage:
        aug_name = f"{image.name}-aug-{image_id}"
        bboxes = list(
            map(
                lambda x: x.get_bbox(),
                filter(lambda x: x.category_name in categories, image.bboxes),
            )
        )
        class_labels = list(
            map(
                lambda x: x.category_name,
                filter(lambda x: x.category_name in categories, image.bboxes),
            )
        )

        aug = self.transform(
            image=image.image, bboxes=bboxes, class_labels=class_labels
        )

        return AnnotatedImage(
            aug["image"],
            [
                BBox(x[0], x[1], x[2], x[3], class_labels[idx])
                for idx, x in enumerate(aug["bboxes"])
            ],
            aug["class_labels"],
            aug_name,
        )


class Dicom:
    @staticmethod
    def dicom_to_image(file_path: str):
        dicom = pydicom.dcmread(file_path)
        image = dicom.pixel_array.astype(float)
        rescale_image = (np.maximum(image, 0) / image.max()) * 255
        image = np.uint8(rescale_image)
        return image


class Dataset:
    def __init__(
        self,
        input_dir: str,
    ) -> None:
        self.input_dir = input_dir
        self.aug = Augmentation()
        self.filenames = [
            f
            for f in os.listdir(self.input_dir)
            if f.endswith(".json") and not f.startswith("config")
        ]

        self.data: list[AnnotationFile] = sorted(
            [
                AnnotationFile.from_json_file(os.path.join(self.input_dir, file))
                for file in self.filenames
            ],
            key=lambda x: x.sourceFile,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> AnnotatedImage:
        if index >= len(self):
            raise IndexError
        file = self.data[index]
        image = Dicom.dicom_to_image(os.path.join(self.input_dir, file.sourceFile))
        return AnnotatedImage.from_annotations(
            image,
            file.annotations,
            file.sourceFile,
        )

    def create_yolo_dataset(self, target_dir, categories, aug_per_image=10):
        train_im_dir = os.path.join(target_dir, "train", "images/")
        train_lab_dir = os.path.join(target_dir, "train", "labels/")
        val_im_dir = os.path.join(target_dir, "val", "images/")
        val_lab_dir = os.path.join(target_dir, "val", "labels/")
        os.makedirs(os.path.dirname(train_im_dir), exist_ok=True)
        os.makedirs(os.path.dirname(train_lab_dir), exist_ok=True)
        os.makedirs(os.path.dirname(val_im_dir), exist_ok=True)
        os.makedirs(os.path.dirname(val_lab_dir), exist_ok=True)

        for image in self:
            is_saved = image.save(train_im_dir, train_lab_dir, categories)
            if is_saved:
                for id in range(aug_per_image):
                    im_aug = self.aug.augment_image(image, categories, id)
                    # im_aug.visualize(categories)
                    im_aug.save(train_im_dir, train_lab_dir, categories)


def main():
    dataset = Dataset("brezen")

    yolo_dir = "dataset"
    categories = ["P4a: patchy opacity"]
    dataset.create_yolo_dataset(yolo_dir, categories,  1)

    # for data in dataset:
    #     data.visualize(["P4a: patchy opacity"])


if __name__ == "__main__":
    main()
