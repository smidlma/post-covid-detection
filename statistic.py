import json
import os


def show_dataset_statistic(dir_path):
    json_file_names = [
        dir_path + "/" + f
        for f in os.listdir(dir_path)
        if f.endswith(".json") and not f.startswith("config")
    ]

    json_files = []

    for file_name in json_file_names:
        with open(file_name, "r") as f:
            json_files.append(json.load(f))

    print(f"Number of samples: {len(json_files)}")
    multiple_markers = []
    attribute_counts = {}
    attribute_file_counts = {}

    for file in json_files:
        file_attributes = set()
        for annotation in file["annotations"]:
            attributes = annotation["markers"]
            for atr in attributes:
                if atr in attribute_counts:
                    attribute_counts[atr] += 1
                else:
                    attribute_counts[atr] = 1

                file_attributes.add(atr)

            if len(attributes) > 1:
                multiple_markers.append(attributes)
        # Update the count of unique files for each attribute
        for atr in file_attributes:
            if atr in attribute_file_counts:
                attribute_file_counts[atr] += 1
            else:
                attribute_file_counts[atr] = 1

    # print(multiple_markers)
    sorted_dict = dict(
        sorted(attribute_counts.items(), key=lambda item: item[1], reverse=True)
    )
    print(sorted_dict)

    sorted_file_dict = dict(
        sorted(attribute_file_counts.items(), key=lambda item: item[1], reverse=True)
    )
    print('\n')
    print(sorted_file_dict)

    # print(multiple_markers)


def show_description():
    pass


def main():
    dir_path = "./anotace/brezen"
    show_dataset_statistic(dir_path)


if __name__ == "__main__":
    main()
