import json
import os


def main():
    directory = "brezen"
    json_file_names = [
        directory + "/" + f
        for f in os.listdir(directory)
        if f.endswith(".json") and not f.startswith("config")
    ]

    json_files = []

    for file_name in json_file_names:
        with open(file_name, "r") as f:
            json_files.append(json.load(f))

    print(f"Number of samples: {len(json_files)}")
    multiple_markers = []
    attribute_counts = {}
    for file in json_files:
        for annotation in file["annotations"]:
            attribute = annotation["markers"]
            if len(attribute) > 0:
                if attribute[0] in attribute_counts:
                    attribute_counts[attribute[0]] += 1
                else:
                    attribute_counts[attribute[0]] = 1
            if len(attribute) > 1:
                multiple_markers.append(attribute)

    # print(multiple_markers)
    sorted_dict = dict(
        sorted(attribute_counts.items(), key=lambda item: item[1], reverse=True)
    )
    print(sorted_dict)


if __name__ == "__main__":
    main()
