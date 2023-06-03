import os
from dotenv import load_dotenv
import supervisely as sly
import csv

# https://www.kaggle.com/datasets/frettapper/micropcb-images

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

workspace_id = int(os.environ["context.workspaceId"])
workspace = api.workspace.get_info_by_id(workspace_id)
if workspace is None:
    raise KeyError(f"Workspace with ID {workspace_id} not found in your account")
else:
    print(f"Workspace name = {workspace.name}, id = {workspace.id}")

datasets = [
    r"C:\Users\German\Documents\PCB\test_coded\test_coded",
    r"C:\Users\German\Documents\PCB\train_coded\train_coded",
]


def conv(x, y, w, h):
    x2, y2 = x + w, y + h
    x1, y1 = x, y
    return x1, y1, x2, y2


def load_image_labels(image_path, dict_value, tags=list):
    image_info = api.image.upload_path(
        dataset.id, os.path.basename(image_path), image_path
    )

    labels = []
    height = image_info.height
    width = image_info.width

    x, y, w, h = dict_value

    x1, y1 = int(x) + int(w), int(y) + int(h)

    # x, y, x1, y1 = conv(float(x), float(y), float(w), float(h))
    bbox_annotation = sly.Rectangle(int(y), int(x), y1, x1)

    obj_class = meta.get_obj_class(names[ord(img_name[0]) - 65])
    label = sly.Label(bbox_annotation, obj_class)
    labels.append(label)

    ann = sly.Annotation(img_size=[height, width], labels=labels, img_tags=tags)
    api.annotation.upload_ann(image_info.id, ann)
    print(f"Image (id:{image_info.id} has been successfully processed and uploaded.)")


# create project and initialize meta
project_name = os.path.basename(os.path.dirname(os.path.dirname(datasets[0])))
project = api.project.get_info_by_name(workspace.id, project_name)
if project is not None:
    api.project.remove(project.id)
project = api.project.create(workspace.id, project_name)
meta = sly.ProjectMeta()

# create object classes
names = [
    "Raspberry Pi A+",
    "Arduino Mega 2560 (Blue)",
    "Arduino Mega 2560 (Black)",
    "Arduino Mega 2560 (Black and Yellow)",
    "Arduino Due",
    "Beaglebone Black",
    "Arduino Uno (Green)",
    "Raspberry Pi 3 B+",
    "Raspberry Pi 1 B+",
    "Arduino Uno Camera Shield",
    "Arduino Uno (Black)",
    "Arduino Uno WiFi Shield",
    "Arduino Leonardo",
]

for name in names:
    obj_class = sly.ObjClass(name, sly.Rectangle)
    meta = meta.add_obj_class(obj_class)
api.project.update_meta(project.id, meta)

for dataset_path in datasets:
    image_path = sly.fs.list_files(dataset_path, valid_extensions=[".jpg"])
    dataset = api.dataset.create(project.id, os.path.basename(dataset_path))

    dict_bboxes = {}

    folder_name_dataset = os.path.basename(dataset_path)
    folder_name_split = folder_name_dataset.split("_")

    with open(
        os.path.join(
            r"C:\Users\German\Documents\PCB", (folder_name_split[0] + "_bboxes.csv")
        )
    ) as file:
        file_read = csv.reader(file, delimiter=",", lineterminator="\r\n")
        next(file_read)
        for row in file_read:
            if row == "":
                continue
            dict_bboxes[row[0]] = row[1], row[2], row[3], row[4]

    # upload bboxes to images
    for path in image_path:
        img_name = sly.fs.get_file_name_with_ext(path)
        dict_bboxes.get(img_name)
        dict_value = dict_bboxes.get(img_name)

        tag_value_type = sly.TagValueType.ANY_STRING
        tag_names = "Angle"
        tag_meta = meta.get_tag_meta(tag_name)
        if tag_meta is None:
            tag_meta = sly.TagMeta(tag_name, tag_value_type)
            meta = meta.add_tag_meta(tag_meta)
            api.project.update_meta(dataset.project_id, meta)

        # add tag to image
        tag = sly.Tag(tag_meta, row[2])

        try:
            load_image_labels(path, dict_value)
        except Exception as e:
            print(e)
            continue
    print(f"Dataset {dataset.id} has been successfully created.")
