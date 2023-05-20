import os
from dotenv import load_dotenv
from collections import defaultdict
import supervisely as sly

# https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes?resource=download

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

workspace_id = int(os.environ["context.workspaceId"])
workspace = api.workspace.get_info_by_id(workspace_id)
if workspace is None:
    raise KeyError(f"Workspace with ID {workspace_id} not found in your account")
else:
    print(f"Workspace name = {workspace.name}, id = {workspace.id}")

project_path = r"C:\Users\German\Documents\weed\agri_data\data"

directory = os.path.dirname(project_path)
with open(r"C:\Users\German\Documents\weed\classes.txt") as names_file:
    names = names_file.read().split("\n")
# print(names)


def yolobbox2bbox(x, y, w, h):
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return x1, y1, x2, y2


def load_image_labels(image_path, labels_path):
    image_info = api.image.upload_path(
        dataset.id, os.path.basename(image_path), image_path
    )
    output = []
    with open(labels_path) as file:
        file_split = file.read().rstrip().split("\n")
    for row in file_split:
        if row == "":
            continue
        output.append(row.split())
    labels = []
    height = image_info.height
    width = image_info.width
    for bbox in output:
        c, x, y, w, h = bbox
        obj_class_name = names[int(c)]

        x1, y1, x2, y2 = yolobbox2bbox(float(x), float(y), float(w), float(h))
        bbox_annotation = sly.Rectangle(
            y1 * height, x1 * width, y2 * height, x2 * width
        )
        obj_class = meta.get_obj_class(obj_class_name)
        label = sly.Label(bbox_annotation, obj_class)
        labels.append(label)

        ann = sly.Annotation(img_size=[height, width], labels=labels)
        api.annotation.upload_ann(image_info.id, ann)
    print(f"Image (id:{image_info.id} has been successfully processed and uploaded.)")


# create project and initialize meta
project_name = os.path.basename(os.path.dirname(project_path))
project = api.project.get_info_by_name(workspace.id, project_name)
if project is not None:
    api.project.remove(project.id)
project = api.project.create(workspace.id, project_name)
meta = sly.ProjectMeta()

# create object classes
for name in names:
    if name == "":
        break
    obj_class = sly.ObjClass(name, sly.Rectangle)
    meta = meta.add_obj_class(obj_class)
    api.project.update_meta(project.id, meta)

image_path = sly.fs.list_files(project_path, valid_extensions=[".jpeg"])
dataset = api.dataset.create(project.id, os.path.basename(project_path))
# upload bboxes to images
for path in image_path:
    l_path = os.path.join(project_path, (os.path.basename(path)[:-5] + ".txt"))
    load_image_labels(path, l_path)
print(f"Dataset {dataset.id} has been successfully created.")
