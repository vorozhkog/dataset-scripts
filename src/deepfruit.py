import os
from dotenv import load_dotenv
from collections import defaultdict
import supervisely as sly

# https://zenodo.org/record/6324489#.ZGJAM3bMIuV?

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

workspace_id = int(os.environ["context.workspaceId"])
workspace = api.workspace.get_info_by_id(workspace_id)
if workspace is None:
    raise KeyError(f"Workspace with ID {workspace_id} not found in your account")
else:
    print(f"Workspace name = {workspace.name}, id = {workspace.id}")

projects = [
    "/Users/germanvorozko/Downloads/yolov5/capsicum/valid",
    "/Users/germanvorozko/Downloads/yolov5/mango/valid",
    "/Users/germanvorozko/Downloads/yolov5/blueberry/valid",
]


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
        output.append(row.split())
    labels = []
    height = image_info.height
    width = image_info.width
    for bbox in output:
        c, x, y, w, h = bbox
        x1, y1, x2, y2 = yolobbox2bbox(float(x), float(y), float(w), float(h))
        bbox_annotation = sly.Rectangle(
            y1 * height, x1 * width, y2 * height, x2 * width
        )
        obj_class = meta.get_obj_class(project_name)
        label = sly.Label(bbox_annotation, obj_class)
        labels.append(label)

        ann = sly.Annotation(img_size=[height, width], labels=labels)
        api.annotation.upload_ann(image_info.id, ann)
    print(f"Image (id:{image_info.id} has been successfully processed and uploaded.)")


for project_directory in projects:
    project_name = os.path.basename(os.path.dirname(project_directory))
    project = api.project.get_info_by_name(workspace.id, project_name)
    if project is not None:
        api.project.remove(project.id)
    project = api.project.create(workspace.id, project_name)
    meta = sly.ProjectMeta()

    image_path = sly.fs.list_files_recursively(
        project_directory, valid_extensions=[".jpg"]
    )
    dataset = api.dataset.create(project.id, os.path.basename(project_directory))
    obj_class = sly.ObjClass(project_name, sly.Rectangle)
    meta = meta.add_obj_class(obj_class)
    api.project.update_meta(project.id, meta)
    for path in image_path:
        l_path = (
            project_directory
            + "/labels/"
            + os.path.splitext(os.path.basename(path))[0]
            + ".txt"
        )
        load_image_labels(path, l_path)
    print(f"Dataset {dataset.id} has been successfully created.")
