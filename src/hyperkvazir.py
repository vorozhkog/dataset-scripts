import os
from dotenv import load_dotenv
from collections import defaultdict
import supervisely as sly
import cv2
import json

# https://osf.io/mh9sj/

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

workspace_id = int(os.environ["context.workspaceId"])
workspace = api.workspace.get_info_by_id(workspace_id)
if workspace is None:
    raise KeyError(f"Workspace with ID {workspace_id} not found in your account")
else:
    print(f"Workspace name = {workspace.name}, id = {workspace.id}")

dataset_path = r"C:\Users\German\Documents\hyper-kvasir-segmented-images\images"


def load_image_labels(labels_path):
    mask = cv2.imread(labels_path[0], cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("", mask)
    height = image_info.height
    width = image_info.width
    bitmap_annotation = sly.Bitmap(
        data=mask,
    )
    obj_class = meta.get_obj_class(class_name)
    label = sly.Label(bitmap_annotation, obj_class)

    ann = sly.Annotation(img_size=[height, width], labels=[label])
    api.annotation.upload_ann(image_info.id, ann)


# create project and initialize meta
project_name = os.path.basename(os.path.dirname(dataset_path))
project = api.project.get_info_by_name(workspace.id, project_name)
if project is not None:
    api.project.remove(project.id)
project = api.project.create(workspace.id, project_name)
meta = sly.ProjectMeta()

bboxes_json = json.load(
    open(r"C:\Users\German\Documents\hyper-kvasir-segmented-images\bounding-boxes.json")
)

dataset = api.dataset.create(project.id, os.path.basename(dataset_path))

for bboxes in bboxes_json:
    image_path = os.path.join(dataset_path, (bboxes + ".jpg"))
    image_info = api.image.upload_path(dataset.id, bboxes + ".jpg", image_path)

    xmin = bboxes_json[bboxes]["bbox"][0]["xmin"]
    ymin = bboxes_json[bboxes]["bbox"][0]["ymin"]
    xmax = bboxes_json[bboxes]["bbox"][0]["xmax"]
    ymax = bboxes_json[bboxes]["bbox"][0]["ymax"]

    bbox = sly.Rectangle(top=ymin, left=xmin, bottom=ymax, right=xmax)
    class_name = bboxes_json[bboxes]["bbox"][0]["label"]
    obj_class = meta.get_obj_class(class_name)
    if obj_class is None:
        obj_class = sly.ObjClass(class_name, sly.Rectangle)
        meta = meta.add_obj_class(obj_class)
        api.project.update_meta(project.id, meta)
    label = sly.Label(bbox, obj_class)

    ann = sly.Annotation(img_size=[image_info.height, image_info.width], labels=[label])
    api.annotation.upload_ann(image_info.id, ann)
    print(f"uploaded bbox to image(id:{image_info.id})")

    mask_path = (
        os.path.join(os.path.dirname(dataset_path), "masks", (bboxes + ".jpg")),
    )
    try:
        load_image_labels(mask_path)
    except Exception as e:
        print(e)
        break
    print(f"uploaded mask to image(id:{image_info.id})")
print(f"Dataset {dataset.id} has been successfully created.")
