import os
from dotenv import load_dotenv
from collections import defaultdict
import supervisely as sly
import cv2

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

dataset = r"C:\Users\German\Documents\hyper-kvasir-segmented-images\images"


def load_image_labels(image_path, labels_path):
    image_info = api.image.upload_path(
        dataset.id, os.path.basename(image_path), image_path
    )
    mask = cv2.imread(labels_path, cv2.IMREAD_GRAYSCALE)
    labels = []
    height = image_info.height
    width = image_info.width
    bitmap_annotation = sly.Bitmap(
        mask,
    )
    obj_class = meta.get_obj_class("Arabidopsis Thaliana")
    label = sly.Label(bitmap_annotation, obj_class)
    labels.append(label)

    ann = sly.Annotation(img_size=[height, width], labels=labels)
    api.annotation.upload_ann(image_info.id, ann)
    print(f"Image (id:{image_info.id} has been successfully processed and uploaded.)")


# create project and initialize meta
project_name = os.path.basename(os.path.dirname(os.path.dirname(dataset)))
project = api.project.get_info_by_name(workspace.id, project_name)
if project is not None:
    api.project.remove(project.id)
project = api.project.create(workspace.id, project_name)
meta = sly.ProjectMeta()

# create object class
obj_class = sly.ObjClass(xxx, sly.Bitmap)
meta = meta.add_obj_class(obj_class)
api.project.update_meta(project.id, meta)

for single_dataset in datasets:
    mask_path = sly.fs.list_files(
        os.path.join(single_dataset + "\\gt"), valid_extensions=[".png"]
    )
    dataset = api.dataset.create(
        project.id, os.path.basename(os.path.dirname(single_dataset))
    )
    # upload bboxes to images
    for path in mask_path:
        image_path = os.path.join(
            single_dataset, (os.path.basename(path)[:-7] + ".png")
        )
        try:
            load_image_labels(image_path, path)
        except Exception as e:
            print(e)
            continue
    print(f"Dataset {dataset.id} has been successfully created.")
