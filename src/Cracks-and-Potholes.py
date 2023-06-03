import os
from dotenv import load_dotenv
import supervisely as sly
import cv2

# https://biankatpas.github.io/Cracks-and-Potholes-in-Road-Images-Dataset/

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

workspace_id = int(os.environ["context.workspaceId"])
workspace = api.workspace.get_info_by_id(workspace_id)
if workspace is None:
    raise KeyError(f"Workspace with ID {workspace_id} not found in your account")
else:
    print(f"Workspace name = {workspace.name}, id = {workspace.id}")

dataset_path = (
    r"C:\Users\German\Documents\Cracks-and-Potholes-in-Road-Images-Dataset\Dataset"
)


def load_image_labels(labels_path):
    mask = cv2.imread(labels_path, cv2.IMREAD_GRAYSCALE)
    im_bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    bitmap_annotation = sly.Bitmap(
        data=im_bw,
    )
    obj_class = meta.get_obj_class(class_name)
    label = sly.Label(bitmap_annotation, obj_class)
    labels.append(label)


# create project and initialize meta
project_name = os.path.basename(os.path.dirname(dataset_path))
project = api.project.get_info_by_name(workspace.id, project_name)
if project is not None:
    api.project.remove(project.id)
project = api.project.create(workspace.id, project_name)
meta = sly.ProjectMeta()

dataset = api.dataset.create(project.id, os.path.basename(dataset_path))

dataset_folder = [f.path for f in os.scandir(dataset_path) if f.is_dir()]

for path in dataset_folder:
    img_path = sly.fs.list_files(path, valid_extensions=[".jpg"])
    masks_path = sly.fs.list_files(path, valid_extensions=[".png"])
    image_info = api.image.upload_path(
        dataset.id, sly.fs.get_file_name(img_path[0]), img_path[0]
    )
    labels = []
    for mask in masks_path:
        filename = sly.fs.get_file_name(mask)
        class_name = filename.replace((os.path.basename(path) + "_"), "")
        obj_class = meta.get_obj_class(class_name)
        if obj_class is None:
            obj_class = sly.ObjClass(class_name, sly.Bitmap)
            meta = meta.add_obj_class(obj_class)
            api.project.update_meta(project.id, meta)
        try:
            load_image_labels(mask)
            print(f"uploaded mask to image(id:{image_info.id})")
        except Exception as e:
            print(f"{class_name} mask is missing. Skipping...")
            continue
    height = image_info.height
    width = image_info.width

    ann = sly.Annotation(img_size=[height, width], labels=labels)
    api.annotation.upload_ann(image_info.id, ann)
    print(f"uploaded annotations to image(id:{image_info.id})")
print(f"Dataset {dataset.id} has been successfully created.")
