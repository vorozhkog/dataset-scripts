import os
from dotenv import load_dotenv
from collections import defaultdict
import supervisely as sly
import cv2

# https://ieee-dataport.org/competitions/endoscopy-disease-detection-and-segmentation-edd2020#files

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
    r"C:\Users\German\Documents\EndoCV2020-Endoscopy-Disease-Detection-Segmentation-subChallenge_data\masks",
    r"C:\Users\German\Documents\EndoCV2020-Endoscopy-Disease-Detection-Segmentation-subChallenge_data\bbox",
]

# read class names
directory = os.path.dirname(datasets[0])
with open(directory + "\\class_list.txt") as class_file:
    class_names = class_file.read().split("\n")

# create project and initialize meta
project_name = os.path.basename(os.path.dirname(datasets[0]))
project = api.project.get_info_by_name(workspace.id, project_name)
if project is not None:
    api.project.remove(project.id)
project = api.project.create(workspace.id, project_name)
meta = sly.ProjectMeta()

# create object classes
for name in class_names:
    if name == "":
        break
    obj_class = sly.ObjClass(name + "_bbox", sly.Rectangle)
    meta = meta.add_obj_class(obj_class)
    obj_class_mask = sly.ObjClass(name, sly.Bitmap)
    meta = meta.add_obj_class(obj_class_mask)

api.project.update_meta(project.id, meta)


def load_image_labels_mask(image_path, labels_path, class_name):
    global meta
    image_info = api.image.get_info_by_name(
        dataset_mask.id, os.path.basename(image_path)
    )
    if image_info is None:
        image_info = api.image.upload_path(
            dataset_mask.id, os.path.basename(image_path), image_path
        )
    mask = cv2.imread(labels_path, cv2.IMREAD_GRAYSCALE)
    labels = []
    height = image_info.height
    width = image_info.width
    bitmap_annotation = sly.Bitmap(mask)
    obj_class = meta.get_obj_class(class_name)
    label = sly.Label(bitmap_annotation, obj_class)
    labels.append(label)

    ann: sly.Annotation = sly.Annotation.from_json(
        api.annotation.download_json(image_info.id), meta
    )
    ann = ann.add_labels(labels)
    # ann = sly.Annotation(img_size=[height, width], labels=labels)
    api.annotation.upload_ann(image_info.id, ann)
    print(f"Image (id:{image_info.id}) has been successfully processed and uploaded.")


# def load_image_labels_bbox(image_path, labels_path):
#     image_info = api.image.upload_path(
#         dataset.id, os.path.basename(image_path), image_path
#     )
#     output = []
#     with open(labels_path) as file:
#         file_split = file.read().rstrip().split("\n")
#     for row in file_split:
#         if row == "":
#             continue
#         output.append(row.split())
#     labels = []
#     height = image_info.height
#     width = image_info.width
#     for bbox in output:
#         xmin, ymin, xmax, ymax, c = bbox
#         obj_class_name = c + "_bbox"

#         bbox_annotation = sly.Rectangle(
#             top=int(ymin), left=int(xmin), bottom=int(ymax), right=int(xmax)
#         )
#         obj_class = meta.get_obj_class(obj_class_name)
#         label = sly.Label(bbox_annotation, obj_class)
#         labels.append(label)

#         ann = sly.Annotation(img_size=[height, width], labels=labels)
#         api.annotation.upload_ann(image_info.id, ann)
#     print(f"Image (id:{image_info.id} has been successfully processed and uploaded.)")


# # FOR BBOXES DATASET
# ann_path = sly.fs.list_files(datasets[1], valid_extensions=[".txt"])
# dataset = api.dataset.create(project.id, os.path.basename(datasets[1]))
# # upload bboxes to images
# for path in ann_path:
#     image_path = os.path.join(
#         os.path.dirname(datasets[1]),
#         "originalImages",
#         (os.path.basename(path)[:-4] + ".jpg"),
#     )
#     try:
#         load_image_labels_bbox(image_path, path)
#     except Exception as e:
#         print(e)
#         continue
# print(f"Dataset {dataset.id} with bboxes has been successfully created.")

mask_paths = sly.fs.list_files(datasets[0], valid_extensions=[".tif"])
dataset_mask = api.dataset.create(project.id, os.path.basename(datasets[0]))
image_paths = sly.fs.list_files(
    os.path.join(os.path.dirname(datasets[0]), "originalImages"),
    valid_extensions=[".jpg"],
)
for path in image_paths:
    image_filename = sly.fs.get_file_name(path)
    for mask in mask_paths:
        mask_filename = sly.fs.get_file_name(mask)
        mask_filename_parts = mask_filename.rstrip().split("_")
        single_mask_name = mask_filename_parts[0] + "_" + mask_filename_parts[1]
        class_name = mask_filename_parts[2]
        if single_mask_name == image_filename:
            try:
                load_image_labels_mask(path, mask, class_name)
            except Exception as e:
                print(e)
                continue
print(f"Dataset (id:{dataset_mask.id}) with masks has been successfully created.")

exit(0)

# FOR MASKS DATASET
mask_paths = sly.fs.list_files(datasets[0])
dataset_mask = api.dataset.create(project.id, os.path.basename(datasets[0]))
for path in mask_paths:
    mask_image_path = os.path.join(
        datasets[1], "originalImages", (os.path.basename(path)[:-4] + ".jpg")
    )
    image_path_dir = os.path.join(os.path.dirname(datasets[0]), "originalImages")
    mask_paths = sly.fs.list_files(datasets[0])
    for mask in mask_paths:
        mask_filename = sly.fs.get_file_name(mask)
        mask_filename_parts = mask_filename.rstrip().split("_")
        single_mask_name = mask_filename_parts[0] + "_" + mask_filename_parts[1]
        image_path = os.path.join(image_path_dir, (single_mask_name + ".jpg"))
        class_name = mask_filename_parts[2]
        try:
            load_image_labels_mask(image_path, mask, class_name)
        except Exception as e:
            print(e)
            continue
print(f"Dataset {dataset_mask.id} with masks has been successfully created.")
