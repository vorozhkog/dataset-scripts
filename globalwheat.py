import os
import csv
from dotenv import load_dotenv
from collections import defaultdict
import supervisely as sly

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

# test value from local.env
workspace_id = int(os.environ["context.workspaceId"])
workspace = api.workspace.get_info_by_id(workspace_id)
if workspace is None:
    raise KeyError(f"Workspace with ID {workspace_id} not found in your account")
else:
    print(f"Workspace name = {workspace.name}, id = {workspace.id}")

datasets = r"C:\Users\German\Documents\gwhd_2021\images"
# create project and initialize meta
project_name = os.path.basename(os.path.dirname(datasets))
project = api.project.get_info_by_name(workspace.id, project_name)
if project is not None:
    api.project.remove(project.id)
project = api.project.create(workspace.id, project_name)
meta = sly.ProjectMeta()

# read dataset meta
ann_paths = sly.fs.list_files(os.path.dirname(datasets), valid_extensions=[".csv"])
for ann_path in ann_paths:
    dataset_name = sly.fs.get_file_name(ann_path)
    dataset = api.dataset.get_or_create(project.id, dataset_name)
    with open(ann_path, "r") as file:
        meta_file = csv.reader(file)
        next(meta_file)
        for row in meta_file:
            image_name = row[0]
            image_path = os.path.join(datasets, image_name)
            image_info = api.image.upload_path(dataset.id, image_name, image_path)
            (
                print(
                    f"image(id:{image_info.id}) is uploaded to dataset (id:{dataset.id})."
                )
            )

            bboxes_list = row[1].split(";")

            # check tagmeta in project and init if needed
            tag_value_type = sly.TagValueType.ANY_STRING
            tag_name = "domain"
            tag_meta = meta.get_tag_meta(tag_name)
            if tag_meta is None:
                tag_meta = sly.TagMeta(tag_name, tag_value_type)
                meta = meta.add_tag_meta(tag_meta)
                api.project.update_meta(dataset.project_id, meta)

            # add tag to image
            tag = sly.Tag(tag_meta, row[2])
            labels = []
            for bbox in bboxes_list:
                if bbox == "no_box":
                    pass
                x_min, y_min, x_max, y_max = bbox.split()
                bbox = sly.Rectangle(
                    top=int(y_min), left=int(x_min), bottom=int(y_max), right=int(x_max)
                )
                class_name = "wheat head"
                obj_class = meta.get_obj_class(class_name)
                if obj_class is None:
                    obj_class = sly.ObjClass(class_name, sly.Rectangle)
                    meta = meta.add_obj_class(obj_class)
                    api.project.update_meta(project.id, meta)
                label = sly.Label(bbox, obj_class)
                labels.append(label)
            ann = sly.Annotation(
                img_size=[image_info.height, image_info.width],
                labels=labels,
                img_tags=[tag],
            )
            api.annotation.upload_ann(image_info.id, ann)
            print(f"uploaded annotation to image(id:{image_info.id})")

    print(f"Dataset {dataset.name} has been successfully processed")
