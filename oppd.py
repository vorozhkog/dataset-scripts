import os
import json
from dotenv import load_dotenv
from collections import defaultdict
import supervisely as sly

# https://vision.eng.au.dk/open-plant-phenotyping-database/

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

datasets = [
    "/Users/germanvorozko/Downloads/OPPD-master-DATA-images_full-ALOMY/DATA/images_full/ALOMY",
    "/Users/germanvorozko/Downloads/OPPD-master-DATA-images_full-APESV/DATA/images_full/APESV",
]
project_name = os.path.basename(os.path.dirname(datasets[0]))
project = api.project.get_info_by_name(workspace.id, project_name)
if project is not None:
    api.project.remove(project.id)
project = api.project.create(workspace.id, project_name)
meta = sly.ProjectMeta()

for dataset_dir in datasets:
    dataset_name = os.path.basename(dataset_dir)
    dataset = api.dataset.get_or_create(
        project.id, dataset_name
    )  # api.dataset.create(project.id, dataset_name)

    ann_paths = sly.fs.list_files(dataset_dir, valid_extensions=[".json"])
    for ann_path in ann_paths:
        with open(ann_path) as json_file:
            ann_json = json.load(json_file)

        # upload image
        img_name = ann_json["filename"]
        image_path = os.path.join(dataset_dir, img_name)
        image_info = api.image.upload_path(dataset.id, img_name, image_path)
        (print(f"image(id:{image_info.id}) is uploaded to dataset (id:{dataset.id})."))

        # check tagmeta in project and init if needed
        tag_name = "Growth condition"
        tag_value_type = sly.TagValueType.ANY_STRING
        tag_meta = meta.get_tag_meta(tag_name)
        if tag_meta is None:
            tag_meta = sly.TagMeta(tag_name, tag_value_type)
            meta = meta.add_tag_meta(tag_meta)
            api.project.update_meta(dataset.project_id, meta)

        # add tag to image
        tag = sly.Tag(tag_meta, ann_json["growth_condition"])

        labels = []
        for plant in ann_json["plants"]:
            xmin = plant["bndbox"]["xmin"]
            ymin = plant["bndbox"]["ymin"]
            xmax = plant["bndbox"]["xmax"]
            ymax = plant["bndbox"]["ymax"]
            # xmin, ymin, xmax, ymax = list(obj["bndbox"].values())
            bbox = sly.Rectangle(top=ymin, left=xmin, bottom=ymax, right=xmax)
            class_name = plant["eppo"]
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
