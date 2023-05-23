import os
from dotenv import load_dotenv
from collections import defaultdict
import supervisely as sly
import json
import xml.etree.ElementTree as ET

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

dataset_voc_path = (
    r"C:\Users\German\Documents\perrenial plants\voc_annotations\voc_annotations"
)
# datasets_json_path = [
#     r"C:\Users\German\Documents\perrenial plants\coco_annotations\coco_annotations\json_annotation_test.json",
#     r"C:\Users\German\Documents\perrenial plants\coco_annotations\coco_annotations\json_annotation_train.json",
#     r"C:\Users\German\Documents\perrenial plants\coco_annotations\coco_annotations\json_annotation_val.json",
# ]


def etree_to_dict(t):
    if type(t) is ET.ElementTree:
        return etree_to_dict(t.getroot())
    return {**t.attrib, "text": t.text, **{e.tag: etree_to_dict(e) for e in t}}


# create project and initialize meta
project_name = os.path.basename(os.path.dirname(os.path.dirname(dataset_voc_path)))
project = api.project.get_info_by_name(workspace.id, project_name)
if project is not None:
    api.project.remove(project.id)
project = api.project.create(workspace.id, project_name)
meta = sly.ProjectMeta()


def nested_dict_pairs_iterator(dict_obj):
    """This function accepts a nested dictionary as argument
    and iterate over all values of nested dictionaries
    """
    # Iterate over all key-value pairs of dict argument
    for key, value in dict_obj.items():
        # Check if value is of dict type
        if isinstance(value, dict):
            # If value is dict then iterate over all its values
            for pair in nested_dict_pairs_iterator(value):
                yield (key, *pair)
        else:
            # If value is not dict type then yield the value
            yield (key, value)


# create dataset and iterate over annotation paths
dataset_voc = api.dataset.create(project.id, os.path.basename(dataset_voc_path))
voc_ann_paths = sly.fs.list_files(dataset_voc_path)
for path in voc_ann_paths:
    tree = ET.parse(path)
    dict_xml = etree_to_dict(tree.getroot())
    for pair in nested_dict_pairs_iterator(dict_xml):
        print(pair)
    # get image path and upload
    image_path = os.path.join(
        r"C:\Users\German\Documents\perrenial plants\raw_images\raw_images",
        dict_xml["filename"]["text"],
    )
    image_info = api.image.upload_path(
        dataset_voc.id, dict_xml["filename"]["text"], image_path
    )

    labels = []
    for k, v in dict_xml.items():
        if k != "object":
            continue
        # get bbox cords
        xmin = int(dict_xml["object"]["bndbox"]["xmin"]["text"])
        ymin = int(dict_xml["object"]["bndbox"]["ymin"]["text"])
        xmax = int(dict_xml["object"]["bndbox"]["xmax"]["text"])
        ymax = int(dict_xml["object"]["bndbox"]["ymax"]["text"])

        # get class name, create labels
        bbox = sly.Rectangle(top=ymin, left=xmin, bottom=ymax, right=xmax)
        class_name = dict_xml["object"]["name"]["text"]
        obj_class = meta.get_obj_class(class_name)
        if obj_class is None:
            obj_class = sly.ObjClass(class_name, sly.Rectangle)
            meta = meta.add_obj_class(obj_class)
            api.project.update_meta(project.id, meta)
        label = sly.Label(bbox, obj_class)
        labels.append(label)

    # upload annotation
    ann = sly.Annotation(
        img_size=[
            int(dict_xml["size"]["height"]["text"]),
            int(dict_xml["size"]["width"]["text"]),
        ],
        labels=labels,
    )
    api.annotation.upload_ann(image_info.id, ann)
    print(f"uploaded bbox to image(id:{image_info.id})")
print(f"Dataset {dataset_voc.id} has been successfully created.")
