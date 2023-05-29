import os
from dotenv import load_dotenv
import supervisely as sly
import xml.etree.ElementTree as ET
import xmltodict

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

dataset_voc_path = r"C:\Users\German\Documents\Maize_Tassels_Recognition"

# datasets_json_path = [
#     r"C:\Users\German\Documents\perrenial plants\coco_annotations\coco_annotations\json_annotation_test.json",
#     r"C:\Users\German\Documents\perrenial plants\coco_annotations\coco_annotations\json_annotation_train.json",
#     r"C:\Users\German\Documents\perrenial plants\coco_annotations\coco_annotations\json_annotation_val.json",
# ]

# create project and initialize meta
project_name = os.path.basename((dataset_voc_path))
project = api.project.get_info_by_name(workspace.id, project_name)
if project is not None:
    api.project.remove(project.id)
project = api.project.create(workspace.id, project_name)
meta = sly.ProjectMeta()


# create dataset and iterate over annotation paths
dataset_voc = api.dataset.create(project.id, os.path.basename(dataset_voc_path))
img_paths = sly.fs.list_files(
    os.path.join(dataset_voc_path, "2_Raw_RGB_Images_Collected_by_UAV")
)
for path in img_paths:
    # get image path and upload
    ann_path = os.path.join(
        dataset_voc_path,
        "3_Labels_for_CNN",
        "Labels",
        (sly.fs.get_file_name(path) + ".xml"),
    )
    try:
        tree = ET.parse(ann_path)
    except Exception as e:
        print(f"EXCEPTION: {e}")
        continue
    xml_data = tree.getroot()
    xmlstr = ET.tostring(xml_data, encoding="utf-8", method="xml")

    data_dict = dict(xmltodict.parse(xmlstr))

    image_info = api.image.upload_path(
        dataset_voc.id, data_dict["annotation"]["filename"], path
    )
    labels = []
    if "object" not in data_dict["annotation"]:
        continue
    if type(data_dict["annotation"]["object"]) == list:
        for obj in data_dict["annotation"]["object"]:
            # get bbox cords
            xmin = int(obj["bndbox"]["xmin"])
            ymin = int(obj["bndbox"]["ymin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymax = int(obj["bndbox"]["ymax"])

            # get class name, create labels
            bbox = sly.Rectangle(top=ymin, left=xmin, bottom=ymax, right=xmax)
            class_name = obj["name"]
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
                int(data_dict["annotation"]["size"]["height"]),
                int(data_dict["annotation"]["size"]["width"]),
            ],
            labels=labels,
        )
        api.annotation.upload_ann(image_info.id, ann)
        print(f"uploaded bbox to image(id:{image_info.id})")
    else:
        xmin = int(data_dict["annotation"]["object"]["bndbox"]["xmin"])
        ymin = int(data_dict["annotation"]["object"]["bndbox"]["ymin"])
        xmax = int(data_dict["annotation"]["object"]["bndbox"]["xmax"])
        ymax = int(data_dict["annotation"]["object"]["bndbox"]["ymax"])
        bbox = sly.Rectangle(top=ymin, left=xmin, bottom=ymax, right=xmax)
        class_name = data_dict["annotation"]["object"]["name"]
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
                int(data_dict["annotation"]["size"]["height"]),
                int(data_dict["annotation"]["size"]["width"]),
            ],
            labels=labels,
        )
        api.annotation.upload_ann(image_info.id, ann)
        print(f"uploaded bbox to image(id:{image_info.id})")
print(f"Dataset {dataset_voc.id} has been successfully created.")
