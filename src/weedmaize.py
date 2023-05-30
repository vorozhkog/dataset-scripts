import os
from dotenv import load_dotenv
import supervisely as sly
import xml.etree.ElementTree as ET
import xmltodict

# https://www.research-collection.ethz.ch/handle/20.500.11850/512332

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
    r"C:\Users\German\Documents\test_set",
    r"C:\Users\German\Documents\train_set",
]


# def load_image_labels(image_path, labels_path):
#     image_info = api.image.upload_path(
#         dataset.id, os.path.basename(image_path), image_path
#     )
#     mask = cv2.imread(labels_path, cv2.IMREAD_GRAYSCALE)
#     # cv2.imwrite(image_info.name, mask)
#     thresh = 127
#     im_bw = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)[1]
#     # mask = cv2.bitwise_not(im_bw)
#     labels = []
#     height = image_info.height
#     width = image_info.width
#     bitmap_annotation = sly.Bitmap(
#         im_bw,
#     )
#     obj_class = meta.get_obj_class(os.path.basename(project_path))
#     if obj_class is None:
#         obj_class = sly.ObjClass(os.path.basename(project_path), sly.Bitmap)
#         meta = meta.add_obj_class(obj_class)
#         api.project.update_meta(project.id, meta)
#     label = sly.Label(bitmap_annotation, obj_class)
#     labels.append(label)

#     ann = sly.Annotation(img_size=[height, width], labels=labels)
#     api.annotation.upload_ann(image_info.id, ann)
#     print(f"Image (id:{image_info.id}) has been successfully processed and uploaded.")


# create project and initialize meta
project_name = "WeedMaize"
project = api.project.get_info_by_name(workspace.id, project_name)
if project is not None:
    api.project.remove(project.id)
project = api.project.create(workspace.id, project_name)
meta = sly.ProjectMeta()

# create object class
# obj_class = sly.ObjClass(os.path.basename(project_path), sly.Bitmap)
# meta = meta.add_obj_class(obj_class)
# api.project.update_meta(project.id, meta)

for single_dataset in datasets:
    mask_path = sly.fs.list_files(
        single_dataset,
        valid_extensions=[
            ".xml",
        ],
    )
    dataset = api.dataset.create(project.id, os.path.basename(single_dataset))
    for path in mask_path:
        tree = ET.parse(path)
        xml_data = tree.getroot()
        xmlstr = ET.tostring(xml_data, encoding="utf-8", method="xml")

        data_dict = dict(xmltodict.parse(xmlstr))
        # get image path and upload
        image_path = os.path.join(
            single_dataset,
            data_dict["annotation"]["filename"],
        )
        image_info = api.image.upload_path(
            dataset.id, data_dict["annotation"]["filename"], image_path
        )

        labels = []

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
    print(f"Dataset {dataset.id} has been successfully created.")

exit(0)

for single_dataset in datasets:
    mask_path = sly.fs.list_files(
        single_dataset,
        valid_extensions=[
            ".xml",
        ],
    )
    dataset = api.dataset.create(project.id, os.path.basename(single_dataset))
    # upload masks to images
    for path in mask_path:
        img_path = mask_path[:-4] + ".jpg"
        try:
            load_image_labels(img_path, path)
        except Exception as e:
            print(f"EXCEPTION: {e}")
            continue
    print(f"Dataset {dataset.id} has been successfully created.")
