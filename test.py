from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_dict, load_coco_json
from detectron2.structures import BoxMode
import json
import random

print("XXXXXXXXXXXXXX")
if __name__ == "__main__":
    register_coco_instances("flowchart_dataset_train", {}, "./inkml_train.json", "./images")
    register_coco_instances("flowchart_dataset_val", {}, "./inkml_val2.json", "./images")

    with open("inkml_val2.json") as fp:
        data = json.load(fp)
        annotations = data["annotations"]
        for a in annotations:
            category_id = a["category_id"]
            if category_id == None:
                print(a)

    '''
    dataset_dicts = DatasetCatalog.get("flowchart_dataset_val")
    flowchart_metadata = MetadataCatalog.get("flowchart_dataset_val")
    for d in random.sample(dataset_dicts, 3):
        print(d)
    '''




