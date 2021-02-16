import json
import os
import datetime
class CocoDataset:
    def __init__(self):
        self.items = []

    def load(self):
        # load the template
        current_dir = os.path.realpath(__file__)
        current_dir = os.path.dirname(current_dir)
        coco_template_file = os.path.join(current_dir, "coco_template.json")
        with open(coco_template_file) as fp:
            self.annotation = json.load(fp)
            self.annotation["info"]["date_created"] = str(datetime.date.today())
            print(self.annotation)

    def add_item(self, item):
        self.items.append(item)

    def get_annotation(self):
        image_id = 0xbc83000
        print("xxxx len", len(self.items))
        for item in self.items:
            image_id += 1
            image = {
                "license" : 1,
                "file_name": item.image_file_name(),
                "coco_url":"",
                "height": item.image_height(),
                "width":item.image_width(),
                "date_captured": "",
                "flickr_url":"",
                "id": image_id
            }

            annotations = item.get_annotations(image_id, image_id * 100)
            print(annotations)
            self.annotation["images"].append(image)
            self.annotation["annotations"] += annotations
        return self.annotation

    def save_images(self, root_dir):
        for item in self.items:
            file_path = os.path.join(root_dir, item.image_file_name())
            item.save_image(file_path)

class CocoItem:
    def __init__(self):
        pass

    def get_annotations(self, image_id, id_prefix):
        []

    def save_image(self, image_path):
        pass

    def image_width(self):
        return 0

    def image_height(self):
        return 0

    def image_file_name(self):
        return ""