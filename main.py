from inkml_dataset import InkMLDataSet
from inkml_dataset import InkMLFile
import inkml_dataset
import sys
import cv2
import json
def processOneFile():

    input_inkml = sys.argv[1]
    output_path = sys.argv[2]
    inkml_file = InkMLFile()
    inkml_file.load(input_inkml)

    cv2.imshow("xxxxx", inkml_file.annotation_imge())
    cv2.waitKey()


def processDataset(file_list, json_out_file, save_image = False):
    dataset = InkMLDataSet("FCinkML/")
    dataset.file_list = file_list
    dataset.load()

    annotation = dataset.get_annotation()
    annotation_str = json.dumps(annotation, indent=2)

    with open(json_out_file, "w") as coco_json_file:
        coco_json_file.write(annotation_str)

    if save_image:
        dataset.save_images("images")

def processAllFile():
    processDataset("listInkML_Train.txt", "inkml_train.json")
    processDataset("listInkML_Test.txt", "inkml_val.json")

if __name__ == "__main__":
    #processOneFile()
    inkml_dataset.set_debugging_mode(True)
    processAllFile()
    #processDataset("listInkML_Dev.txt", "inkml_dev.json")