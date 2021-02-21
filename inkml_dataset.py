import inkml
from inkml_dataset_utlis import ChartCategory
from inkml_dataset_utlis import get_categroy_id_from_label
import coco_dataset
import os
import numpy as np
import cv2
import math
import json

debugging_mode = False
def set_debugging_mode(enable):
    debugging_mode = enable

def is_debugging_mode():
    return False

N = inkml.N
def get_arrow_segmatation(id, ls):
    background_color = (0, 0, 0)
    img = np.zeros((inkml.N, inkml.N, 3), dtype=np.uint8)
    for subls in ls:
        data = np.array(subls)
        img = cv2.polylines(img, [convertDataToPloyPts(data)], False, (255, 255, 255), 6)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hiberachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return segemation_from_contours(contours)

def convertDataToPloyPts(data):
    pts = np.array(data, np.int32)
    pts = pts.reshape((-1,1,2))
    return pts

def get_distance_to_bbox(x, y, x1, y1, x2, y2):
    x_x1 = abs(x - x1)
    x_x2 = abs(x - x2)
    y_y1 = abs(y - y1)
    y_y2 = abs(y - y2)
    if x >= x1 and x <= x2:
        if y <= y1:
            return y_y1
        elif y >= y2:
            return y_y2
        else:
            return min(y_y1, y_y2, x_x1, x_x2)

    if y >= y1 and y <= y2:
        if x <= x1:
            return x_x1
        elif x >= x2:
            return x_x2
        else:
            return min(y_y1, y_y2, x_x1, x_x2)

    if x < x1 and y < y1:
        return math.sqrt(x_x1**2 + y_y1**2)
    elif x > x2 and y < y1:
        return math.sqrt(x_x2**2 + y_y1**2)
    elif x < x1 and y > y2:
        return math.sqrt(x_x1**2 + y_y2**2)
    elif x > x2 and y > y2:
        return math.sqrt(x_x2**2 + y_y2**2)
    else:
        print('Unhandled: ', x, y, [x1, y1, x2, y2])
        exit()

def get_min_distance_pt_to_bbox(traces, bbox, logging = False):
    min_distance = 10000000
    x1, y1, x2, y2 = bbox
    ret = None
    for trace in traces:
        for pt in trace:
            dist = get_distance_to_bbox(pt[0], pt[1], x1, y1, x2, y2)
            if dist < min_distance:
                ret = pt
                min_distance = dist

    return ret, min_distance


def get_keypoints(trace_group, debugging = False):
    if trace_group.label != "arrow":
        return None, None

    if trace_group.source == None or trace_group.target == None:
        return None, None

    if debugging:
        print("#source: ", trace_group.source)
        print("#target: ", trace_group.target)

    start_pt, dist1 =   get_min_distance_pt_to_bbox(trace_group.traces, trace_group.source.bbox, debugging)
    end_pt, dist2 =     get_min_distance_pt_to_bbox(trace_group.traces, trace_group.target.bbox, debugging)

    if debugging:
        pass
    return ChartCategory.LineArrow1, [round(start_pt[0]), round(start_pt[1]), 2, round(end_pt[0]), round(end_pt[1]), 2]


def segemation_from_contours(contours):
    if len(contours) == 0 or len(contours[0]) == 0:
        return [], []

    minX = 1000000
    minY = 1000000
    maxX = -1000000
    maxY = -1000000

    area = 0

    segs = []
    for contour in contours:
        seg = []
        area += cv2.contourArea(contour)
        for pts in contour:
            for pt in pts:
                x = int(pt[0])
                y = int(pt[1])
                seg.append(x)
                seg.append(y)

                minX = min(x, minX)
                maxX = max(x, maxX)
                minY = min(y, minY)
                maxY = max(y, maxY)

        segs.append(seg)
    return area, [int(minX), int(minY), int(maxX - minX), int(maxY - minY)], segs

class InkMLFile(coco_dataset.CocoItem):
    def __init__(self):
        super().__init__()

    def load(self, file_path):
        print("load : ", file_path)
        self.file_path = file_path
        self.basename = os.path.basename(file_path)
        title, ext = os.path.splitext(self.basename)
        self.image_filename =  "ink_" + title + ".png"
        self.traces_data = inkml.get_traces_data(file_path)
        self.__parse()
        if is_debugging_mode():
            cv2.imshow("debugging_mode", self.annotation_imge())
            cv2.waitKey()

    def __parse(self):
        # frist, parse
        self.__parse_no_arrow()
        self.__parse_arrow()

    def __parse_arrow(self):
        traces = self.traces_data
        annotations = []
        for elem in traces:
            ls = elem.traces
            minX, minY, maxX, maxY = elem.bbox
            bbox_x = round(minX)
            bbox_y = round(minY)
            bbox_w = round((maxX - minX))
            bbox_h = round((maxY - minY))

            label = elem.label
            if label != "arrow":
                continue

            bbox = [bbox_x, bbox_y, bbox_w, bbox_h]
            area = bbox_w * bbox_h
            segmentation = [[maxX, minY, maxX, maxY, minX, maxY, minX, minY]]


            categroy_id = None
            area, bbox, segmentation = get_arrow_segmatation(elem.id, ls)
            categroy_id, keypoints = get_keypoints(elem)

            if categroy_id == None:
                categroy_id = get_categroy_id_from_label(label)


            elem.annotation = {
                "category_id": categroy_id,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "segmentation": segmentation
            }

            if keypoints != None and len(keypoints) > 0 and len(keypoints) % 3 == 0:
                elem.annotation["keypoints"] = keypoints
                elem.annotation["num_keypoints"] = len(keypoints) // 3
            else:
                elem.annotation["keypoints"] = [0,0,0, 0,0,0]
                elem.annotation["num_keypoints"] = 0

    def __parse_no_arrow(self):
        traces = self.traces_data
        annotations = []
        for elem in traces:
            ls = elem.traces
            minX, minY, maxX, maxY = elem.bbox
            bbox_x = round(minX)
            bbox_y = round(minY)
            bbox_w = round((maxX - minX))
            bbox_h = round((maxY - minY))

            label = elem.label
            if label == None:
                continue

            if label == "arrow":
                continue

            bbox = [bbox_x, bbox_y, bbox_w, bbox_h]
            area = bbox_w * bbox_h
            segmentation = [[maxX, minY, maxX, maxY, minX, maxY, minX, minY]]
            categroy_id = get_categroy_id_from_label(label)

            elem.annotation = {
            "category_id": categroy_id,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0,
            "segmentation": segmentation,
            "keypoints": [],
            "num_keypoints": 0
            }

    def get_annotations(self, image_id, id_prefix):
        traces = self.traces_data
        id = 0
        annotations = []
        for elem in traces:
            id += 1
            annotation = elem.annotation
            if annotation == None:
                continue

            annotation["id"] = id_prefix + id
            annotation["image_id"] = image_id
            annotations.append(annotation)
        return annotations

    def pure_image(self):
        traces = self.traces_data

        background_color = (255, 255, 255)
        img = np.zeros((N, N, 3), dtype=np.uint8)
        img = cv2.rectangle(img, (0, 0), (N, N), background_color, thickness=-1)

        for elem in traces:
            ls = elem.traces
            thickness = 1

            # draw traces
            for subls in ls:
                data = np.array(subls)
                pts = np.array(data, np.int32)
                pts = pts.reshape((-1,1,2))
                img = cv2.polylines(img, [pts], False, (0, 0, 0), thickness=2)
                #img = cv2.rectangle(img, (round(minX), round(minY)), (round(maxX), round(maxY)), color, thickness)
        return img

    def keypoints_image(self, img, annotation):
        keypoints = annotation.get("keypoints")
        if keypoints  == None or len(keypoints) == 0:
            return img

        start_pt = (keypoints[0], keypoints[1])
        end_pt = (keypoints[3], keypoints[4])

        img = cv2.circle(img, start_pt, 10, (255, 0, 0), -1)
        img = cv2.circle(img, end_pt, 10, (0, 255, 0), -1)

        return img

    def annotation_imge(self):
        img = self.pure_image()
        traces = self.traces_data
        for elem in traces:
            annotation = elem.annotation
            x, y, w, h = annotation["bbox"]
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (127, 127, 0), thickness=1)
            img = self.keypoints_image(img, annotation)

            # annotaion text
            fontScale = 0.7
            # Line thickness of 2 px
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX

            img = cv2.putText(img,  elem.id, (x, y), font,
                fontScale, (127, 127, 0), thickness, cv2.LINE_AA)
            #cv2.polylines(img, annotation["segmentation"], False, (127, 1, 0), thickness=1)

        return img

    def save_image(self, output_path):
        img = self.pure_image()
        cv2.imwrite(output_path, img)

    def image_width(self):
        return inkml.N

    def image_height(self):
        return inkml.N

    def image_file_name(self):
        return self.image_filename

class InkMLDataSet(coco_dataset.CocoDataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.file_list = "listInkML.txt"

    def load(self):
        super().load()
        with open(self.root_dir + self.file_list) as file:
            for line in file:
                file_path = self.root_dir + line.strip()
                inkml_file = InkMLFile()
                inkml_file.load(file_path)
                self.add_item(inkml_file)


if __name__ == "__main__":
    pass