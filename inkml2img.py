import numpy as np
import xml.etree.ElementTree as ET
from io import StringIO
import cv2
import coco_dataset
from enum import IntEnum
import os
import json
class ChartCategory(IntEnum):
    Rect = 1
    RounedRect = 2
    Ellipse = 3
    Cirle = 4
    Triangle = 5
    Diamond = 6
    Parallelogram = 7
    Ploygon4 = 8
    PloygonN = 9
    Star = 10
    Cylinder = 11
    Band = 12
    BlockArrow1 = 13
    BlockArrow2 = 14
    BlockArrow3 = 15
    LineArrow1 = 16
    LineArrow2 = 17
    LineArrow3 = 18
    Text = 19

def get_categroy_id_from_label(label):
    return int(get_categroy_id_from_label_enum(label))

def get_categroy_id_from_label_enum(label):
    if label == "process":
        return ChartCategory.Rect
    elif label == "text":
        return ChartCategory.Text
    elif label == "decision":
        return ChartCategory.Diamond
    elif label == "data":
        return ChartCategory.Parallelogram
    elif label == "arrow":
        return ChartCategory.LineArrow1
    elif label == "terminator":
        return ChartCategory.RounedRect
    elif label == "connection":
        return ChartCategory.Cirle
    else:
        raise "unknown-label: " + label
        print("", label)
        exit()
        return ChartCategory.Rect

def get_trace(traces_all, id):
    for trace in traces_all:
        if trace["id"] == id:
            return trace
    return None

N = 800
MARGIN = 10
K = (N - 2 * MARGIN) / 2000.0

xml = '{http://www.w3.org/XML/1998/namespace}'

def get_traces_data(inkml_file_abs_path, xmlns='{http://www.w3.org/2003/InkML}'):

        traces_data = []
        trace_groups = []

        tree = ET.parse(inkml_file_abs_path)
        root = tree.getroot()
        # doc_namespace = "{http://www.w3.org/2003/InkML}"
        doc_namespace = xmlns
        k = 1000
        'Stores traces_all with their corresponding id'
        traces_all = [{'id': trace_tag.get('id'),
                        'coords': [[(float(axis_coord) * K + MARGIN) \
                                        for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
                                    else [(float(axis_coord) * K + MARGIN) \
                                        for axis_coord in coord.split(' ')] \
                                for coord in (trace_tag.text).replace('\n', '').split(',')]} \
                                for trace_tag in root.findall(doc_namespace + 'trace')]

        trace_dict = dict()
        for trace in traces_all:
            trace_dict[trace["id"]] = trace
        'Sort traces_all list by id to make searching for references faster'
        traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))

        'Always 1st traceGroup is a redundant wrapper'
        traceGroupWrapper = root.find(doc_namespace + 'traceGroup')

        if traceGroupWrapper is not None:
            for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):

                label = traceGroup.find(doc_namespace + 'annotation').text
                traceGroupId = traceGroup.get(xml + 'id')
                'traces of the current traceGroup'
                traces_curr = []
                for traceView in traceGroup.findall(doc_namespace + 'traceView'):
                    'Id reference to specific trace tag corresponding to currently considered label'
                    traceDataRef = (traceView.get('traceDataRef'))

                    'Each trace is represented by a list of coordinates to connect'

                    single_trace = get_trace(traces_all, traceDataRef)['coords']
                    traces_curr.append(single_trace)

                traces_data.append({'id': traceGroupId,'label': label, 'trace_group': traces_curr})

        else:
            'Consider Validation data that has no labels'
            [traces_data.append({'trace_group': [trace['coords']]}) for trace in traces_all]

        return traces_data

def get_min_coords(trace_group):

        min_x_coords = []
        min_y_coords = []
        max_x_coords = []
        max_y_coords = []

        for trace in trace_group:

            x_coords = [coord[0] for coord in trace]
            y_coords = [coord[1] for coord in trace]

            min_x_coords.append(min(x_coords))
            min_y_coords.append(min(y_coords))
            max_x_coords.append(max(x_coords))
            max_y_coords.append(max(y_coords))

        return min(min_x_coords), min(min_y_coords), max(max_x_coords), max(max_y_coords)

def get_label_color(label):
    if label == "process":
        return (255, 0, 0)
    elif label == "text":
        return (127, 127, 127)
    elif label == "decision":
        return (0, 255, 0)
    elif label == "data":
        return (255, 255, 0)
    elif label == "arrow":
        return (255, 0, 0)
    elif label == "terminator":
        return (0, 0, 255)
    else:
        return (0, 0, 0)

def get_arrow_segmatation(id, ls):
    background_color = (0, 0, 0)
    img = np.zeros((N, N, 3), dtype=np.uint8)
    for subls in ls:
        data = np.array(subls)
        img = cv2.polylines(img, [convertDataToPloyPts(data)], False, (255, 255, 255), 6)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hiberachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img, contours, -1, (0, 0, 255), 1)


    return segemation_from_contours(contours)

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
                minX = min(y, minY)
                maxY = max(y, maxY)

        segs.append(seg)

    return area, [int(minX), int(minY), int(maxX - minX), int(maxY - minY)], segs

def convertDataToPloyPts(data):
    pts = np.array(data, np.int32)
    pts = pts.reshape((-1,1,2))
    return pts

ARROW_MARGIN = 5
def cv2inkml2img(input_path, output_path, color='black'):
    traces = get_traces_data(input_path)

    background_color = (255, 255, 255)
    img = np.zeros((N, N, 3), dtype=np.uint8)
    img = cv2.rectangle(img, (0, 0), (N, N), background_color, thickness=-1)

    for elem in traces:
        #print(elem)
        ls = elem['trace_group']

        minX, minY, maxX, maxY = get_min_coords(ls)
        bbox = (minX, minY, (maxX - minX), (maxY - minY))
        #print(minX, minY, maxX, maxY)

        label = elem.get("label")
        id = elem.get("id")

        if label == 'arrow':
            get_arrow_segmatation(id, ls)

        segs = []
        for subls in ls:
            data = np.array(subls)
            if label == 'arrow':
                seg_bottom = []
                seg_top = []
                for pt in data:
                    seg_bottom.append([pt[0] + ARROW_MARGIN, pt[1] + ARROW_MARGIN])
                    seg_top.append([pt[0] - ARROW_MARGIN, pt[1] - ARROW_MARGIN])
                seg = seg_bottom + seg_top[::-1]
                #img = cv2.polylines(img, [convertDataToPloyPts(seg)], True, (128, 0, 128), thickness=1)

                #img =cv2.putText(img, label + ": " + id, (round(minX), round(minY)), cv2.FONT_HERSHEY_SIMPLEX, 1, get_label_color(label))
                # get_arrow_segmatation(id, )
            thickness = 5 if label == 'arrow' else 2
            if label == 'arrow':
                img = cv2.polylines(img, [convertDataToPloyPts(data)], False, get_label_color(label), thickness)
            #img = cv2.rectangle(img, (round(minX), round(minY)), (round(maxX), round(maxY)), color, thickness)

    cv2.imwrite(output_path, img)
    cv2.imshow("xxxxx", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

class InkMLFile(coco_dataset.CocoItem):
    def __init__(self):
        super().__init__()

    def load(self, file_path):
        self.file_path = file_path
        self.basename = os.path.basename(file_path)
        title, ext = os.path.splitext(self.basename)
        self.image_filename =  "ink_" + title + ".png"
        self.traces_data = get_traces_data(file_path)
        self.__parse()

    def __parse(self):
        traces = self.traces_data
        annotations = []
        for elem in traces:
            ls = elem['trace_group']
            minX, minY, maxX, maxY = get_min_coords(ls)
            bbox_x = round(minX)
            bbox_y = round(minY)
            bbox_w = round((maxX - minX))
            bbox_h = round((maxY - minY))

            label = elem.get("label")
            if label == None:
                continue

            bbox = [bbox_x, bbox_y, bbox_w, bbox_h]
            area = bbox_w * bbox_h
            segmentation = [[maxX, minY, maxX, maxY, minX, maxY, minX, minY]]

            if label == 'arrow':
                area, bbox, segmentation = get_arrow_segmatation(elem['id'], ls)

            elem["__annotation__"] = {
            "category_id": get_categroy_id_from_label(label),
            "area": area,
            "bbox": bbox,
            "iscrowd": 0,
            "segmentation": segmentation
            }

    def get_annotations(self, image_id, id_prefix):
        traces = self.traces_data
        id = 0
        annotations = []
        for elem in traces:
            id += 1
            ls = elem['trace_group']
            label = elem.get("label")
            annotation = elem.get("__annotation__")
            if annotation == None:
                continue

            annotation["id"] = id_prefix + id
            annotation["image_id"] = image_id
            annotations.append(annotation)
        return annotations

    def save_image(self, output_path):
        traces = self.traces_data

        background_color = (255, 255, 255)
        img = np.zeros((N, N, 3), dtype=np.uint8)
        img = cv2.rectangle(img, (0, 0), (N, N), background_color, thickness=-1)

        for elem in traces:
            ls = elem['trace_group']
            thickness = 1

            # draw traces
            for subls in ls:
                data = np.array(subls)
                pts = np.array(data, np.int32)
                pts = pts.reshape((-1,1,2))
                img = cv2.polylines(img, [pts], False, (0, 0, 0), thickness=2)
                #img = cv2.rectangle(img, (round(minX), round(minY)), (round(maxX), round(maxY)), color, thickness)

        cv2.imwrite(output_path, img)

    def image_width(self):
        return N

    def image_height(self):
        return N

    def image_file_name(self):
        return self.image_filename

class InkMLDataSet(coco_dataset.CocoDataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir

    def load(self):
        super().load()
        with open(self.root_dir + "listInkML.txt") as file:
            for line in file:
                file_path = self.root_dir + line.strip()
                inkml_file = InkMLFile()
                inkml_file.load(file_path)
                self.add_item(inkml_file)

def processOneFile():
    import sys
    input_inkml = sys.argv[1]
    output_path = sys.argv[2]
    cv2inkml2img(input_inkml, output_path)

def processAllFile():
    dataset = InkMLDataSet("FCinkML/")
    dataset.load()

    print(len(dataset.items))
    annotation = dataset.get_annotation()
    print("xxxx: ", annotation)
    annotation_str = json.dumps(annotation, indent=2)
    with open("inkml_val.json", "w") as coco_json_file:
        coco_json_file.write(annotation_str)

    dataset.save_images("images")


if __name__ == "__main__":
    processAllFile()

    #input_inkml = 'FCinkML/test.inkml' #sys.argv[1]
    #output_path = 'test.png'#sys.argv[2]

    #inkml2img(input_inkml, output_path, color='#284054')


    #dataset.save_images("images")
