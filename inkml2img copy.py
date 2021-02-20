import numpy as np
import xml.etree.ElementTree as ET
from io import StringIO
import cv2
import coco_dataset
from enum import IntEnum
import os
import json
import sys
from scipy import stats
import math

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

def get_trace_bbox_xyxy(subls):
    minX = 1000000
    minY = 1000000
    maxX = -1000000
    maxY = -1000000

    for pt in subls:
        x = int(pt[0])
        y = int(pt[1])

        minX = min(x, minX)
        maxX = max(x, maxX)
        minY = min(y, minY)
        maxY = max(y, maxY)
    return minX, minY, maxX, maxY


arrow_trace_stat = dict()
'''
xxxx len 171
2 1845
3 539
4 133
1 523
6 9
5 36
7 3
'''


keypoint2_stat = dict()
print(keypoint2_stat)

NOT_LINE = 0
H_LINE = 1
V_LINE = 2

def get_trace_xy_buckets(subls):
    x_buckets = dict()
    y_buckets = dict()
    for pt in subls:
        x = int(pt[0])
        y = int(pt[1])

        x = (x // 10) * 10
        y = (y // 10) * 10

        x_buckets[x] = x_buckets.get(x, 0) + 1
        y_buckets[y] = y_buckets.get(y, 0) + 1

    return x_buckets, y_buckets

def get_np_xy(subls):
    x = []
    y = []
    for pt in subls:
        x.append(pt[0])
        y.append(pt[1])
    return x, y

def get_lines(subls):
    x,y = get_np_xy(subls)
    #w = np.polyfit(x, y, 1, False, True)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_value = r_value * r_value
    print(slope, intercept, r_value, p_value, std_err)

    slope, intercept, r_value, p_value, std_err = stats.linregress(y, x)
    r_value = r_value * r_value
    print(slope, intercept, r_value, p_value, std_err)

def get_atan2(subls):
    deltaN = 3
    angle_buckets = []
    for i in range(len(subls)):
        if i > deltaN:
            x1 = subls[i-deltaN][0]
            y1 = subls[i-deltaN][1]
            x2 = subls[i][0]
            y2 = subls[i][1]
            a = math.atan2(y2- y1, x2-x1)
            a = abs(180 * a / np.pi)
            a =  int(round(a / 10) * 10)
            angle_buckets.append(a)

    print("atan2: ", angle_buckets)



class Line:
    def __init__(self):
        self.type = NOT_LINE
        self.start = None
        self.end = None
        pass


def get_keypoint2(ls):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (128, 0, 128), (0, 0, 128), (128, 128, 0)]
    color_index = 0

    background_color = (0, 0, 0)
    img = np.zeros((N, N, 3), dtype=np.uint8)

    print("##########")
    str_info = ""
    lines = []
    for subls in ls:
        line_type = NOT_LINE
        color = (255, 255, 255)
        color_index += 1
        x1, y1, x2, y2 = get_trace_bbox_xyxy(subls)
        print("----------")
        print('bbox_w h', x2-x1, y2-y1)
        #get_lines(subls)
        get_atan2(subls)
        w = x2 - x1
        if w == 0:
            w = 0.01
        h = y2 - y1
        #print(x1, x2, y1, y2, w, h)
        #print(1.0 * h/w)
        K = int(100.0 * h / w)
        K = min(K, 10000)
        #print(K)

        str_info += "h: " + str(h) + " w: " + str(w) + " K: " + str(K) + "  |  "

        if (h > 30 and w > 30) or (h < 30 and w < 30 and K < 400 and K > 20):
            pass
        else:
            if h < 13 or K < 10:
                draw_arrow = True
                line = Line()
                line.type = H_LINE
                color = (255, 255, 0)
                cy = round((y1+y2)/2)
                if subls[0][0] > subls[-1][0]:
                    line.start = (x2, cy)
                    line.end = (x1, cy)
                else:
                    line.start = (x1, cy)
                    line.end = (x2, cy)
                lines.append(line)

            elif w < 13 or K > 400 : #V-Line
                line = Line()
                line.type = H_LINE
                color = (255, 255, 0)
                cx = round((x1+x2)/2)
                if subls[0][1] > subls[-1][1]:
                    line.start = (cx, y2)
                    line.end = (cx, y1)
                else:
                    line.start = (cx, y1)
                    line.end = (cx, y2)
                lines.append(line)
                color = (0, 0, 255)
            else:
                pass


        data = np.array(subls)
        fontScale = 1

        # Line thickness of 2 px
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        print("subls:", subls)
        img = cv2.polylines(img, [convertDataToPloyPts(data)], False, color, 2)

    img = cv2.putText(img,  str_info, (50, 50), font,
                fontScale, (255, 255, 255), thickness, cv2.LINE_AA)

    if len(lines) == 1:
        pass

    print("lines: ", lines)
    for line in lines:
        img = cv2.circle(img, line.start, 5, (0, 255, 255), thickness =2)
        img = cv2.circle(img, line.end, 10, (0, 255, 0), thickness =2)


    cv2.imshow("xxxx", img)
    cv2.waitKey()
    return None, None

    #img = cv2.drawContours(img, contours, -1, (128, 128, 128), 1)

def get_arrow_segmatationv2(id, ls):
    background_color = (0, 0, 0)
    img = np.zeros((N, N, 3), dtype=np.uint8)
    for subls in ls:
        data = np.array(subls)
        img = cv2.polylines(img, [convertDataToPloyPts(data)], False, (255, 255, 255), 6)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hiberachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    epsilon = 0.06 * cv2.arcLength(contours[0], True)
    approxCurve = cv2.approxPolyDP(contours[0], epsilon, True)

    print("approxCurve: ", approxCurve)
    #img = cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

    img = cv2.drawContours(img, [approxCurve], 0, (0, 255, 0), 3)

    cv2.imshow("xxx1111", img)
    cv2.waitKey()

    return segemation_from_contours(contours)

def get_arrow_segmatation(id, ls):
    background_color = (0, 0, 0)
    img = np.zeros((N, N, 3), dtype=np.uint8)
    for subls in ls:
        data = np.array(subls)
        img = cv2.polylines(img, [convertDataToPloyPts(data)], False, (255, 255, 255), 6)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hiberachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return segemation_from_contours(contours)

def get_keypoints(id, ls):
    background_color = (0, 0, 0)
    img = np.zeros((N, N, 3), dtype=np.uint8)

    #print("########", id, "len: ", len(ls))
    num_of_traces = len(ls)
    if num_of_traces == 2:
        return get_keypoint2(ls)
    else:
        return None, None

def get_keypoints_bak(id, ls):
    background_color = (0, 0, 0)
    img = np.zeros((N, N, 3), dtype=np.uint8)

    print("########", id, "len: ", len(ls))
    num_of_traces = len(ls)
    if num_of_traces == 2:
        return get_keypoint2(ls)

    arrow_trace_stat[len(ls)] = arrow_trace_stat.get(len(ls), 0) + 1
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (128, 0, 128), (0, 0, 128), (128, 128, 0)]
    color_index = 0
    for subls in ls:
        color = colors[color_index]
        color_index += 1
        x1, y1, x2, y2 = get_trace_bbox_xyxy(subls)
        w = x2 - x1
        h = y2 - y1
        #print(x1, x2, y1, y2, w, h, 1.0 * h/w)

        data = np.array(subls)

        count_N = 4
        #print('-----------------------------')
        for i in range(len(data)):
            if i < count_N:
                continue
            next_pt = data[i]
            curr_pt = data[i-count_N]
            #print(curr_pt, next_pt)
            #print("K:" , (next_pt[1] - curr_pt[1]) / (next_pt[0] - curr_pt[0]))
            curr_pt = data[i]

        next_pt = data[len(data) - 1]
        curr_pt = data[0]
        #print("Total K:" , (next_pt[1] - curr_pt[1]) / (next_pt[0] - curr_pt[0]))
        img = cv2.polylines(img, [convertDataToPloyPts(data)], False, (255, 255, 255), 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    #lines = cv2.HoughLinesP(binary, 1, np.pi/180, 1, minLineLength=30, maxLineGap=1)
    contours, hiberachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #img = cv2.drawContours(img, contours, -1, (128, 128, 128), 1)

    #cv2.imshow(id, img)
    #cv2.waitKey()
    return

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

        segs = []
        for subls in ls:
            data = np.array(subls)
            img = cv2.polylines(img, [convertDataToPloyPts(data)], False, get_label_color(label), thickness=2)

        if label == 'arrow':
            area, bbox, segmentation = get_arrow_segmatationv2(id, ls)
            #cat, keypoints = get_keypoints(id, ls)
            # fontScale
            fontScale = 1

            # Line thickness of 2 px
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX

            img = cv2.putText(img, elem["id"], (bbox[0], bbox[1]), font,
                   fontScale, (128, 0, 128), thickness, cv2.LINE_AA)

            #img = cv2.polylines(img, [convertDataToPloyPts(data)], False, get_label_color(label), thickness=2)
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1]), (128, 0, 128), thickness=2)

    cv2.imwrite(output_path, img)
    cv2.imshow("xxxxx", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

class InkMLFile(coco_dataset.CocoItem):
    def __init__(self):
        super().__init__()

    def load(self, file_path):
        #print("load : ", file_path)
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


            categroy_id = None
            if label == 'arrow':
                area, bbox, segmentation = get_arrow_segmatationv2(elem['id'], ls)
                #categroy_id, keypoints = get_keypoints(elem['id'], ls)


            if categroy_id != None:
                categroy_id = get_categroy_id_from_label(label)


            elem["__annotation__"] = {
            "category_id": categroy_id,
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
        self.file_list = "listInkML.txt"

    def load(self):
        super().load()
        with open(self.root_dir + self.file_list) as file:
            for line in file:
                file_path = self.root_dir + line.strip()
                inkml_file = InkMLFile()
                inkml_file.load(file_path)
                self.add_item(inkml_file)

def processOneFile():

    input_inkml = sys.argv[1]
    output_path = sys.argv[2]
    cv2inkml2img(input_inkml, output_path)


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
    processAllFile()
    #processDataset("listInkML_Dev.txt", "inkml_dev.json")

    for k, v in arrow_trace_stat.items():
        print(k, v)


    print("keypoint2_stat: ", keypoint2_stat)
    #input_inkml = 'FCinkML/test.inkml' #sys.argv[1]
    #output_path = 'test.png'#sys.argv[2]

    #inkml2img(input_inkml, output_path, color='#284054')


    #dataset.save_images("images")
