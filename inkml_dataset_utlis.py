
from enum import IntEnum
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
        return ChartCategory.Ellipse
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