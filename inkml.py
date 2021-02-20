import xml.etree.ElementTree as ET

class TraceGroup:
    def __init__(self):
        self.annotation_href = ""
        self.id = ""
        self.label = ""
        self.traces = []
        self.bbox = []

    def __str__(self):
        s = "label: " + self.label + " id: " + self.id + " annotation_href: " + self.annotation_href
        a_s = self.additional_info()
        if len(a_s) > 0:
            s += "\t" + "info: " + a_s
        return s

    def additional_info(self):
        return ""

class Text(TraceGroup):
    def __init__(self):
        super().__init__()

class Symbol(TraceGroup):
    def __init__(self):
        super().__init__()

class Arrow(TraceGroup):
    def __init__(self):
        super().__init__()
        self.source = None
        self.target = None

    def additional_info(self):
        return "source: " + str(self.source) + " target: " + str(self.target)

N = 800
MARGIN = 10
K = (N - 2 * MARGIN) / 2000.0

xml = '{http://www.w3.org/XML/1998/namespace}'

xml_id = xml + "id"

def get_trace(traces_all, id):
    for trace in traces_all:
        if trace["id"] == id:
            return trace
    return None

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

        return [min(min_x_coords), min(min_y_coords), max(max_x_coords), max(max_y_coords)]

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

    href_trace_groups = dict()
    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):

            label = traceGroup.find(doc_namespace + 'annotation').text
            trace_troup = None
            if label == 'arrow':
                trace_troup = Arrow()
            elif label == "text":
                trace_troup = Text()
            else:
                trace_troup = Symbol()

            trace_troup.label = label

            traceGroupId = traceGroup.get(xml_id)

            trace_troup.id = traceGroupId

            trace_troup_annotation_xml = traceGroup.find(doc_namespace + 'annotationXML')
            if trace_troup_annotation_xml != None:
                trace_troup.annotation_href =  traceGroup.find(doc_namespace + 'annotationXML').get("href")

            'traces of the current traceGroup'
            traces_curr = []
            for traceView in traceGroup.findall(doc_namespace + 'traceView'):
                'Id reference to specific trace tag corresponding to currently considered label'
                traceDataRef = (traceView.get('traceDataRef'))

                'Each trace is represented by a list of coordinates to connect'

                single_trace = get_trace(traces_all, traceDataRef)['coords']
                traces_curr.append(single_trace)
            trace_troup.traces = traces_curr
            trace_troup.bbox = get_min_coords(traces_curr)
            traces_data.append(trace_troup)
            if  trace_troup.annotation_href != None:
                href_trace_groups[trace_troup.annotation_href] = trace_troup

    else:
        'Consider Validation data that has no labels'
        pass

    # now parse the annotationXML for arrow
    annotationXMLNode = root.find(doc_namespace + "annotationXML")
    if annotationXMLNode != None:
        flowchar_xmlns = '{LUNAM/IRCCyN/FlowchartML}'
        arrows = annotationXMLNode.find( flowchar_xmlns + "flowchart").findall(flowchar_xmlns + 'arrow')
        for a in arrows:
            href_id =a.get(xml_id)
            source = a.get("source")
            target = a.get("target")
            arrow_trace_group = href_trace_groups.get(href_id)
            if arrow_trace_group != None:
                arrow_trace_group.source = href_trace_groups.get(source)
                arrow_trace_group.target = href_trace_groups.get(target)

    return traces_data

if __name__ == "__main__":
    import sys
    input_inkml = sys.argv[1]
    output_path = sys.argv[2]
    trace_groups = get_traces_data(input_inkml)
    for tg in trace_groups:
        print(tg)


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

