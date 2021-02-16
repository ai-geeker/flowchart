import numpy as np
from skimage.draw import line
from skimage.morphology import thin
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from io import StringIO
import cv2
def get_traces_data(inkml_file_abs_path, xmlns='{http://www.w3.org/2003/InkML}'):

        traces_data = []

        tree = ET.parse(inkml_file_abs_path)
        root = tree.getroot()
        # doc_namespace = "{http://www.w3.org/2003/InkML}"
        doc_namespace = xmlns
        k = 1000
        'Stores traces_all with their corresponding id'
        traces_all = [{'id': trace_tag.get('id'),
                        'coords': [[(float(axis_coord)) \
                                        for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
                                    else [(float(axis_coord)) \
                                        for axis_coord in coord.split(' ')] \
                                for coord in (trace_tag.text).replace('\n', '').split(',')]} \
                                for trace_tag in root.findall(doc_namespace + 'trace')]

        'Sort traces_all list by id to make searching for references faster'
        traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))

        'Always 1st traceGroup is a redundant wrapper'
        traceGroupWrapper = root.find(doc_namespace + 'traceGroup')

        if traceGroupWrapper is not None:
            for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):

                label = traceGroup.find(doc_namespace + 'annotation').text

                'traces of the current traceGroup'
                traces_curr = []
                for traceView in traceGroup.findall(doc_namespace + 'traceView'):

                    'Id reference to specific trace tag corresponding to currently considered label'
                    traceDataRef = int(traceView.get('traceDataRef'))
                    print(traceDataRef)

                    'Each trace is represented by a list of coordinates to connect'


                    single_trace = traces_all[traceDataRef]['coords']
                    traces_curr.append(single_trace)


                traces_data.append({'label': label, 'trace_group': traces_curr})

        else:
            'Consider Validation data that has no labels'
            [traces_data.append({'trace_group': [trace['coords']]}) for trace in traces_all]

        return traces_data

def convert_to_imgs(traces_data, box_size=int(100)):

        patterns_enc = []
        classes_rejected = []

        for pattern in traces_data:

            trace_group = pattern['trace_group']

            'mid coords needed to shift the pattern'
            min_x, min_y, max_x, max_y = get_min_coords(trace_group)

            'traceGroup dimensions'
            trace_grp_height, trace_grp_width = max_y - min_y, max_x - min_x

            'shift pattern to its relative position'
            shifted_trace_grp = shift_trace_grp(trace_group, min_x=min_x, min_y=min_y)

            'Interpolates a pattern so that it fits into a box with specified size'
            'method: LINEAR INTERPOLATION'
            try:
                interpolated_trace_grp = interpolate(shifted_trace_grp, \
                                                     trace_grp_height=trace_grp_height, trace_grp_width=trace_grp_width, box_size=box_size - 1)
            except Exception as e:
                print(e)
                print('This data is corrupted - skipping.')
                classes_rejected.append(pattern.get('label'))

                continue

            'Get min, max coords once again in order to center scaled patter inside the box'
            min_x, min_y, max_x, max_y = get_min_coords(interpolated_trace_grp)

            centered_trace_grp = center_pattern(interpolated_trace_grp, max_x=max_x, max_y=max_y, box_size=box_size)

            'Center scaled pattern so it fits a box with specified size'
            pattern_drawn = draw_pattern(centered_trace_grp, box_size=box_size)
            # Make sure that patterns are thinned (1 pixel thick)
            pat_thinned = 1.0 - thin(1.0 - np.asarray(pattern_drawn))
            plt.imshow(pat_thinned, cmap='gray')
            plt.show()
            pattern_enc = dict({'features': pat_thinned, 'label': pattern.get('label')})

            # Filter classes that belong to categories selected by the user
#             if pattern_enc.get('label') in self.classes:

            patterns_enc.append(pattern_enc)

        return patterns_enc, classes_rejected

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


def shift_trace_grp(trace_group, min_x, min_y):

        shifted_trace_grp = []

        for trace in trace_group:
            shifted_trace = [[coord[0] - min_x, coord[1] - min_y] for coord in trace]

            shifted_trace_grp.append(shifted_trace)

        return shifted_trace_grp

def interpolate(trace_group, trace_grp_height, trace_grp_width, box_size):

        interpolated_trace_grp = []

        if trace_grp_height == 0:
            trace_grp_height += 1
        if trace_grp_width == 0:
            trace_grp_width += 1

        '' 'KEEP original size ratio' ''
        trace_grp_ratio = (trace_grp_width) / (trace_grp_height)

        scale_factor = 1.0
        '' 'Set \"rescale coefficient\" magnitude' ''
        if trace_grp_ratio < 1.0:

            scale_factor = (box_size / trace_grp_height)
        else:

            scale_factor = (box_size / trace_grp_width)

        for trace in trace_group:
            'coordintes convertion to int type necessary'
            interpolated_trace = [[round(coord[0] * scale_factor), round(coord[1] * scale_factor)] for coord in trace]

            interpolated_trace_grp.append(interpolated_trace)

        return interpolated_trace_grp

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
def center_pattern(trace_group, max_x, max_y, box_size):

        x_margin = int((box_size - max_x) / 2)
        y_margin = int((box_size - max_y) / 2)

        return shift_trace_grp(trace_group, min_x= -x_margin, min_y= -y_margin)

def draw_pattern(trace_group, box_size):

        pattern_drawn = np.ones(shape=(box_size, box_size), dtype=np.float32)
        for trace in trace_group:

            ' SINGLE POINT TO DRAW '
            if len(trace) == 1:
                x_coord = trace[0][0]
                y_coord = trace[0][1]
                pattern_drawn[y_coord, x_coord] = 0.0

            else:
                ' TRACE HAS MORE THAN 1 POINT '

                'Iterate through list of traces endpoints'
                for pt_idx in range(len(trace) - 1):
                    print (pt_idx, trace[pt_idx])

                    'Indices of pixels that belong to the line. May be used to directly index into an array'
                    pattern_drawn[line(r0=int(trace[pt_idx][1]), c0=int(trace[pt_idx][0]),
                                       r1=int(trace[pt_idx + 1][1]), c1=int(trace[pt_idx + 1][0]))] = 0

        return pattern_drawn


def inkml2img(input_path, output_path, color='black'):
    traces = get_traces_data(input_path)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.axes().spines['top'].set_visible(False)
    plt.axes().spines['right'].set_visible(False)
    plt.axes().spines['bottom'].set_visible(False)
    plt.axes().spines['left'].set_visible(False)
    colors = ['#ff0000', '#ffff00', '#00ff00', '#0000ff']
    color_index = 0
    for elem in traces:
        ls = elem['trace_group']
        color_index = 0
        for subls in ls:
            data = np.array(subls)
            x,y=zip(*data)
            print("(x, y) = \n", x, y)
            plt.plot(x,y,linewidth=2,c=colors[color_index])

            if color_index < 3:
                color_index += 1
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.gcf().clear()


def cv2inkml2img(input_path, output_path, color='black'):
    traces = get_traces_data(input_path)
    N = 2000
    background_color = (255, 255, 255)
    img = np.zeros((N, N, 3), dtype=np.uint8)
    img = cv2.rectangle(img, (0, 0), (N, N), background_color, thickness=-1)

    for elem in traces:
        ls = elem['trace_group']
        color_index = 0
        for subls in ls:
            data = np.array(subls)
            print('-------------')
            print(data)
            print('-------------')
            #data = data.round()
            # pts=np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
            pts = np.array(data, np.int32)
            pts = pts.reshape((-1,1,2))
            #data = np.array([[910,641],[206 ,632],[696, 488],[458, 485]])
            #data = data.reshape((-1,1,2))
            print('-------------')
            print(data)
            img = cv2.polylines(img, [pts], False, (0, 0, 0), thickness=3)
            print('-------------')


    #cv2.imshow("xxxx", img)
    cv2.waitKey()
    cv2.imwrite("test3.png", img)

if __name__ == "__main__":
    import sys
    #input_inkml = 'FCinkML/test.inkml' #sys.argv[1]
    #output_path = 'test.png'#sys.argv[2]
    input_inkml = sys.argv[1]
    output_path = sys.argv[2]
    #inkml2img(input_inkml, output_path, color='#284054')
    cv2inkml2img(input_inkml, output_path)