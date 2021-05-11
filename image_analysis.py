from scipy import ndimage
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

def preprocess(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    BW_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    
    return BW_image


def find_hline(BW_image, length, row=0): 
    
    n_columns = BW_image.shape[1]
    line_segments = []
    for i_col in range(n_columns):
        if BW_image[row, i_col:i_col+length].sum() == 0:
            line_segments.append(i_col)
    
    if line_segments == []:
        return None
    else:
        line = [line_segments[0], row, line_segments[-1], row]
    
    return line

def find_vline(BW_image, length, col=0): 

    image = BW_image.T
    n_rows = image.shape[1]
    
    line_segments = []
    for i_row in range(n_rows):
        if image[col, i_row:i_row+length].sum() == 0:
            line_segments.append(i_row)
    
    if line_segments == []:
        return None
    else:
        line = [col, line_segments[0], col, line_segments[-1]]
    
    return line

def hlines(image, length):
    
    lines = []
    for i in range(image.shape[0]):
        new_line = find_hline(image, length, i)
        if new_line == None:
            continue
        else: 
            lines.append(new_line)
            
    return lines

def vlines(image, length):
    
    lines = []
    for i in range(image.shape[1]):
        new_line = find_vline(image, length, i)
        if new_line == None:
            continue
        else: 
            lines.append(new_line)
            
    return lines


def strong_lines(image, n, position, h=2, l=20, long=10): # h - hight of window, l - length of window
    
    lines = []
    for i in range(image.shape[0]):
        line = []
        for j in range(image.shape[1]):
            if image[i:i+h, j:j+l].sum() == 0:
                line.append(j)
            else:
                continue
            lines.append([i, line[0], i, line[-1]])
    
    long_lines = []
    for l in lines:
        if l[3] - l[1] > long:
            long_lines.append(l)
            
    rows = []
    for l in long_lines:
        if l[0] in rows:
            continue
        else:
            rows.append(l[0])

    for indent in range(2, 1000):
        strong_lines = []
        for i in range(len(rows)):
            if i == 0: # для первой горизонтали создаем первый кластер
                strong = [rows[i]] 
            elif i == len(rows): # если дошли до последней горизомнтали
                strong_lines.append(strong)
            elif rows[i] - rows[i-1] < indent:
                strong.append(rows[i])
                if i == len(rows)-1: # если горизонталь последняя, добавляем кластер в общий список
                    strong_lines.append(strong)
            else:
                strong_lines.append(strong)
                strong = [rows[i]]
                if i == len(rows)-1: # если горизонталь последняя, добавляем кластер в общий список
                    strong_lines.append(strong)
        if len(strong_lines) < n:
            return print('We cannot find', n, " ", position, 'lines. Try to make different photo, or redraw lines (make them more bold and direct).')
        elif len(strong_lines) == n:
            break
        else:
            continue
    
    final_lines = []
    if position == 'vertical':
        for i in strong_lines:
            i = round(np.array(i).mean())
            final_lines.append([i, 0, i, image.shape[1]])
    elif position == 'horisontal':
        for i in strong_lines:
            i = round(np.array(i).mean())
            final_lines.append([0, i, image.shape[1], i])
    else: print('Please, set "horisontal" or "vertical" on position argument')
            
    return np.array(final_lines, dtype = 'int')


def set_cage(image, n, m, h=3, l=10, long=10):
    
    hl = hlines(image, 20)
    vl = vlines(image, 20)
    
    baseh = np.full(image.shape, 255, dtype='uint8')
    basev = np.full(image.shape, 255, dtype='uint8')
    
    for line in hl:
        cv2.line(baseh, (line[0], line[1]), (line[2], line[3]), (0,0,0), 2) # there 2 is thickness of line, can be set as parameter in the future
        
    for line in vl:
        cv2.line(basev, (line[0], line[1]), (line[2], line[3]), (0,0,0), 2) 
    
    horisontals = strong_lines(baseh, n, 'horisontal', h, l, long)
    verticals = strong_lines(basev.T, m, 'vertical', h, l, long)
    
    cage = np.full(image.shape, 255, dtype='uint8')

    for line in horisontals:
        cv2.line(cage, (line[0], line[1]), (line[2], line[3]), (0, 0, 0), 2)

    for line in verticals:
        cv2.line(cage, (line[0], line[1]), (line[2], line[3]), (0, 0, 0), 2)
        
    coordinates = [[horisontals], [verticals]]

    return cage, coordinates

def squares(coordinates):

    horisontals = coordinates[0]
    verticals = coordinates[1]

    intersections = []
    for i in horisontals[0]:
        intersection = []
        for j in verticals[0]:
            intersection.append([i[1], j[0]])
        intersections.append(intersection)

    squares_list = []
    for i in range(len(intersections)-1):
        for j in range(len(intersections[0])-1):
            squares_list.append([intersections[i][j], intersections[i+1][j], intersections[i][j+1], intersections[i+1][j+1]])
    return squares_list

def binary_classification(squares, prep_image):

    sq_list = []
    for i in squares:
        sq = prep_image[i[0][0]:i[1][0], i[0][1]:i[2][1]]
        sq_list.append(sq.mean())
    
    sq_list = np.array(sq_list).reshape(-1,1)
    
    kmeans_model = KMeans(n_clusters=2, random_state=1).fit(sq_list)
    labels = kmeans_model.labels_
    
    class_score_0 = 0
    count_0 = 0
    for n, i in enumerate(labels):
        if i == 0:
            class_score_0 += sq_list[n]
            count_0 += 1
        else:
            continue
            
    class_score_1 = 0
    count_1 = 0
    for n, i in enumerate(labels):
        if i == 1:
            class_score_1 += sq_list[n]
            count_1 += 1
        else:
            continue
            
            
    norm_class_score_0 = class_score_0/count_0
    norm_class_score_1 = class_score_1/count_1
    
    class_names = []
    if norm_class_score_0 <= norm_class_score_1:
        class_names.append('FILLED')
        class_names.append('EMPTY')
    else:
        class_names.append('EMPTY')
        class_names.append('FILLED')
    
            
    c = 1
    for n, i in enumerate(squares):
        sq = prep_image[i[0][0]:i[1][0], i[0][1]:i[2][1]]
        print(c)
        print("MEAN:",sq.mean())
        print('Class: ', class_names[labels[n]])
        plt.imshow(sq, cmap='gray')
        plt.show()
        c+=1
    
    return labels, class_names

