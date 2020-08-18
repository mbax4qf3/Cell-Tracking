import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import time
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

### each frame has an info entry，info contains cells' info
### info[0] labels
### info[1] centers
### info[2] areas
### info[3] bounding box
info = np.empty([500], dtype = np.ndarray)
### frame counting
num = 0
### line_img - to store all the past trace
img_path_tmp = "./COMP9517 20T2 Group Project Image Sequences/PhC-C2DL-PSC/Sequence 1/t000.tif"
img_tmp = cv.imread(img_path_tmp,0)
line_img = np.zeros(img_tmp.shape, np.uint8)

### data for task 3
total_dist = []
net_dist = []

get_current_speed = {}
get_total_dist = {}
get_net_dist = {}
get_ratio = {}
mitosis_cnt = {}

## init for each sequence
def init():
    global info, num, total_dist, net_dist, get_current_speed, get_total_dist, get_net_dist, get_ratio
    info = np.empty([500], dtype = np.ndarray)
    num = 0
    total_dist = []
    net_dist = []

    line_img[line_img != 0] = 0
    
    get_current_speed = {}
    get_total_dist = {}
    get_net_dist = {}
    get_ratio = {}
    mitosis = {}
    
### h-maxima
def HMax(src, dst, h, kernel):
    mask = np.zeros(src.shape, np.uint8)

    cv.subtract(src, np.zeros(src.shape, np.uint8) + h, mask)
    cv.min(src, mask, dst)
    cv.dilate(dst, kernel, dst)
    cv.min(src, dst, dst)
    
    tmp1 = np.zeros(src.shape, np.uint8)
    tmp2 = np.zeros(src.shape, np.uint8)
    
    while True:
        tmp1 = np.copy(dst)
        cv.dilate(dst, kernel, dst)
        cv.min(src, dst, dst)
        cv.compare(tmp1, dst, cv.CMP_NE, tmp2)
        if cv.sumElems(tmp2)[0] == 0:
            break

    return dst

### main body for functions
def label_cells(img, cnt,dataset):
    global info, num, line_img, total_dist, net_dist, get_current_speed, get_total_dist, get_net_dist, get_ratio, mitosis
    
    #img_path = "./COMP9517 20T2 Group Project Image Sequences/PhC-C2DL-PSC/Sequence " + dataset + "/" + cnt + ".tif"
    #img = cv.imread(img_path,0)
    colorimg = img.copy()
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ### top-hat
    kernel = np.ones((13,13),np.uint8)
    top_hat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    #remove background noise
    top_hat[top_hat<60]=0

    ### top_hat_binary and top_hat_binary_remove_small are used to remove small objects like dust
    top_hat_binary = top_hat.copy()
    top_hat_binary[top_hat_binary>0]=255
    top_hat_binary = top_hat_binary > 0

    top_hat_binary_remove_small = morphology.remove_small_objects(top_hat_binary, min_size = 20,connectivity = 1)
    top_hat_binary_remove_small = np.multiply(top_hat_binary_remove_small, 1)
    top_hat_binary_remove_small[top_hat_binary_remove_small==1]=255
    top_hat_binary_remove_small_type = top_hat_binary_remove_small.astype(np.uint8)
    
    top_hat_binary_remove_small[top_hat_binary_remove_small>0]=1
    top_hat_tmp = np.uint8(np.multiply(top_hat, top_hat_binary_remove_small))

    ### gaussian filter
    gaussian = cv.GaussianBlur(top_hat_tmp, (0,0), sigmaX = 2)

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    h_max_recon = np.zeros(gaussian.shape, np.uint8)

    HMax(gaussian, h_max_recon, 7, kernel)

    maxi = np.subtract(gaussian, h_max_recon)
    maxi[maxi>0] = 255 
   
    color = (0,255,0)
    framed_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    
    ### watershed
    kk = np.ones((3,3),np.uint8)
    # background
    sure_bg = cv.dilate(top_hat_binary_remove_small_type,kk,iterations=3)

    # foreground
    dist_transform = cv.distanceTransform(top_hat_binary_remove_small_type,cv.DIST_L2,5)

    # search unknown
    sure_fg = np.uint8(maxi)
    unknown = cv.subtract(sure_bg,sure_fg)

    # labels
    ret2, markers1 = cv.connectedComponents(sure_fg)
    markers = markers1+1
    markers[unknown==255] = 0
    
    img_water = colorimg
    img_water= img_water.astype(np.uint8)
    markers3 = cv.watershed(img_water, markers)
    img_water[markers3 == -1] = [0, 0, 255]

    pure_cell = np.zeros(markers3.shape,np.uint8)
    outline = np.zeros(markers3.shape,np.uint8)
    
    pure_cell[markers3!=1]=255
    
    outline[markers3==-1]=255
    outline[markers3!=-1]=0
    
    ## cells separated with original size
    cell_seg = pure_cell - outline
    
    num_labels, labels, stats, centers = cv.connectedComponentsWithStats(cell_seg,connectivity = 4)
    info[num] = np.empty([4], dtype = np.ndarray)
    info[num][0] = labels
    info[num][1] = centers
    info[num][2] = stats[:,4] # area
    info[num][3] = stats[:,:4] # bounding box
    num += 1
    
    ### used for task 3
    key_label = list(range(1,num_labels))
    value_dist = [0] * (num_labels-1)
    dict1 = dict(zip(key_label, value_dist))
    total_dist.append(dict1)
        
    init_centers = centers[1:].tolist()
    dict2 = dict(zip(key_label, init_centers))
    net_dist.append(dict2)
    
    # output img
    color = (0,255,255)
    framed_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    framed_img[line_img==1] = [255,0,0]
    
    seg_padding = cell_seg.copy()
    seg_padding = np.pad(seg_padding,((15,15),(15,15)),'constant',constant_values = (0,0))
    
    ### previous frame for fixing mitosis
    pre_img_num = "00" + str(num-2)
    pre_img_path = "./out_trace/data3_s" + dataset + "/labeledt" + pre_img_num[-3:] + ".tif"
    pre_img = None
    if num >= 2:
        pre_img = cv.imread(pre_img_path)
    
    mitosis_cnt[num-1] = 0
    ### task 1 frame cells
    for t in range(1,num_labels):
        x,y,w,h,area=stats[t]
        if w > 50:
            continue
        cv.rectangle(framed_img, (x,y), (x+w, y+h),color,1)
        cv.putText(framed_img, str(t), (x, y), cv.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
    cv.putText(framed_img, "number of cells: "+ str(num_labels), (40, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    
    ### from the second frame
    if num > 1:
        n_1_frame = np.max(info[num-2][0]) + 1
        n_frame = np.max(info[num-1][0]) + 1
        
        matrix = np.zeros([n_1_frame,n_frame], np.float32)
        dist_ = np.zeros([n_1_frame,n_frame], np.float32)
        areas_ = np.zeros([n_1_frame,n_frame], np.float32)
        # overlap has few effect but make the program so slow
        #overlap_ = np.zeros([n_1_frame,n_frame], np.float32)
        
        # make matrix for matching
        for i in range(n_1_frame):
            for j in range(n_frame):
                if i != 0 and j != 0:
                    # DIST
                    a=np.square(info[num-2][1][i][0] - info[num-1][1][j][0])
                    b=np.square(info[num-2][1][i][1] - info[num-1][1][j][1])
                    c=np.sqrt(a+b)
                    dist_[i][j] = c / (np.sqrt(np.square(img.shape[0]) + np.square(img.shape[1])))
                    # AREA same if close to 0
                    areas_[i][j] = abs(1 - (info[num-2][2][i] / info[num-1][2][j]))
                    # OVERLAP 部分，太慢，效果一般，暂时不用
                    #n_1_label = np.zeros(img.shape, np.uint8)
                    #n_label = np.zeros(img.shape, np.uint8)
                    #n_1_label[info[num-2][0] == i] = 1
                    #n_label[info[num-1][0] == j] = 1
                    #overlap = np.multiply(n_1_label, n_label)             
                    #overlap_area = np.sum(overlap)
                    #print(overlap_area)
                    #overlap_[i][j] = 1 - ( (overlap_area * overlap_area) / (info[num-2][2][i] * info[num-1][2][j]))
                #matrix[i][j] = 0.9*dist[i][j] + 0.1*areas[i][j]
        dist_ = normalize(dist_, axis=0, norm='max')
        
        # build matrix
        matrix = 5 * dist_ + 0.1 * areas_# + 0.15 * overlap_
        # find matching
        row_ind, col_ind = linear_sum_assignment(matrix)
        # find miss matching for detecting mitosis
        find_miss1 = list(range(num_labels))
        find_miss = list(set(find_miss1)-set(col_ind))
        
        # for each pair of matching
        for k in range(col_ind.size):
            if k == 0 or col_ind[k] == 0 or row_ind[k] == 0:
                continue
            n_1_choice = row_ind[k]
            n_choice = col_ind[k]
            
            # matching points pair
            start_x = int(info[num-2][1][n_1_choice][0])
            start_y = int(info[num-2][1][n_1_choice][1])
            end_x = int(info[num-1][1][n_choice][0])
            end_y = int(info[num-1][1][n_choice][1])
            
            # if dist is too large, update mismatching
            if int(np.square(start_x-end_x) + np.square(start_y-end_y)) > 8000 or n_choice == 0 or n_1_choice == 0:
                find_miss.append(n_choice)
                continue
            if int(np.square(start_x-end_x) + np.square(start_y-end_y)) > 250:
                continue
            # record trace
            cv.line(line_img, (start_x,start_y), (end_x,end_y), 1, 1, 4)
            cv.line(framed_img, (start_x,start_y), (end_x,end_y), (255,0,0), 1, 4)
            # for task 3
            if n_choice != 0 and n_1_choice != 0:
                #get_current_speed, get_total_dist, get_net_dist, get_ratio
                # speed
                cell_dist = int(np.sqrt(np.square(start_x-end_x) + np.square(start_y-end_y)))
                get_current_speed[(num-1, n_choice)] = cell_dist
                #cv.putText(framed_img, "s "+str(cell_dist), (end_x+info[num-1][3][n_choice][2], end_y), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
                # total dist
                total_dist[-1][n_choice] = total_dist[-2][n_1_choice] + cell_dist
                get_total_dist[(num-1, n_choice)] = total_dist[-1][n_choice]
                #cv.putText(framed_img, "t "+str(total_dist[-1][n_choice]), (end_x+info[num-1][3][n_choice][2], end_y+10), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
                # net dist
                net_dist[-1][n_choice] = net_dist[-2][n_1_choice]
                tmp_sqr_dist1 = int(np.square(int(end_x) - int(net_dist[-1][n_choice][0])))
                tmp_sqr_dist2 = int(np.square(int(end_y) - int(net_dist[-1][n_choice][1])))
                current_net_dist = int(np.sqrt(tmp_sqr_dist1+tmp_sqr_dist2))
                get_net_dist[(num-1, n_choice)] = current_net_dist
                #cv.putText(framed_img, "n "+str(current_net_dist), (end_x+info[num-1][3][n_choice][2], end_y+20), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
                # ratio
                if current_net_dist < 0.01:
                    get_ratio[(num-1, n_choice)] = 0.0
                else:
                    ratio = round(1.0 * total_dist[-1][n_choice] / current_net_dist, 2)
                    get_ratio[(num-1, n_choice)] = ratio
                #cv.putText(framed_img, "r "+str(get_ratio[(num-1, n_choice)]), (end_x+info[num-1][3][n_choice][2], end_y+30), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
            '''
            # Previous method for detecting mitosis, abandoned
            elif int(np.square(start_x-end_x) + np.square(start_y-end_y)) > 60:
                #cv.rectangle(framed_img, (start_x-10,start_y-10), (start_x+10, start_y+10),(255,255,0),1)
                #cv.rectangle(framed_img, (end_x-10,end_y-10), (end_x+10, end_y+10),(255,255,255),1)
                if end_x <= 700 or end_y <= 576:
                    # search four directions for 15 pixels
                    # SEARCH area of x+-15 y+-15 in current frame
                    roi_mask_cur = np.ones([31,31], np.uint32)
                    xx = end_x #if end_x <= 700 else 700
                    yy = end_y #if end_y <= 576 else 576
                    
                    # search labels
                    roi_mask_cur = np.multiply(roi_mask_cur, seg_padding[yy:yy+31,xx:xx+31])
                    current_cnt = np.unique(roi_mask_cur).shape
                    #cv.rectangle(framed_img, (end_x-15,end_y-15), (end_x+15, end_y+15),(255,255,0),1)
                    roi_mask_cur[roi_mask_cur!=0] = 1
                    current_area = np.sum(roi_mask_cur)
                    #print("current_cnt", current_cnt)
                    
                    # search labels for previous frame
                    # SEARCH area of x+-15 y+-15 in previous frame
                    roi_mask_pre = np.ones([31,31], np.uint32)
                    pre_padding = info[num-2][0].copy()
                    pre_padding = np.pad(pre_padding,((15,15),(15,15)),'constant',constant_values = (0,0))
                    roi_mask_pre = np.multiply(roi_mask_pre, pre_padding[yy:yy+31,xx:xx+31])
                    previous_cnt = np.unique(roi_mask_pre).shape
                    roi_mask_pre[roi_mask_pre!=0] = 1
                    previous_area = np.sum(roi_mask_pre)
                    #print("previous_c", previous_cnt)
                    #print()
                    
                    # detection consition
                    #if (previous_cnt[0] == 2 and current_cnt[0] > 2) or (previous_cnt[0] > 2 and current_cnt[0] >= previous_cnt[0]):
                    #if (previous_cnt[0] > current_cnt[0] and current_cnt[0]>=2):
                    #if previous_cnt[0] > 2:
                        #if abs(current_area - previous_area) < 0.2 * current_area:
                        #cv.rectangle(framed_img, (end_x-10,end_y-10), (end_x+10, end_y+10),(0,0,255),1)
                        #mitosis_cnt += 1
                        #continue
            '''
        # detect mitosis, for those miss matching cells, search near area, check other labels
        for i in find_miss:
            if i != 0:       
                s_w_tl = 0 if int(info[num-1][1][i][0]) - 10 < 0 else int(info[num-1][1][i][0]) - 10
                s_w_tr = 1099 if int(info[num-1][1][i][0]) + 10 > 719 else int(info[num-1][1][i][0]) + 10
                s_w_bl = 0 if int(info[num-1][1][i][1]) - 10 < 0 else int(info[num-1][1][i][1]) - 10
                s_w_br = 699 if int(info[num-1][1][i][1]) + 10 > 575 else int(info[num-1][1][i][1]) + 10
                search_window = labels[s_w_bl:s_w_br, s_w_tl:s_w_tr]
                if (np.unique(search_window).shape[0] <= 2):
                    pass
                else:
                    if info[num-1][3][i][2] > 50:
                        continue
                    cv.rectangle(framed_img, (info[num-1][3][i][0],info[num-1][3][i][1]), \
                             (info[num-1][3][i][0] + info[num-1][3][i][2], \
                              info[num-1][3][i][1] + info[num-1][3][i][3]),(255,0,255),1)
                    # restart daughtaur cell
                    get_current_speed[(num-1, i)] = 0
                    get_total_dist[(num-1,i)] = 0
                    get_net_dist[(num-1,i)] = 0
                    net_dist[-1][i] = info[num-1][1][i]
                    get_ratio[(num-1,i)] = 0
                    total_dist[-1][i] = 0
                    
                    near_cells = np.unique(search_window)
                    nearest = 9999
                    friend = None
                    # find the nearest cell
                    for near in range(near_cells.shape[0]):
                        if near_cells[near] == 0 or near_cells[near] == i:
                            pass
                        else:
                            cell_dist1 = np.square(info[num-1][1][i][0] - info[num-1][1][near_cells[near]][0])
                            cell_dist2 = np.square(info[num-1][1][i][1] - info[num-1][1][near_cells[near]][1])
                            cell_dist = np.sqrt(cell_dist1 + cell_dist2)
                            if cell_dist < nearest:
                                friend = near_cells[near]
                    pre_label_ind = np.where(col_ind == friend)
                    pre_label = row_ind[pre_label_ind[0]]
                    if pre_label.size > 0:
                        # restart daughtaur cell 
                        get_current_speed[(num-1, friend)] = 0
                        get_total_dist[(num-1,friend)] = 0
                        get_net_dist[(num-1,friend)] = 0
                        net_dist[-1][friend] = info[num-1][1][friend]
                        get_ratio[(num-1,friend)] = 0
                        total_dist[-1][friend] = 0
                        
                        mitosis_cnt[num-1] += 1
                        mitosis_cnt[num-2] += 1
                        # frame brother cell and parent cell
                        if info[num-1][3][friend][3] > 50 or info[num-2][3][pre_label][0][3] > 50:
                            continue
                        cv.rectangle(framed_img, (info[num-1][3][friend][0],info[num-1][3][friend][1]), \
                                    (info[num-1][3][friend][0] + info[num-1][3][friend][2], \
                                    info[num-1][3][friend][1] + info[num-1][3][friend][3]),(255,0,255),1)
                        cv.rectangle(pre_img, (info[num-2][3][pre_label][0][0],info[num-2][3][pre_label][0][1]), \
                                    (info[num-2][3][pre_label][0][0] + info[num-2][3][pre_label][0][2], \
                                    info[num-2][3][pre_label][0][1] + info[num-2][3][pre_label][0][3]),(255,0,255),1)
    # update previous frame            
    if num >= 2:
        cv.putText(pre_img, "number of mitosis: "+ str(mitosis_cnt[num-2]), (40, 75), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv.imwrite(pre_img_path, pre_img)
    # output current frame
    if num-1 == 425:
        cv.putText(framed_img, "number of mitosis: "+ str(mitosis_cnt[num-1]), (40, 75), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    output_path = "./out_trace/data3_s" + dataset + "/labeled" + cnt + ".tif"
    cv.imwrite(output_path, framed_img)
