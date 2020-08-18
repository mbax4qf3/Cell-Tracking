import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import time
from scipy.optimize import linear_sum_assignment

np.set_printoptions(threshold=np.inf)

### each frame has an info entry，info contains cells' info
### info[frame][0] labels
### info[1] centers
### info[2] areas
### info[3] intensity
### info[4] bounding box
info = np.empty([500], dtype = np.ndarray)
### frame counting
num = 0
### line_img - to store all the past trace
img_path_tmp = "./COMP9517 20T2 Group Project Image Sequences/Fluo-N2DL-HeLa/Sequence 1/t000.tif"
img_tmp = cv.imread(img_path_tmp,0)
line_img = np.zeros(img_tmp.shape, np.uint8)

### data for task 3
total_dist = []
net_dist = []

get_current_speed = {}
get_total_dist = {}
get_net_dist = {}
get_ratio = {}
mitosis = {}

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

def label_cells(img, cnt,dataset):
    global info, num, line_img, total_dist, net_dist, get_current_speed, get_total_dist, get_net_dist, get_ratio, mitosis
    #img_path = "./COMP9517 20T2 Group Project Image Sequences/Fluo-N2DL-HeLa/Sequence " + dataset + "/" + cnt + ".tif"
    #img = cv.imread(img_path,0)
    colorimg = img.copy()
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    img[img<=129] = 0
    
    # top-hat
    kernel = np.ones((31,31),np.uint8)
    top_hat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)

    ### top_hat_binary and top_hat_binary_remove_small are used to remove small 
    top_hat_binary = top_hat.copy()
    top_hat_binary[top_hat_binary>0]=255
    top_hat_binary = top_hat_binary > 0

    top_hat_binary_remove_small = morphology.remove_small_objects(top_hat_binary, min_size = 20,connectivity = 1)
    top_hat_binary_remove_small = np.multiply(top_hat_binary_remove_small, 1)
    top_hat_binary_remove_small[top_hat_binary_remove_small==1]=255
    top_hat_binary_remove_small_type = top_hat_binary_remove_small.astype(np.uint8)
    top_hat_binary_remove_small[top_hat_binary_remove_small>0]=1
    top_hat_tmp = np.uint8(np.multiply(top_hat, top_hat_binary_remove_small))
    
    # gaussian filter
    gaussian = cv.GaussianBlur(top_hat_tmp, (0,0), sigmaX = 3)

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    h_max_recon = np.zeros(gaussian.shape, np.uint8)
    HMax(gaussian, h_max_recon, 1, kernel)

    maxi = np.subtract(gaussian, h_max_recon)
    maxi[maxi>0] = 255 
    if num == 90:
        cv.imwrite("maxi.tif", maxi)
    
    ### watershed
    kk = np.ones((3,3),np.uint8)
    # background
    sure_bg = cv.dilate(top_hat_binary_remove_small_type,kk,iterations=2)

    # unknown
    sure_fg = np.uint8(maxi)
    unknown = cv.subtract(sure_bg,sure_fg)

    # label
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
    
    num_labels, labels, stats, centers = cv.connectedComponentsWithStats(cell_seg,connectivity=4)
    labels_tmp = np.zeros(labels.shape,np.uint8)
    labels_tmp[labels != 0] = 1
    cell_seg_intensity = np.multiply(labels_tmp, img)
        
    info[num] = np.empty([6], dtype = np.ndarray)
    info[num][0] = labels # whole labels
    info[num][1] = centers # center
    info[num][2] = stats[:,4] # area
    info[num][3] = np.zeros((num_labels,),np.float32) # intensity
    info[num][4] = stats[:,:4] # bounding box
    #info[num][5] = np.array([0])
    num += 1
    
    # used for task 3
    key_label = list(range(1,num_labels))
    value_dist = [0] * (num_labels-1)
    dict1 = dict(zip(key_label, value_dist))
    total_dist.append(dict1)
        
    init_centers = centers[1:].tolist()
    dict2 = dict(zip(key_label, init_centers))
    net_dist.append(dict2)
        
    # output img
    color = (0,255,0)
    framed_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    framed_img[line_img==1] = [255,255,255]
    
    cv.putText(framed_img, "number of cells: "+ str(num_labels), (30, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    # task 1 frame cells
    for t in range(1,num_labels):
        x,y,w,h,area=stats[t]
        if w > 50:
            continue
        cv.rectangle(framed_img, (x,y), (x+w, y+h),color,1)
        cv.putText(framed_img, str(t), (x-3, y-3), cv.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
    
        t_inten = np.sum(cell_seg_intensity[labels == t]) / np.sum(labels == t)
        info[num-1][3][t] = t_inten
    
    mitosis[num-1] = 0
    tmp_cnt = cnt+".tif"
    pre_img = img.copy()
    rewrite_flag = 0

    ### from the second frame
    if num > 1:
        n_1_frame = np.max(info[num-2][0]) + 1
        n_frame = np.max(info[num-1][0]) + 1

        # make matrix for matching
        dist = np.empty([n_1_frame,n_frame], np.uint32)
        for i in range(n_1_frame):
            for j in range(n_frame):
                a=np.square(int(info[num-2][1][i][0]) - int(info[num-1][1][j][0]))
                b=np.square(int(info[num-2][1][i][1]) - int(info[num-1][1][j][1]))
                c=int(np.sqrt(a+b))
                dist[i][j] = c
                
        # find matching            
        row_ind, col_ind = linear_sum_assignment(dist)
          
        invalid_ind = []
        # for each pair of matching
        for k in range(col_ind.size):
            n_1_choice = row_ind[k]
            n_choice = col_ind[k]
            start_x = int(info[num-2][1][n_1_choice][0])
            start_y = int(info[num-2][1][n_1_choice][1])
            end_x = int(info[num-1][1][n_choice][0])
            end_y = int(info[num-1][1][n_choice][1])
            cell_dist = int(np.square(start_x-end_x) + np.square(start_y-end_y))
            if cell_dist > 1700:
                invalid_ind.append(n_choice)
        
        # for each pair of matching
        for k in range(col_ind.size):
            n_1_choice = row_ind[k]
            n_choice = col_ind[k]
            start_x = int(info[num-2][1][n_1_choice][0])
            start_y = int(info[num-2][1][n_1_choice][1])
            end_x = int(info[num-1][1][n_choice][0])
            end_y = int(info[num-1][1][n_choice][1])
            cell_dist = int(np.square(start_x-end_x) + np.square(start_y-end_y))
            
            # if dist is too large, update mismatching
            if cell_dist > 1700:
                pass
            else:
                cv.line(line_img, (start_x,start_y), (end_x,end_y), 1, 1, 4)
                cv.line(framed_img, (start_x,start_y), (end_x,end_y), (255,255,255), 1, 4)
                ### for task 3
                cell_dist = int(np.sqrt(cell_dist))
                if n_choice != 0 and n_1_choice != 0:
                    #get_current_speed, get_total_dist, get_net_dist, get_ratio
                    # speed
                    get_current_speed[(num-1, n_choice)] = cell_dist
                    #cv.putText(framed_img, "s "+str(cell_dist), (end_x+info[num-1][4][n_choice][2], end_y), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
                    # total dist
                    total_dist[-1][n_choice] = total_dist[-2][n_1_choice] + cell_dist
                    get_total_dist[(num-1, n_choice)] = total_dist[-1][n_choice]
                    #cv.putText(framed_img, "t "+str(total_dist[-1][n_choice]), (end_x+info[num-1][4][n_choice][2], end_y+10), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
                    # net dist
                    net_dist[-1][n_choice] = net_dist[-2][n_1_choice]
                    tmp_sqr_dist1 = int(np.square(int(end_x) - int(net_dist[-1][n_choice][0])))
                    tmp_sqr_dist2 = int(np.square(int(end_y) - int(net_dist[-1][n_choice][1])))
                    current_net_dist = int(np.sqrt(tmp_sqr_dist1+tmp_sqr_dist2))
                    get_net_dist[(num-1, n_choice)] = current_net_dist
                    #cv.putText(framed_img, "n "+str(current_net_dist), (end_x+info[num-1][4][n_choice][2], end_y+20), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
                    # ratio
                    if current_net_dist < 0.01:
                        get_ratio[(num-1, n_choice)] = 0.0
                    else:
                        ratio = round(total_dist[-1][n_choice] / current_net_dist, 2)
                        get_ratio[(num-1, n_choice)] = ratio
                    #cv.putText(framed_img, "r "+str(ratio), (end_x+info[num-1][4][n_choice][2], end_y+30), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
            
            # use intensity to detect mitosis
            if abs(info[num-1][3][n_choice] - info[num-2][3][n_1_choice]) >= 3 or info[num-1][3][n_choice] > 139:
                mitosis[num-1] += 1
                # frame current
                if info[num-1][4][n_choice][2] > 50:
                    continue
                else:
                    cv.rectangle(framed_img, (info[num-1][4][n_choice][0],info[num-1][4][n_choice][1]), \
                                 (info[num-1][4][n_choice][0] + info[num-1][4][n_choice][2], \
                                  info[num-1][4][n_choice][1] + info[num-1][4][n_choice][3]),(0,0,255),1)
                # frame previous
                tmp_cnt = "00" + str(num-2)
                tmp_cnt = "t" + tmp_cnt[-3:] + ".tif"
                pre_img = cv.imread("./out_trace/data2_s"+dataset + "/labeled" + tmp_cnt)

                if n_choice not in invalid_ind:
                    if info[num-2][4][n_1_choice][2]>50:
                        continue
                    else:
                        cv.rectangle(pre_img, (info[num-2][4][n_1_choice][0],info[num-2][4][n_1_choice][1]), \
                                     (info[num-2][4][n_1_choice][0] + info[num-2][4][n_1_choice][2], \
                                      info[num-2][4][n_1_choice][1] + info[num-2][4][n_1_choice][3]),(0,0,255),1)
                        mitosis[num-2] += 1
                        rewrite_flag = 1
                    
                # recover mistake
                s_w_tl = 0 if int(info[num-1][1][n_choice][0]) - 55 < 0 else int(info[num-1][1][n_choice][0]) - 55
                s_w_tr = 1099 if int(info[num-1][1][n_choice][0]) + 55 > 1099 else int(info[num-1][1][n_choice][0]) + 55
                s_w_bl = 0 if int(info[num-1][1][n_choice][1]) - 55 < 0 else int(info[num-1][1][n_choice][1]) - 55
                s_w_br = 699 if int(info[num-1][1][n_choice][1]) + 55 > 699 else int(info[num-1][1][n_choice][1]) + 55
                search_window = labels[s_w_bl:s_w_br, s_w_tl:s_w_tr]

                unbox_flag = 1
                for match_id in np.unique(search_window):
                    if match_id != 0:
                        if match_id in invalid_ind or match_id not in col_ind and match_id != n_choice:
                            unbox_flag = 0
                            if info[num-1][4][match_id][2]>50:
                                continue
                            cv.rectangle(framed_img, (info[num-1][4][match_id][0],info[num-1][4][match_id][1]), \
                                 (info[num-1][4][match_id][0] + info[num-1][4][match_id][2], \
                                  info[num-1][4][match_id][1] + info[num-1][4][match_id][3]),(0,0,255),1) 
                            #restart daughtaur cell
                            get_current_speed[(num-1, n_choice)] = 0
                            get_total_dist[(num-1,n_choice)] = 0
                            get_net_dist[(num-1,n_choice)] = 0
                            net_dist[-1][n_choice] = info[num-1][1][n_choice]
                            get_ratio[(num-1,n_choice)] = 0
                            total_dist[-1][n_choice] = 0
                
                if unbox_flag == 1:
                    mitosis[num-1] -= 1
                    mitosis[num-2] -= 1
    # update previous frame
    if rewrite_flag == 1:
        cv.putText(pre_img, "number of mitosis: "+ str(mitosis[num-2]), (30, 75), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        rewrite_out = "./out_trace/data2_s" + dataset + "/labeled" + tmp_cnt
        print(tmp_cnt)
        cv.imwrite(rewrite_out, pre_img)
    elif num >= 2:
        tmp_cnt = "00" + str(num-2)
        tmp_cnt = "t" + tmp_cnt[-3:] + ".tif"
        print(tmp_cnt)
        rewrite_out = "./out_trace/data2_s" + dataset + "/labeled" + tmp_cnt
        pre_img = cv.imread("./out_trace/data2_s" + dataset + "/labeled" + tmp_cnt)
        cv.putText(pre_img, "number of mitosis: "+ str(mitosis[num-2]), (30, 75), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv.imwrite(rewrite_out, pre_img)
    # output current frame
    if num-1==91:
        cv.putText(framed_img, "number of mitosis: "+ str(mitosis[num-1]), (30, 75), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    
    output_path = "./out_trace/data2_s"+dataset + "/labeled" + cnt + ".tif"
    cv.imwrite(output_path, framed_img)
