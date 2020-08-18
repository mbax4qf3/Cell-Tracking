import cv2
import numpy as np  
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy.optimize import linear_sum_assignment
from collections import Counter
import copy

### each frame has an info entry，info contains cells' info
### info[0] labels
### info[1] centers
### info[2] areas
### info[3] bounding box
info = np.empty([500], dtype = np.ndarray)
### frame counting
num = 0
### line_img - to store all the past trace
img_path_tmp = "./COMP9517 20T2 Group Project Image Sequences/DIC-C2DH-Hela/Sequence 1/t000.tif"
img_tmp = cv2.imread(img_path_tmp,0)
line_img = np.zeros(img_tmp.shape, np.uint8)

### data for task 3
total_dist = []
net_dist = []

get_current_speed = {}
get_total_dist = {}
get_net_dist = {}
get_ratio = {}
mitosis_cnt = {}

read_path = "./COMP9517 20T2 Group Project Image Sequences/DIC-C2DH-HeLa/Sequence 1/"

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

def label_cells(img,tif):
    global info, num, line_img, total_dist, net_dist, get_current_speed, get_total_dist, get_net_dist, get_ratio, mitosis

    #if num == 11:
        #hist = cv2.calcHist([img],[0],None,[256],[0,256])
        #plt.plot(hist)
        #plt.show()
        #exit
    path = 'data1/'

    #info = np.empty([500], dtype = np.ndarray)
    color = (0,255,0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enh = clahe.apply(img)
    cv2.imwrite('data1_enhance.jpg', img_enh)

    
    ##img_enh[img_enh<=100] = 255
    ##img_enh[img_enh<150] = 0
    ##img_enh[img_enh>=150] = 255
    ##cv2.imwrite('data1_bin.jpg', img_enh)

    img_enh = cv2.GaussianBlur(img_enh,(55,55),0)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #img_enh = clahe.apply(img)
    #cv2.imwrite('data1_enhance2.jpg', img_enh)

    ##kernel = np.ones((31,31),np.uint8)
    ##top_hat = cv2.morphologyEx(img_enh, cv2.MORPH_TOPHAT, kernel)
    ##
    ##cv2.imwrite('data1_enhance3.jpg', top_hat)

    ##img_enh = cv2.GaussianBlur(img_enh,(5,5),0)
    ##kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    ##dst = cv2.filter2D(img, -1, kernel=kernel)

    Blur = cv2.GaussianBlur(img_enh, (9,9), 0)
    _,thresh = cv2.threshold(Blur,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
    closing = thresh

    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    #cv2.imwrite('data1_opening.jpg',opening)

    kernel=np.ones((9,9),np.uint8)
    erosion=cv2.erode(opening,kernel,iterations=3) 
    dilation=cv2.dilate(erosion,kernel,iterations=1) 

    #cv2.imwrite('data1_dilation.jpg',dilation)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilation, 4, cv2.CV_8U)

    labels = np.array(labels, dtype=np.float)
    #print(labels.shape)
    toplabels = labels[...,0]
    bottomlabels = labels[...,511]
    leftlabels = labels[0,...]
    rightlabels = labels[511,...]

    sqr = [toplabels[toplabels!=0],bottomlabels[bottomlabels!=0],
           leftlabels[leftlabels!=0],rightlabels[rightlabels!=0]]
    background = np.zeros_like(labels)


    maxnum = Counter(labels.flatten()).most_common(3)
    maxnum = sorted([x[0] for x in maxnum])
    #background = np.zeros_like(labels)
    if len(maxnum) == 1:
        pass
    elif len(maxnum) == 2:
        background[labels == maxnum[1]] = 1
    else:
        background[labels == maxnum[1]] = 1
        background[labels == maxnum[2]] = 1
        
    dilation[background != 0]=0

    cv2.imwrite('data1_dilation.jpg',dilation)

    for l in sqr:
        for i in l[l!=0]:
            dilation[labels==i]=0

    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(dilation,connectivity = 4)
    ##image, contours, hierarchy = cv2.findContours(dilation, 2, 1)
    ##
    ##hull_img = copy.copy(img)
    ##for cnt in contours:
    ##    #print(cv2.contourArea(cnt))
    ##    if cv2.contourArea(cnt) > 3000 or cv2.contourArea(cnt)<80:
    ##        rm_label = labels[cnt[0][0][1],cnt[0][0][0]]
    ##        dilation[labels==rm_label]=0
    ##        
    ##num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(dilation,connectivity = 4)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img[line_img==1]=[255,0,0]
    info[num] = np.empty([4], dtype = np.ndarray)
    info[num][0] = labels
    info[num][1] = centers
    info[num][2] = stats[:,4] # area
    info[num][3] = stats[:,:4] # bounding box
    num += 1

    for t in range(1,num_labels):
        x,y,w,h,area=stats[t]
        cv2.rectangle(img, (x,y), (x+w, y+h),color,1)
        cv2.putText(img, str(t), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        #if abs((w/h)-1) < 0.02:
            #cv2.rectangle(img, (x,y), (x+w, y+h),(0,0,255),1)
    cv2.putText(img, "number of cells: "+ str(num_labels), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
    #cv2.imwrite("data1/labeled"+tif+".tif",img)
    
    if num > 1:
        #print(2)
        #print(info[num-2][0])
        #print(labels)
        bright_pixel = np.zeros(num_labels,np.uint8)
        #print(stats)
        for ii in range(1,num_labels):
            #print(ii)
            bright_pixel[ii] = np.max(img[stats[ii][0]:stats[ii][0]+stats[ii][2],stats[ii][1]:stats[ii][1]+stats[ii][3]])

        for jj in range(1,num_labels):
            brightest = np.where(bright_pixel == np.max(bright_pixel))
        #print(brightest)
        #cv2.rectangle(img, (stats[brightest[0]][0],stats[brightest[0]][1]), (stats[brightest[0]][0]+stats[brightest[0]][2], \
                                                                             #stats[brightest[0]][1]+stats[brightest[0]][3]),(0,0,255),1)
        
        n_1_frame = np.max(info[num-2][0]) + 1
        n_frame = np.max(info[num-1][0]) + 1
        matrix = np.zeros([n_1_frame,n_frame], np.float32)
        dist_ = np.zeros([n_1_frame,n_frame], np.float32)
        area = np.zeros([n_1_frame,n_frame], np.float32)

        for i in range(n_1_frame):
            for j in range(n_frame):
                if i != 0 and j != 0:
                    a=np.square(info[num-2][1][i][0] - info[num-1][1][j][0])
                    b=np.square(info[num-2][1][i][1] - info[num-1][1][j][1])
                    c=np.sqrt(a+b)
                    dist_[i][j] = c
        matrix = dist_

        row_ind, col_ind = linear_sum_assignment(matrix)
        find_miss1 = list(range(num_labels))
        find_miss = list(set(find_miss1)-set(col_ind))
        #print(1)
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

            if dist_[n_1_choice][n_choice] < 70:
                cv2.line(line_img, (start_x,start_y), (end_x,end_y), 1, 1, 4)
                cv2.line(img, (start_x,start_y), (end_x,end_y), (255,0,0), 1, 4)

    cv2.imwrite("data1/labeled"+tif+".tif",img)

            
for i in range(84):
    if i < 10:
        tif = "t00" + str(i)
    elif i < 100:
        tif = "t0" + str(i)
    else:
        tif = "t" + str(i)
    img = cv2.imread("./COMP9517 20T2 Group Project Image Sequences/DIC-C2DH-HeLa/Sequence 1/"+tif+".tif",0)
    label_cells(img,tif)    
