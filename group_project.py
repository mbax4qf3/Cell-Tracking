import cv2
import numpy as np
import signal
from os import walk
from sys import argv
import data2_implement
import data3_implement



def n(img):
    new_img = img.copy().astype(np.float32)
    new_img -= np.mean(new_img)
    new_img /= np.linalg.norm(new_img)
    new_img = np.clip(new_img,0,255)
    new_img *= (1./float(new_img.max()))

    return (img*255).astype(np.uint8)
    

def read_sequences(dataset_path):
    sequence_img = dict()
    sequence_n_img = dict()
    mask= dict()
    for root, dirs, files in walk(dataset_path):
        if files:
            files.sort()
        #从文件夹下读图片
        dirs.sort()
        if dirs == []:
            cur_seq = root.split('/')[-1]
            if 'Masks' in cur_seq:
                mask[cur_seq]=dict()
                for file in files:
                    img = cv2.imread(root+'/'+file)
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    mask[cur_seq][file]=np.array(gray)
                    a = np.array(gray)
                    
                    if a[a!=0] != []:
                        a *= 255
                        cv2.imwrite(f"{file}_255.jpg",a)
                continue
            sequence_img[cur_seq] = []
            sequence_n_img[cur_seq] = []
            for file in files:
                img = cv2.imread(root+'/'+file)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                n_gray = n(gray)
##                cv2.imshow("n_gray",n_gray)
##                cv2.waitKey()
##                cv2.destroyWindow()
                sequence_img[cur_seq].append(img)
                #灰度图数据集
                sequence_n_img[cur_seq].append(n_gray)
                #正则化的图片集

    return sequence_img,sequence_n_img,mask

#datasets = {'PhC-C2DL-PSC/':data3_implement}
datasets = {'Fluo-N2DL-HeLa/':data2_implement}#,'PhC-C2DL-PSC/':data3_implement}
path = './COMP9517 20T2 Group Project Image Sequences/'


for dataset in datasets:
    print("start read: ",dataset)
    if dataset == 'PhC-C2DL-PSC/':
        st = '3'
    else:
        st = '2'
    img_path_tmp = "./COMP9517 20T2 Group Project Image Sequences/"+dataset+"Sequence 1/t000.tif"
    img_tmp = cv2.imread(img_path_tmp,0)
    line_img = np.zeros(img_tmp.shape, np.uint8)
    implement = datasets[dataset]
    implement.init()
    dataset = path + dataset
    sequences,sequences_n,mask = read_sequences(dataset)
##    sequence = {
##        'sequence 1':[[001],[002],[003]]
##     ...
##        }
    seq = 1

    for key in sequences:
        print(key)
        implement.init()
        if st == '2':
            lasttif = 92
        else:
            lasttif = 201
        for i in range(lasttif):
            if i < 10:
                tif = "t00" + str(i)
            elif i < 100:
                tif = "t0" + str(i)
            else:
                tif = "t" + str(i)
            implement.label_cells(sequences[key][i],tif,str(seq))
        wirte_path = "./out_trace/data"+st+"_s"+str(seq)+"/"
        np.savez(wirte_path+'labels.txt',np.array(implement.info))
        file_handle=open(wirte_path+'get_current_speed.txt',mode='w')
        file_handle.write(str(implement.get_current_speed))
        file_handle.close()
        file_handle=open(wirte_path+'get_total_dist.txt',mode='w')
        file_handle.write(str(implement.get_total_dist))
        file_handle.close()
        file_handle=open(wirte_path+'get_net_dist.txt',mode='w')
        file_handle.write(str(implement.get_net_dist))
        file_handle.close()
        file_handle=open(wirte_path+'get_ratio.txt',mode='w')
        file_handle.write(str(implement.get_ratio))
        file_handle.close()
        seq += 1
        print("next_sequence")

    print("finish ", dataset)
##    for seq in range(1,5):
##        path = '/Users/pololuo/Desktop/comp9517/project/out_trace/data2_s'+str(seq)+'/labeled'
##        for i in range(92):
##            if i < 10:
##                tif = "t00" + str(i)
##            elif i < 100:
##                tif = "t0" + str(i)
##            else:
##                tif = "t" + str(i)
##            read_img = cv2.imread(path+tif+'.tif')
##            window = str(seq)+' '+tif
##            cv2.namedWindow(window)
##            cv2.setMouseCallback(window, on_EVENT_LBUTTONDOWN)
##            cv2.imshow(window, read_img)
##            cv2.waitKey(0)
##            cv2.destroyAllWindows()



