import cv2
import numpy as np
import copy
from sys import argv

if argv[1] == '2':
    dataset = 'data2_s'
    length=92
elif argv[1] == '3':
    dataset = 'data3_s'
    length=201


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        label = inf[i][0][y,x]
        info=""

        speed=str(get_current_speed[(i,label)]) if get_current_speed.get((i,label)) else '0'
        info+="speed: "+speed
        total_dist=str(get_total_dist[(i,label)]) if get_total_dist.get((i,label)) else '0'
        info += "; total dis: "+total_dist
        net_dist=str(get_net_dist[(i,label)]) if get_net_dist.get((i,label)) else '0'
        info += "; net dis: "+net_dist
        ratio=str(get_ratio[(i,label)]) if get_ratio.get((i,label)) else '0'
        info += "; ratio: "+ ratio
        print(str(label)+"-----"+info)
        cv2.circle(read_img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(read_img,info, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 255, 255), thickness=1)
        cv2.imshow(window, read_img)

for seq in range(1,5):
    wirte_path = "./out_trace/"+dataset+str(seq)+"/"
    data=np.load(wirte_path+'labels.txt.npz')
    inf = data['arr_0']
    file=open(wirte_path+'get_current_speed.txt',mode='r')
    get_current_speed=copy.copy(eval(file.read()))
    file.close()
    file=open(wirte_path+'get_total_dist.txt',mode='r')
    get_total_dist=copy.copy(eval(file.read()))
    file.close()
    file=open(wirte_path+'get_net_dist.txt',mode='r')
    get_net_dist=copy.copy(eval(file.read()))
    file.close()
    file=open(wirte_path+'get_ratio.txt',mode='r')
    get_ratio=copy.copy(eval(file.read()))
    file.close()
    path = './out_trace/'+dataset+str(seq)+'/labeled'
    startframe=input("please input time frame: ")
    startframe=int(startframe) if startframe else 0

        
    for i in range(startframe,length):
        if i < 10:
            tif = "t00" + str(i)
        elif i < 100:
            tif = "t0" + str(i)
        else:
            tif = "t" + str(i)
        read_img = cv2.imread(path+tif+'.tif')
        window = str(seq)+' '+tif
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_EVENT_LBUTTONDOWN)
        cv2.imshow(window, read_img)
        cv2.moveWindow(window,0,50)
        flag = cv2.waitKey(0)
        
        if flag ==13:
            label = int(input("input label: "))
            print("speed: ",str(get_current_speed[(i,label)]))
            print("total distance: ", str(get_total_dist[(i,label)]))
            print("net distance: ", str(get_net_dist[(i,label)]))
            print("ratio: ", str(get_ratio[(i,label)]))
            cv2.waitKey(0)
        elif flag == 100:
            cv2.destroyAllWindows()
            break
        cv2.destroyAllWindows()
    
        


##while(1):
##    roi=cv2.cv2.selectROI("img",img)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
##    print(roi)

