import cv2
import numpy as np
import matplotlib.pyplot as plt

import skimage as sk
from skimage.morphology import *
from segment import *
from background_marker import *
from otsu_segmentation import *
from review import files

def ImageColor(image):
    if(image.shape[2]==1):
        isColor=True
    if(image.shape[2] == 3):
        isColor=False
    return isColor
    


def get_single_lcc(image,new):
    '''
    compute largest Connect component of an labeled image
    Parameters:
    ---
    bw_img:
        binary image
    Example:
    ---
        >>> lcc = largestConnectComponent(bw_img)

    '''
#    找背景
    original_image,masker = generate_background_marker(image)
    original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
    original_image = cv2.GaussianBlur(original_image,(5,5),0)
    _, original_image = cv2.threshold(original_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    mask = np.logical_not(masker)
    new[mask]=0  

    count_0=0
    count_255=0
    for x in range(new.shape[0]):
        for y in range(new.shape[1]):
            if new[x,y]<=140:
                new[x,y]=0
                count_0 += 1
            else:
                new[x,y]=255
                count_255+=1

    if count_0==new.shape[0]*new.shape[1]:
        new[new.shape[0]//2,new.shape[1]//2]=255
    if count_255==new.shape[0]*new.shape[1]:
        new[new.shape[0]//2,new.shape[1]//2]=0

    largest_mask = \
            select_largest_obj(new, fill_mode='FLOOD',
                               smooth_boundary=False)
            
    segmented_image = apply_marker(original_image, largest_mask, background=0)       
    labeled_img, num = label(segmented_image, neighbors=4, background=0, return_num=True)    

    max_label = 0
    max_num = 0
    for i in range(1, num): # 这里从1开始，防止将背景设置为最大连通域
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    single_lcc = (labeled_img == max_label)

    return single_lcc

def new_mat(a,b):
    return a+b
    
def get_combine_marker(original_image):
    
    LGB = cv2.cvtColor(original_image,cv2.COLOR_RGB2LAB)#cv2.COLOR_BGR2LAB

    
#    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    L,a,b = cv2.split(LGB) 
    
    new_ab=new_mat(a,b)
    new_b=new_mat(b,np.zeros(b.shape, dtype=np.uint8))
    

    
    lcc_ab=get_single_lcc(original_image,new_ab)#background is False
    lcc_b=get_single_lcc(original_image,new_b)
    lcc =  np.full((lcc_ab.shape[0], lcc_ab.shape[1]), False)

    for x in range(lcc_ab.shape[0]):
        for y in range (lcc_ab.shape[1]):
            if lcc_ab[x,y]==False and lcc_b[x,y]==False:
                lcc[x,y]=False
            else:
                lcc[x,y]=True
                
    unique, counts = np.unique(lcc, return_counts=True)
    unique_counts = dict(zip(unique, counts))
    if (False not in unique_counts ):
        lcc = lcc_ab
        
    segmented_image = apply_marker(original_image, lcc, background=0 , inverse=True) 
#    +滤波  去阴影不够 后面用
    #noise removal using grayscale morphology 
    segmented_image = cv2.cvtColor(segmented_image,cv2.COLOR_BGR2GRAY)

 #    可画出轮廓
#    segmented_image = cv2.adaptiveThreshold(segmented_image,255,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY,9,2 )
#    cv2.THRESH_OTSU
#    _, segmented_image = cv2.threshold(segmented_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)
    
    segment_of_orignal =  np.full((original_image.shape[0], original_image.shape[1],original_image.shape[2]), fill_value=0, dtype=np.uint8)  

    for k in range(segment_of_orignal.shape[2]):
            for x in range(segment_of_orignal.shape[0]):
                for y in range (segment_of_orignal.shape[1]):
                    if segmented_image[x,y] == 0 :
                        segment_of_orignal[x,y,k]=0 
                    else:
                        segment_of_orignal[x,y,k]=original_image[x,y,k]
        
    return segment_of_orignal,segmented_image,lcc
#输入的为前景inverse=False ,背景inverse=True
def sgement_by_markers(original_image,lcc,inverse):
    segmented_image = apply_marker(original_image, lcc, background=0,inverse = inverse) 
#    +滤波  去阴影不够 后面用
    #noise removal using grayscale morphology 
    segmented_image = cv2.cvtColor(segmented_image,cv2.COLOR_BGR2GRAY)

 #    可画出轮廓
#    segmented_image = cv2.adaptiveThreshold(segmented_image,255,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY,9,2 )
#    cv2.THRESH_OTSU
#    _, segmented_image = cv2.threshold(segmented_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    segment_of_orignal =  np.full((original_image.shape[0], original_image.shape[1],original_image.shape[2]), fill_value=0, dtype=np.uint8)  

    for k in range(segment_of_orignal.shape[2]):
            for x in range(segment_of_orignal.shape[0]):
                for y in range (segment_of_orignal.shape[1]):
                    if segmented_image[x,y] == 0 :
                        segment_of_orignal[x,y,k]=0
                    else:
                        segment_of_orignal[x,y,k]=original_image[x,y,k]
    return segment_of_orignal,segmented_image
    
#def BGR2GRAY_markers(original_image):
#     gray=np.zeros([original_image.shape[0],original_image.shape[1]])
#     B,G,R=cv2.split(original_image)
#     area_notZero = original_image[original_image!=[0,0,0]].size
#    _gray=
#     _row=np.where()
    
if __name__ == '__main__':
    
    segment_of_orignal,segmented_image,lcc = get_combine_marker(files['jpg1'])
    cv.imshow('segment_of_orignal',segment_of_orignal)
    segment_of_orignal = cv2.cvtColor(segment_of_orignal,cv2.COLOR_BGR2GRAY)
    _, image_marker = get_marker(segment_of_orignal)
    
    cv.imshow('segmented_image',segmented_image)
    cv.imshow('image_marker',image_marker)
    cv.waitKey() 
    '''
    for i in range(1,len(files)):
        original_image = read_image(files['jpg'+str(i)])
        LGB = cv2.cvtColor(original_image,cv2.COLOR_RGB2LAB)#cv2.COLOR_BGR2LAB
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        L,a,b = cv2.split(LGB) 
        new_ab=new_mat(a,b)
        new_b=new_mat(b,np.zeros(b.shape, dtype=np.uint8))
        
        _,lcc_ab=get_single_lcc(files['jpg'+str(i)],new_ab)#background is False
        _,lcc_b=get_single_lcc(files['jpg'+str(i)],new_b)
        lcc =  np.full((lcc_ab.shape[0], lcc_ab.shape[1]), False)
    
        for x in range(lcc_ab.shape[0]):
            for y in range (lcc_ab.shape[1]):
                if lcc_ab[x,y]==False and lcc_b[x,y]==False:
                    lcc[x,y]=False
                else:
                    lcc[x,y]=True
                    
        unique, counts = np.unique(lcc, return_counts=True)
        unique_counts = dict(zip(unique, counts))
        if (False not in unique_counts ):
            lcc = lcc_ab
            
     
    #    cv2.imshow("0",original_image)
    #    cv2.waitKey()
    #    segmented_image(0,255)  lcc(TF)
        segmented_image = apply_marker(original_image, lcc, background=0) 
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
#        _,binary = cv2.threshold(segmented_image,0.1,1,cv2.THRESH_BINARY)
#        image,contours,hierarch=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#        for i in range(len(contours)):
#            area = cv2.contourArea(contours[i])
#            if area < threshold:
#                cv2.drawContours(image,[contours[i]],0,0,-1)

#        image ,contours,hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
##        绘制独立轮廓，如第四个轮廓
#        imag = cv2.drawContours(image,contours,-1,(0,0,255),3)
        
        segment_of_orignal =  np.full((original_image.shape[0], original_image.shape[1],original_image.shape[2]), fill_value=0, dtype=np.uint8)  
        for k in range(segment_of_orignal.shape[2]):
            for x in range(segment_of_orignal.shape[0]):
                for y in range (segment_of_orignal.shape[1]):
                    if closed[x,y] == np.uint8(0) :
                        segment_of_orignal[x,y,k]=0
                    else:
                        segment_of_orignal[x,y,k]=original_image[x,y,k]
        cv2.imshow("segmented_image",segmented_image)
        cv2.waitKey()       
        cv2.imshow("closed",closed)
        cv2.waitKey()
        cv2.imshow("segment_of_orignal",segment_of_orignal)
        cv2.waitKey() 
'''
'''计算颜色直方图         
    img = read_image(files['jpg6'])
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        _, segmented_image = cv2.threshold(histr,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        plt.plot(histr, color=col)
    plt.xlim([0, 256])
    plt.show()
 '''   
        
        
        
        
        
        
        
        
        
#    #    +滤波  去阴影不够 后面用
#        #noise removal using grayscale morphology
#        kernel = np.ones((2,2),np.uint8)
#        closing = cv2.morphologyEx(segmented_image,cv2.MORPH_CLOSE,kernel,iterations = 5)
#        
#        gradient = cv2.morphologyEx(gray,cv2.MORPH_GRADIENT,kernel)
##        cv2.imshow("gradient"+'_'+str(i),gradient)
#         
#        segmented_image = cv2.cvtColor(segmented_image,cv2.COLOR_BGR2GRAY)
#        _, segmented_image = cv2.threshold(segmented_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
#        _, markers = cv2.connectedComponents(segmented_image)
#        markers = markers+1
#        
#        markers = cv2.watershed(original_image,markers)
#        markers_image = apply_marker(original_image, markers)
##        cv2.imshow("markers_image"+'_'+str(i),markers_image)
##        cv2.waitKey()
      











  
#        _,contours_g,hierarchy = cv2.findContours(gradient,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        
#        imag_s = cv2.drawContours(segmented_image,contours_s,-1,(0,255,0),3)#-1为填充模式
#        imag_g = cv2.drawContours(segmented_image,contours_g,-1,(0,255,0),3)
#        cv2.imshow("markers"+'_'+str(i),markers)
#        cv2.imshow("original_image"+'_'+str(i),original_image)
#        cv2.imshow("segmented_image"+'_'+str(i),segmented_image)
#        cv2.imshow("img_s"+'_'+str(i),img_s)
#        cv2.imshow("img_g"+'_'+str(i),img_g)
#        cv2.waitKey()
       
    #    优化
#        masker = texture_filter(segmented_image, lcc, threshold=200, window=3) 
#        opt_image = apply_marker(original_image, masker, background=0)
#        cv2.imshow("opt_image"+'_'+str(i),opt_image)
#        cv2.waitKey()
         
    
    #    image, contours, hierarchy = cv2.findContours(segmented_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)             
    #    plt.figure('original image with contours')
    #    plt.imshow(img , cmap = 'gray')
    
    
 