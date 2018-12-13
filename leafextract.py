
import cv2 as cv
import numpy as np
from common import Sketcher
from max_arear import *
from collections import Counter
import random
#from matplotlib import pyplot as plt
def Grayscale_morphology(image):
    
    segment_of_orignal,marker_image,lcc = get_combine_marker(image)
    '''cv.imshow('segment_of_orignal',segment_of_orignal)'''
    
    segment_of_orignal_gray = cv2.cvtColor(segment_of_orignal,cv2.COLOR_BGR2GRAY)
    
    #阈值image_marker 更细化
    _, image_marker = get_marker(segment_of_orignal_gray)
    image_marker=cv2.bitwise_not(image_marker)
#    cv.imshow('image_marker',image_marker)  
       
    #noise removal using grayscale morphology
#    kernel = np.ones((3,3),np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
#    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    #closing = cv.morphologyEx(segmented_image,cv.MORPH_CLOSE,kernel,iterations = 5)
    #opening = cv.morphologyEx(closing,cv.MORPH_OPEN,kernel,iterations = 3)
    
    #sure background area
    #膨胀 可以分开两个物体（两个前景）
    
    image_marker = cv.morphologyEx(image_marker, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(image_marker, kernel, iterations=3)
#    cv.imshow('sure_bg',sure_bg)
    # distance transform
    dist = cv.distanceTransform(image_marker, cv.DIST_L2, 5)
#    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
#    cv.imshow("dist_output", dist_output*70)
    ret, surface = cv.threshold(dist, dist.max()*0.1, 255, cv.THRESH_BINARY)
    
    #Finding unknown region  sure_bg=0
    sure_fg = np.uint8(surface)
    unknown = cv.subtract(sure_bg,sure_fg)
    #Marker labelling 0,1
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure bacground is not 0, but 1
    markers = markers+1
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv.watershed(original_image,markers)
    

    original_image_src=original_image.copy()
    original_image_src[markers == -1] = [0,0,255]
    return original_image,markers,original_image_src,ret,sure_fg,lcc
    # 分水岭算法
def water_image(src):

    blurred = cv.pyrMeanShiftFiltering(src, 10, 100)    # 去除噪点
 
    # gray\binary image
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # morphology operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(mb, kernel, iterations=3)
    
    # distance transform
    dist = cv.distanceTransform(mb, cv.DIST_L2, 5)
#    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
#    cv.imshow("dist_output", dist_output*70)
 
    ret, surface = cv.threshold(dist, dist.max()*0.5, 255, cv.THRESH_BINARY)

    surface_fg = np.uint8(surface)
    unknown = cv.subtract(sure_bg, surface_fg)
    ret, markers = cv.connectedComponents(surface_fg)


    # watershed transfrom
    markers += 1
    markers[unknown == 255] = 0
    markers = cv.watershed(src, markers=markers)
    src[markers == -1] = [0, 0, 255]
#    cv.imshow("watershed_rusult", src)

    return markers,src,surface_fg,ret
#  需修整  
def watershed_gradient(original_image):
#    if (ImageColor(original_image)):
    original_image_of_gray = original_image
#    else:
#        original_image_of_gray=original_image
    kernel = np.ones((3,3),np.uint8)
    gradient = cv2.morphologyEx(original_image_of_gray,cv2.MORPH_GRADIENT,kernel)
    #laplacian = cv2.Laplacian(segment_of_orignal,cv2.CV_64F)
    
    #原图灰度图的局部最大    
    row_kernel,_=kernel.shape
    window =row_kernel - row_kernel//2 - 1
    local_max_original = 0;local_min_gradient = 255;local_min_markers=255
    row_loc = 0;col_loc = 0

    for i in range (window,original_image.shape[0]-window):
        for j in range(window,original_image.shape[1]-window):
    #        躲过边界
            if (i >= window+1 and j>=window+1)or(
                    (i<=original_image.shape[0]-window-1 )and(j<=original_image.shape[1]-window-1 )
                    ):
                for p in range(i-window,i+window+1): 
                    for q in range(j-window,j+window+1):
        #                local_sum_gradient=local_sum_gradient+gradient[p,q]
                        if gradient[p,q]<local_min_gradient:
                            local_min_gradient=gradient[p,q]
                        if markers[p,q]<local_min_markers:
                            local_min_markers=markers[p,q]
                        if original_image_of_gray[p,q]>local_max_original:
                            local_max_original = original_image_of_gray[p,q]
                            row_loc = p;col_loc = q
                
    #            gradient[row_loc,col_loc]=local_min_gradient
                gradient[row_loc,col_loc]=local_max_original
                markers[row_loc,col_loc]=local_min_markers
                local_max_original=0
                local_min_gradient=255
    #                else:
    #                    local_mean_gradient[]=local_sum_gradient//(kernel*kernel)
    #                    gradient[i-window,j+window+1]
    
#    cv.imshow('gradient_1',gradient)
    gradient[markers == -1] = 255
#    cv.imshow('gradient',gradient)
#    cv.waitKey()
    return gradient,markers
    
def part_isin_sufg(original_image):
    _,markers,src,ret,sure_fg,_ = Grayscale_morphology(original_image)
    isin=np.full(ret,False)  
    for count in range(1,ret+1):
        index_markers = np.where(markers == count)
    #        判断每个 index_markers 是不是都在 index_surface_fg 里面
        for i in range(len(index_markers[0])):
            if sure_fg[index_markers[0][i],index_markers[1][i]] == 255:
    #                isin[count]=False
                break          
        
            else:
                if i<(len(index_markers[0])-1): #这个区域没有循环完
                    continue
                else:
                    isin[count-1]=True 
        else:
            continue
        break
    return isin
def area_measure(original_image):
    _,markers,original_image_src,ret,sure_fg,lcc = Grayscale_morphology(original_image)
    markers,ret=review_markers(markers,ret) 
    isin = part_isin_sufg(original_image)
    part = np.where(isin == True)[0][np.argsort(-np.where(isin == True)[0])][0]+1
    area_part = markers[markers==part].size
    original_image_HLS = cv.cvtColor(original_image,cv2.COLOR_BGR2HLS)
    original_image_of_gray = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)

    #    HSL代表色调(Hue)，饱和度(Saturation)和亮度(Lightness)
    H,L,S=cv.split(original_image_HLS)
    #最大区域的 HL均值计算    
#    light_max=0.0; hue_max=0.0; pix_max=0.0
    max_row = np.where(markers==part)[0]; max_col=np.where(markers==part)[1]
    most_light=[0]*area_part;most_hue=[0]*area_part;most_pix=[0]*area_part
    for i in range(area_part):
        most_hue[i] = H[max_row[i],max_col[i]]
        most_pix[i] = original_image_of_gray[max_row[i],max_col[i]]
        most_light[i] = L[max_row[i],max_col[i]]
#        pix_max += original_image_of_gray[max_row[i],max_col[i]]
#        light_max += L[max_row[i],max_col[i]]   
#        hue_max += H[max_row[i],max_col[i]]
        
    most_light_total=most_point(most_light)
    most_hue_total=most_point(most_hue)
    most_pix_total=most_point(most_pix)
#    arg_light_max=light_max/area_part; arg_hug_max=hue_max/area_part ; arg_pix_max=pix_max/area_part
    
    print("max_area finish!")
    
    #其他班区域的 HL均值计算 way1 
#    light_other=np.zeros(ret); hue_other=np.zeros(ret); pix_other = np.zeros(ret)
#    arg_light_other = np.zeros(ret); arg_hug_other = np.zeros(ret)
#    arg_pix_other = np.zeros(ret)
#    change_simple = np.full(ret,False);
    area_part_other = np.zeros(ret);
    change_markers = np.full(ret,False)
    change_most = np.full(ret,False)
    
    #    other_row = np.zeros(ret); other_col = np.zeros(ret)
    for count in range(ret):
        
        area_part_other[count] = len(np.where(markers==count+1)[0])
        other_row = np.where(markers==count+1)[0];other_col =np.where(markers==count+1)[1]
        
#        simple_light_other= np.zeros(area_part_other[count].astype(int))
#        simple_hue_other= np.zeros(area_part_other[count].astype(int))
#        simple_pix_other= np.zeros(area_part_other[count].astype(int))
        
        most_light_other=[0]*int(area_part_other[count])
        most_hue_other=[0]*int(area_part_other[count])
        most_pix_other=[0]*int(area_part_other[count])
        for i in range(area_part_other[count].astype(int)):
            most_light_other[i]=L[other_row[i].astype(int),other_col[i].astype(int)]
            most_hue_other[i]=H[other_row[i].astype(int),other_col[i].astype(int)]
            most_pix_other[i]=original_image_of_gray[other_row[i].astype(int),other_col[i].astype(int)]
            
#            simple_light_other[i]= L[other_row[i].astype(int),other_col[i].astype(int)]-arg_light_max
#            simple_hue_other[i]= H[other_row[i].astype(int),other_col[i].astype(int)]-arg_hug_max
#            simple_pix_other[i]= original_image_of_gray[other_row[i].astype(int),other_col[i].astype(int)]-arg_pix_max
#            
#            light_other[count] += L[other_row[i].astype(int),other_col[i].astype(int)]
#            hue_other[count] += H[other_row[i].astype(int),other_col[i].astype(int)]
#            pix_other[count] += original_image_of_gray[other_row[i].astype(int),other_col[i].astype(int)]
        
#        arg_light_other[count]=light_other[count]/area_part_other[count]
#        arg_hug_other[count]=hue_other[count]/area_part_other[count]
#        arg_pix_other[count]=(pix_other[count]+pix_max)/(area_part_other[count]+area_part)
        

        most_light_total_other=most_point(most_light_other)
        most_hue_total_other=most_point(most_hue_other)
        most_pix_total_other=most_point(most_pix_other)
        
#        change_markers[count]=(
#                                (arg_pix_max-10<= arg_pix_other[count]<= arg_pix_max-10)and
#                                (arg_hug_max-10 <= arg_hug_other[count] <= arg_hug_max+10)and
#                                (arg_light_max-10 <= arg_light_other[count] <= arg_light_max+10)
#                        )
        #有可能亮度均匀   前景选择好效果更佳   arg_light_max  应该改为最多加次多...
#        change_simple[count]=(
#                       (abs(simple_light_other[-10<=simple_light_other].size-
#                        simple_light_other[simple_light_other<=10].size)/area_part_other[count].astype(int)>=0.5)and
#                       (abs(simple_hue_other[-10<=simple_hue_other].size-
#                       simple_hue_other[simple_hue_other<=10].size)/area_part_other[count].astype(int)>=0.5)and
#                       (abs(simple_pix_other[-10<=simple_pix_other].size-
#                       simple_pix_other[simple_pix_other<=10].size)/area_part_other[count].astype(int)>=0.5)
#                      )
        change_most[count] =(                       
                        abs(most_light_total_other-most_light_total)<=10 or
                        abs(most_hue_total_other-most_hue_total)<=10 or
                        abs(most_pix_total_other-most_pix_total)<=10
                      )               
#        if change_markers[count] and count != part-1:
#            markers[markers==count+1] = part
#       
#        if change_simple[count] and count != part-1:
#            markers[markers==count+1] = part

        if change_most[count] and count != part-1:
            markers[markers==count+1] = part

        change_markers[count] = change_most[count]
        
    print("other_area finish!")
       
#    way2
    
    
#    消去联合后的-1,去边界（寻找-1值得附近）
    area_edge=len(np.where(markers==-1)[0])
    edge_row=np.where(markers==-1)[0];edge_col=np.where(markers==-1)[1];ischange = False
    for i in range(0,ret):
        if i != part-1:
            ischange = ischange or change_markers[i]
    for i in range(area_edge):
        if ( ischange and
            (edge_row[i] not in (0,markers.shape[0]-1))and (edge_col[i] not in (0,markers.shape[1]-1))and
            (markers[edge_row[i]+1,edge_col[i]] in (-1,part))and
            (markers[edge_row[i]-1,edge_col[i]] in (-1,part))and
            (markers[edge_row[i],edge_col[i]+1] in (-1,part))and
            (markers[edge_row[i],edge_col[i]-1] in (-1,part))
            ):
                markers[markers == -1] = part
       
    print("change endge finish!")
               
    
    return part,markers
def num_to_bool(markers_num,part):
    
    markers_bool=np.full([markers_num.shape[0],markers_num.shape[1]],True)
    part_row=np.where(markers_num == part)[0]; part_col=np.where(markers_num == part)[1]  
    for i in range(len(part_row)):
        markers_bool[part_row[i],part_col[i]]=False

    return markers_bool
def edge(img):
    #高斯模糊,降低噪声
    blurred = cv.GaussianBlur(img,(3,3),0)
    #灰度图像
    if len(img.shape)==3:
        gray=cv.cvtColor(blurred,cv.COLOR_RGB2GRAY)
    else:
        gray = img
    #图像梯度
    xgrad=cv.Sobel(gray,cv.CV_16SC1,1,0)
    ygrad=cv.Sobel(gray,cv.CV_16SC1,0,1)
    #计算边缘
    #50和150参数必须符合1：3或者1：2
    edge_output=cv.Canny(xgrad,ygrad,50,150)
    return edge_output

def most_point(most_light):
    most_sort=sorted((Counter(most_light)).items(), key=lambda d:d[1],reverse=True)
    max_point=most_sort[0][0]
    a=0;num_copy=0
    while(num_copy/len(most_light)<0.7):
        num=0;most_total=0;
        for i in range(len(most_sort)):
            if max_point-(20+a)<= most_sort[i][0] <= max_point+20+a:
                most_total += most_sort[i][0]*most_sort[i][1]
                num += most_sort[i][1]
        a+=1;num_copy = num             
    most_total=most_total/num
    return most_total
def review_markers(markers,ret):
    for i in range(1,ret+1):
         if markers[markers==i].size==0:
             for j in range(i+1,ret+1):
                 markers[markers == j] = j-1
         ret=ret-1
    return markers,ret

def SaltAndPepper(edge_markers,markers,part,percetage):  
    SP_NoiseImg=edge_markers 
    rand_row=np.where(markers==part)[0];rand_col=np.where(markers==part)[1]
    SP_NoiseNum=int(percetage*len(rand_row))
    for i in range(SP_NoiseNum): 
        rand_num=np.random.random_integers(0,len(rand_row)-1)   
        if np.random.random_integers(0,1)==0: 
            SP_NoiseImg[rand_row[rand_num],rand_col[rand_num]]=0 
        else: 
            SP_NoiseImg[rand_row[rand_num],rand_col[rand_num]]=255 
    return SP_NoiseImg
def max_area(closing):
    
    closeing_all=closing.copy()
    _,contours_all,hierarchy = cv2.findContours(closeing_all,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    row_all=closing.shape[0]; col_all=closing.shape[1]
    paint_max_area= np.full((row_all, col_all), fill_value=0, dtype=np.uint8)
    
    c_max = [];  max_area = 0; max_cnt = 0  
    for i in range(len(contours_all)):  
        cnt = contours_all[i]  
        area = cv2.contourArea(cnt)  
        # find max countour  
        if (area>max_area):  
            if(max_area!=0):  
                c_min = []  
                c_min.append(max_cnt)  
                cv2.drawContours(paint_max_area, c_min, -1, (0,0,0), cv2.FILLED)  
            max_area = area  
            max_cnt = cnt  
        else:  
            c_min = []  
            c_min.append(cnt)  
            cv2.drawContours(paint_max_area, c_min, -1, (0,0,0), cv2.FILLED)  
            
    c_max.append(max_cnt)        
    cv2.drawContours(paint_max_area, c_max, -1, (255, 255, 255), thickness=-1)
    return  paint_max_area
def remove_inner_invone(markers,part):
    edge_row=np.where(markers==-1)[0];edge_col=np.where(markers==-1)[1]
    area_edge=len(edge_row)
    for i in range(area_edge):
        a=edge_row[i];b=edge_col[i]
        if (a not in (0,row_all-1))and (b not in (0,col_all-1)):
            up_part=markers[a,b+1];down_part=markers[a,b-1]
            left_part=markers[a-1,b];right_part=markers[a+1,b]
        
            if (up_part not in (-1,part)):
                markers[a][b] = up_part
                continue
            elif(down_part not in (-1,part)):  
                markers[a][b] = down_part
                continue
            elif(left_part not in (-1,part)):
                markers[a][b]  = left_part
                continue
            elif(right_part not in (-1,part)):   
                markers[a][b]  = right_part
                continue
            else:
                markers[a][b]  = part
    #去除四个边界    
    markers[0,:]=markers[1,:]
    markers[row_all-1,:]=markers[row_all-2,:]
    markers[:,0]=markers[:,1]
    markers[:,col_all-1]=markers[:,col_all-2]
    return markers
#def get_edge_markers():
    
def edge_segment(original_image,markers,part):
    #    去掉非前景部分  
    row_all=original_image.shape[0]; col_all=original_image.shape[1]
    markers_combine = np.full([row_all,col_all],False)     
#    经过体积测量的标签
    markers_bool = num_to_bool(markers,part)
#    前景标签
    markers_in_fg = num_to_bool(sure_fg,0)
    for i in range(row_all):
        for j in range(col_all):
            if markers_in_fg[i][j]==False and markers_bool[i][j]==False:
                continue
            else:
                markers_combine[i][j]=True
     
    markers = remove_inner_invone(markers,part)     
    segment_of_orignal,marker_image = sgement_by_markers(original_image,markers_bool,inverse=False)
#    cv.imshow('segment_of_orignal',segment_of_orignal)

   #边界
    edge_review = np.full((row_all,col_all), fill_value=0, dtype=np.uint8)
    #if (markers_combine[markers_combine==False].size/sure_fg[sure_fg==0].size<0.95):
    edge_markers=edge(segment_of_orignal)
      
    # fg_edge_per前景边缘处先得概率
    fg_edge_per=edge_markers[edge_markers==255].size/sure_fg.size
    # fg_edge_per>0.12时需要去燥
    if (fg_edge_per>=0.12):
        segment_of_orignal = cv.GaussianBlur(segment_of_orignal, (3, 3), 0) #高斯模糊去噪 
        edge_markers=edge(segment_of_orignal)
 
    # fg_edge_per<0.13时需要加燥 （椒盐噪声）   
    while(fg_edge_per < 0.13):
        edge_markers=SaltAndPepper(edge_markers,markers,part,0.03)
        fg_edge_per=edge_markers[edge_markers==255].size/sure_fg.size
#    cv2.imshow('edge_markers',edge_markers)
       
    for i in (0,row_all-3):
        edge_markers[i,:]= 0
        edge_markers[i+1,:]= 0
        edge_markers[i+2,:]= 0    
        edge_markers[:,i]= 0
        edge_markers[:,i+1]= 0      
        edge_markers[:,i+2]= 0
    edge_original=edge(original_image)

    _,masker_background = generate_background_marker(original_image)
    background_ = apply_marker(original_image, masker_background, background = 0, inverse = True)

    edge_background = edge(background_)

    #与背景相取非 
    #与原图像相与 会丢失边界信息
    for i in range(row_all):
        for j in range(col_all):
            if edge_markers[i][j]==255 and edge_original[i][j]==255:
               edge_review[i][j] =255
            if edge_background[i][j]==255:     
               edge_review[i][j] = 0
    
#    cv.imshow('edge_markers',edge_markers)
    
    kernel = np.ones((3,3),np.uint8)
    closing = cv.morphologyEx(edge_review,cv.MORPH_CLOSE,kernel,iterations = 7)
    #清除closing在其他part的值    
    area_edge_row=np.where(closing==255)[0];area_edge_col=np.where(closing==255)[1]
    area_edge=len(area_edge_row)
    
    for i in range(area_edge):
        if markers[area_edge_row[i]][area_edge_col[i]]!=part:
           closing[area_edge_row[i]][area_edge_col[i]]=0 

    edge_closing = edge(closing)
    out_closing=np.full((row_all,col_all), fill_value=0, dtype=np.uint8)
    # find 4 dege in edge_closing
    for i in range(row_all):
        for j in range(col_all):
            if edge_closing[i][j]==255:
                out_closing[i][j]=255
                break
        else:   
            continue
        continue   
    for i in range(row_all-1,0,-1):
        for j in range(col_all-1,0,-1):
            if edge_closing[i][j]==255:
                out_closing[i][j]=255
                break
        else:   
            continue
        continue    
    for j in range(col_all):
        for i in range(row_all):
            if edge_closing[i][j]==255:
                out_closing[i][j]=255
                break
        else:   
            continue
        continue      
    for j in range(col_all-1,0,-1):
        for i in range(row_all-1,0,-1):
            if edge_closing[i][j]==255:
                out_closing[i][j]=255
                break
        else:   
            continue
        continue
    
    #闭合有效
    Block_Size=29
    imgAdapt = cv2.adaptiveThreshold(out_closing,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,Block_Size,2)
    # 消除out_closing外的imgAdapt
    for i in range(row_all):
        edge_out_count=0
        for j in range(col_all):
            if imgAdapt[i][j]==255 and edge_out_count<((Block_Size-1)/2)-1:
                imgAdapt[i][j]=0
                edge_out_count+=1
                
    for i in range(row_all-1,0,-1):
        edge_out_count=0
        for j in range(col_all-1,0,-1):
            if imgAdapt[i][j]==255 and edge_out_count<((Block_Size-1)/2)-1:
                imgAdapt[i][j]=0
                edge_out_count+=1
                
                
    for j in range(col_all):
        edge_out_count=0
        for i in range(row_all):
            if imgAdapt[i][j]==255 and edge_out_count<((Block_Size-1)/2)-1:
                imgAdapt[i][j]=0
                edge_out_count+=1  
                
                
    for j in range(col_all-1,0,-1):
        edge_out_count=0
        for i in range(row_all-1,0,-1):
            if imgAdapt[i][j]==255 and edge_out_count<((Block_Size-1)/2)-1:
                imgAdapt[i][j]=0
                edge_out_count+=1  
    
#    cv2.imshow('imgAdapt',imgAdapt) 
    
    #imgAdapt与前面分割图片的背景结合
    segment_row=np.where(segment_of_orignal==0)[0];segment_col=np.where(segment_of_orignal==0)[1]
    for i in range(len(segment_row)):
       imgAdapt[segment_row[i]][segment_col[i]]=0 

    #closing与imgAdapt与前面分割图片的背景结合，消除边界部分项相连的区域
    for i in range(row_all):
        for j in range(col_all):
            if imgAdapt[i][j] == 255 :
                closing[i][j]=255
#    cv2.imshow('closing',closing)
#    cv2.imshow('edge_closing',edge_closing)
#    cv2.imshow('out_closing',out_closing)
  
    # 计算最大的区域
    paint_max_area=max_area(closing)
    return paint_max_area
     
if __name__ == '__main__':
    

#    for k in range(1,len(files)+1):
    original_image = read_image(files['jpg8'])
    row_all=original_image.shape[0]; col_all=original_image.shape[1]
#    markers,src,surface_fg,ret = water_image(original_image)
    _,_,_,_,sure_fg,_ = Grayscale_morphology(original_image)    
#    isin = part_isin_sufg(original_image)
    part,markers = area_measure(original_image)    
   

    paint_max_area=edge_segment(original_image,markers,part)
    markers_edge=num_to_bool(paint_max_area,255)
    segment_edge_orignal,marker_image_edge =sgement_by_markers(original_image,markers_edge,inverse=False)   
    cv2.imshow('mask',paint_max_area)  
    cv2.imshow('segment_edge_orignal',segment_edge_orignal)
    
    cv2.waitKey()
    cv2.destroyAllWindows()    

      

    

'''
#    Region8_x=[-1,0,1,1,1,0,-1,-1]
#    Region8_y=[-1,-1,-1,0,1,1,1,0]
#    begin_row=np.where(out_closing==255)[0][0];begin_col=np.where(out_closing==255)[1][0]
#    begin=out_closing[begin_row][begin_col]
#    count_Region8=0;next_point=out_closing[0][0]
#    new_edge=np.fullnp.full((row_all,col_all), fill_value=0, dtype=np.uint8)
#    new_edge[begin_row][begin_col]=255
#    next_point=out_closing[begin_row][begin_col]
#    count_begin=0
#    while(next_point != begin):
#        
#        for i in range(len(Region8_x)):
#            if out_closing[begin_row+Region8_x[i]][begin_col+Region8_x[i]]==255:
#                new_edge[begin_row+Region8_x[i]][begin_col+Region8_x[i]]=255
#                if count_begin!= 0:
#                    out_closing[begin_row][begin_col]=0
#                begin_row=begin_row+Region8_x[i]
#                begin_col=begin_col+Region8_x[i]
#                next_point=out_closing[begin_row][begin_col]
#                count_begin+=1
#                break
#        else:

    
#    edge_closing_row=np.where(edge_closing==255)[0];edge_closing_col=np.where(edge_closing==255)[1]
#    edge_markers_closing=edge_closing.copy()
#    for i in range(len(edge_closing_row)):
#        if edge_markers[edge_closing_row[i]][edge_closing_col[i]]==255:
#            edge_markers_closing[edge_closing_row[i]][edge_closing_col[i]]=255
#    
#    cv2.imshow('closing_edge_markers',edge_markers_closing)
#    
 

    # 计算最大的区域
    #    paint_max_area_2=max_area(closing_more)
        
    #    paint_max_area_new=np.full((row_all,col_all), fill_value=0, dtype=np.uint8)
    #    for i in range(row_all):
    #        for j in range(col_all):
    #            if paint_max_area_2[i][j] == 255 and paint_max_area[i][j] == 255 :
    #                paint_max_area_new[i][j]=255
#    _,contours,hierarchy = cv2.findContours(edge_output,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    sgo_contours = cv2.drawContours(segment_of_orignal_gray,contours,-1,(255,255,255),3)             
##    edge_Laplacian = cv2.Laplacian(edge_output,cv2.CV_8U)
#    for i, contour in enumerate(contours):
#        cv.drawContours(segment_of_orignal_gray, contours, i, (0, 0, 255), 2)
#        
#    cv.imshow('segment_of_orignal_gray',segment_of_orignal_gray)
#    cv2.waitKey()    
    
#    _,contours,hierarchy = cv2.findContours(edge_output,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    sgo_contours = cv2.drawContours(segment_of_orignal_gray,contours,-1,(255,255,255),3)   
        # otsu method
     # otsu method
#    sgo_contours = cv2.cvtColor(sgo_contours,cv2.COLOR_BGR2GRAY)
#    threshold,imgOtsu = cv2.threshold(sgo_contours,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#    	# adaptive gaussian threshold 
#
#    imgAdapt = cv2.adaptiveThreshold(sgo_contours,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

#    cv2.imshow('segment_of_orignal_gray',segment_of_orignal_gray)
#    cv2.imshow('sgo_contours',sgo_contours) 
    
##        
#    _,contours,hierarchy = cv2.findContours(edge_candy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
##        绘制独立轮廓，如第四个轮廓
#    sgo_contours = cv2.drawContours(segment_of_orignal,contours,-1,(255,255,255),3)
#        
#    segment_of_orignal_gray = cv2.adaptiveThreshold(edge_candy,255,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY,9,2 )
##    cv2.THRESH_OTSU
#    _, segment_of_orignal_gray = cv2.threshold(segment_of_orignal_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            
        
        
#        
#        original_image_part=original_image_gray.copy()
#        original_image_part[markers!=count] = 0
#        cv.imshow('original_image_part'+'_'+str(count),original_image_part)
#        res = cv2.matchTemplate(original_image_gray,original_image_part,cv2.TM_CCOEFF)
#        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#        print(min_val, max_val, min_loc, max_loc)
#            
#    cv.waitKey()  
#    cv.destroyAllWindows()

   
#    cv.imshow('original_image',original_image)
#    original_image_of_gray = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
#    segment_of_orignal_gray = cv2.cvtColor(segment_of_orignal,cv2.COLOR_BGR2GRAY)
#    
#    ##直方图均衡化
#    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#    #cl1 = clahe.apply(segment_of_orignal_gray)
#    #cv.imshow('cl1',cl1)
#    
#    #阈值image_marker 更细化
#    _, image_marker = get_marker(segment_of_orignal_gray)
##    image_marker=cv2.bitwise_not(image_marker)
#    cv.imshow('image_marker',image_marker)
#    #for i in range(image_marker.shape[0]):
#    #    for j in range(image_marker.shape[1]):
#    #        if image_marker[i,j]==0:
#    #            image_marker[i,j]=255
#    #        else:
#    #            image_marker[i,j]=0
#    #椭圆形marker 
#    #cv.imshow('segmented_image',segmented_image)
#    
#    
#
#    plt.subplot(221), plt.title('original_image'), plt.imshow(original_image, 'gray')
#    plt.subplot(222), plt.title('image_marker'),plt.imshow(image_marker,'gray')
#    plt.subplot(223), plt.imshow(segment_of_orignal, 'gray')
#    
#    #颜色直方图
#    plt.subplot(224)
#    color = ('b', 'g', 'r')
#    for i,col in enumerate(color):
#        hist_mask = cv2.calcHist([original_image],[i],image_marker,[256],[0,256])
#        plt.plot(hist_mask,color = col)
#    plt.xlim([0,256])
#    plt.show()
#   
#    #im0 = cv.imread('Test2.jpg')
#    
#    #gray1 = cv.cvtColor(segmented_image,cv.COLOR_BGR2GRAY)
#    #cv.imshow('image1',gray1)
#    
#    ####### LEAF MARKER ############
#    
#    #Using Otsu's thresholding after gaussian filtering 
#    #blur = cv.GaussianBlur(gray1,(5,5),0)
#    #ret, thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
#    
#    #noise removal using grayscale morphology
#    kernel = np.ones((3,3),np.uint8)
#    ##
#    #closing = cv.morphologyEx(segmented_image,cv.MORPH_CLOSE,kernel,iterations = 5)
#    #cv.imshow('image',closing)
#    ##good
#    #opening = cv.morphologyEx(closing,cv.MORPH_OPEN,kernel,iterations = 3)
#    #cv.imshow('image2',opening)
#    
#    #sure background area
#    #sure_bg
#    #for k in range(segment_of_orignal.shape[2]):
#    #    for x in range(segment_of_orignal.shape[0]):
#    #        for y in range (segment_of_orignal.shape[1]):
#    #            if segmented_image[x,y,k] == np.uint8(0) :
#    #                segment_of_orignal[x,y,k]=0
#    #            else:
#    #                segment_of_orignal[x,y,k]=original_image[x,y,k]
#    #输入为二值图像 bg=255 fg= 0
#    #膨胀 可以分开两个物体（两个前景）
#    sure_bg = cv.dilate(image_marker,kernel,iterations = 2)
#    #cv.imshow('image_sure_bg',sure_bg)
#    
#    #Finding sure foreground area
#    dist_transform = cv.distanceTransform(image_marker,cv.DIST_L2,5)
#    ret, sure_fg = cv.threshold(dist_transform,0.01*dist_transform.max(),255,0)
#    #cv.imshow('dist_transform',dist_transform)
#    
#    #Finding unknown region
#    sure_fg = np.uint8(sure_fg)
#    unknown = cv.subtract(sure_bg,sure_fg)
#    #cv.imshow('unknown',unknown)
#    
#    
#    
#    #Marker labelling 0,1
#    ret, markers = cv.connectedComponents(sure_fg)
#    #cv.imshow('sure_fg',sure_fg)
#    # Add one to all labels so that sure bacground is not 0, but 1
#    markers = markers+1
#    
#    # Now, mark the region of unknown with zero
#    markers[unknown==255] = 0
#    #cv.imshow('unknown',unknown)
#    
#    markers = cv.watershed(original_image,markers)
#    original_image[markers == -1] = [255,0,0]
#    cv.imshow('original_image_1',original_image)
#    original_image_1 = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
#    
#    
#    
#    
#    gradient = cv2.morphologyEx(original_image_1,cv2.MORPH_GRADIENT,kernel)
#    #laplacian = cv2.Laplacian(segment_of_orignal,cv2.CV_64F)
#    
#    cv.imshow('gradient',gradient)
#    #原图灰度图的局部最大
#    
#    row_kernel,_=kernel.shape
#    window =row_kernel - row_kernel//2 - 1
#    local_max_original = 0;local_min_gradient = 255;local_sum_gradient = 0;local_min_markers=255
#    row_loc = 0;col_loc = 0
#    #local_mean_gradient =np.full(
#    #        (gradient.shape[0]-2*window, gradient.shape[1]-2*window), fill_value=0, dtype=np.uint8
#    #    ) 
#    #gradient_review =np.full(
#    #        (gradient.shape[0], gradient.shape[1]), fill_value=0, dtype=np.uint8
#    #    ) 
#    for i in range (window,original_image.shape[0]-window):
#        for j in range(window,original_image.shape[1]-window):
#    #        躲过边界
#            if (i >= window+1 and j>=window+1)or(
#                    (i<=original_image.shape[0]-window-1 )and(j<=original_image.shape[1]-window-1 )
#                    ):
#                for p in range(i-window,i+window+1): 
#                    for q in range(j-window,j+window+1):
#        #                local_sum_gradient=local_sum_gradient+gradient[p,q]
#                        if gradient[p,q]<local_min_gradient:
#                            local_min_gradient=gradient[p,q]
#                        if markers[p,q]<local_min_markers:
#                            local_min_markers=markers[p,q]
#                        if original_image_of_gray[p,q]>local_max_original:
#                            local_max_original = original_image_of_gray[p,q]
#                            row_loc = p;col_loc = q
#                
#    #            gradient[row_loc,col_loc]=local_min_gradient
#                gradient[row_loc,col_loc]=local_max_original
#                markers[row_loc,col_loc]=local_max_original
#                local_max_original=0
#                local_min_gradient=255
#    #                else:
#    #                    local_mean_gradient[]=local_sum_gradient//(kernel*kernel)
#    #                    gradient[i-window,j+window+1]
#    
#    cv.imshow('gradient_1',gradient)
#    original_image[markers == -1] = [255,0,0]
#    cv.imshow('original_image',original_image)
#    cv.waitKey()
   

    #gray2 = cv.cvtColor(original_image,cv.COLOR_BGR2GRAY)
    #im_color = cv.applyColorMap(gray2, cv.COLORMAP_JET)
    
    #cv.imshow('im_color',im_color)
    #k = cv.waitKey(0) & 0xFF
    #if k == 27: #wait for ESC to exit
    #    cv.destroyAllWindows()
    #elif k == ord('s'): #wait for 's' key to save and exit
    #    cv.imwrite('result.jpg',original_image)
    #    cv.destroyAllWindows()
        
    
    #w = watershed();
'''