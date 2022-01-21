# -*- coding: utf-8 -*-
"""
@author: serdarhelli
"""


import numpy as np
import math
import cv2 
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut



def find_center(img):
    thresh=(img)*(255/np.max(img))
    thresh = thresh.astype(np.uint8)
    kernel =( np.ones((5,5), dtype=np.float32))
    ret,thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY)
    thresh=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,iterations=1 )
    thresh=cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations=1 )
    thresh=cv2.erode(thresh,kernel,iterations =1)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)!=0:
        c_area=np.zeros([len(contours)])
        for i in range(len(contours)):
            c_area[i]= cv2.contourArea(contours[i])
        c_1=contours[np.argmax(c_area)]    
        M = cv2.moments(c_1)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX,cY
    else:
        return 0,0

def points_center_mass(predict):
    points=np.zeros([6,2])
    for i in range(6):
        points[i,:]=find_center(predict[0,:,:,i])
    return np.int32(points)


def points_max_value(predict):
    points=np.zeros([6,2])
    for i in range(6):
        pre=predict[0,:,:,i]
        points[i,:]=np.where(pre == pre.max())
    return np.fliplr(np.int32(points))


def read_dicom(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    # data=data*255
    # data = np.uint8(data)
    try:
        PatientName=str(dicom.PatientName.components[0])
    except:
        PatientName="Empty"
        pass
    
    try:
        PatientID=str(dicom.PatientID)
    except:
        PatientID="Empty"
        pass
        
    try:
        SOPInstanceUID=str(dicom.SOPInstanceUID.name)
    except:
        SOPInstanceUID="Empty"
        pass
        
    try:
        StudyDate=str(dicom.StudyDate)
    except:
        StudyDate="Empty"
        pass
        
    try:
        InstitutionAddress=str(dicom.InstitutionName)
    except:
        InstitutionAddress="Empty"
        pass
        
    try:
        PatientAge=str(dicom.PatientAge)
    except:
        PatientAge="Empty"
        pass
        
    try:
        PatientSex=str(dicom.PatientSex)
    except:
        PatientSex="Empty"
        pass
     
    #data -> np.uint16
    return data,PatientName,PatientID,SOPInstanceUID,StudyDate,InstitutionAddress,PatientAge,PatientSex



def modification_cropping(roi):
    if roi.shape[0]!=roi.shape[1]:
        if roi.shape[0]>roi.shape[1]:
            img2=np.zeros([roi.shape[0],roi.shape[0]])
            add=(roi.shape[0]-roi.shape[1])
            a1=add//2
            a2=add-a1
            img2[:,a1:(roi.shape[0]-a2)]=roi
            
        if roi.shape[1]>roi.shape[0]:
            img2=np.zeros([roi.shape[1],roi.shape[1]])
            add=(roi.shape[1]-roi.shape[0])
            a1=add//2
            a2=add-a1
            img2[a1:(roi.shape[1]-a2),:]=roi
    else:
        img2=roi
    return img2


def croping(img,x, y, w, h):
    if y<0:
        y=0
    if abs(w)<abs(h):
        z=np.abs(h-w)
        if img.shape[1]<x+w+(z//2):
            if x-(z//2)>0:
                img2=img[y:y+h, x-(z//2):img.shape[1]].copy()
            else:
                img2=img[y:y+h, 0:img.shape[1]].copy()
        else:
            if x-(z//2)>0:
                img2=img[y:y+h, x-(z//2):x+w+(z//2)].copy()
            else:
                img2=img[y:y+h, 0:x+w+(z//2)].copy()                
    if abs(h)<abs(w): 
        z=np.abs(h-w)
        if img.shape[0]<y+h+(z//2):
            if y-(z//2)>0:
                img2=img[y-(z//2):img.shape[0], x:x+w].copy()
            else:
                img2=img[0:img.shape[0], x:x+w].copy()
        else:
            if y-(z//2)>0:
                img2=img[y-(z//2):y+h+(z//2), x:x+w].copy()
            else:
                img2=img[0:y+h+(z//2), x:x+w].copy()           
    if abs(h)==abs(w):
        img2=img[y:y + h, x:x + w].copy()
    return img2






def crop_resize(path):
    try:
        data,PatientName,PatientID,SOPInstanceUID,StudyDate,InstitutionAddress,PatientAge,PatientSex=read_dicom(path,False,True)
    except:
        data,PatientName,PatientID,SOPInstanceUID,StudyDate,InstitutionAddress,PatientAge,PatientSex=read_dicom(path,True,True)
        pass
    img = np.copy(data)
    
    #Denoise Image 
    kernel =( np.ones((5,5), dtype=np.float32))
    img2=cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel,iterations=2 )
    img2=cv2.erode(img2,kernel,iterations =2)
    if len(img2.shape)==3:
        img2=img2[:,:,0]
        
    #Threshhold 100- 4096
    ret,thresh = cv2.threshold(img2,100, 4096, cv2.THRESH_BINARY)
    
    #To Thresh uint8 becasue "findContours" doesnt accept uint16
    thresh =((thresh/np.max(thresh))*255).astype('uint8')
    a1,b1=thresh.shape
    #Find Countours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #If There is no countour
    if len(contours)==0:
        return thresh,PatientName,PatientID,SOPInstanceUID,StudyDate,InstitutionAddress,PatientAge,PatientSex
    
    #Get Areas 
    c_area=np.zeros([len(contours)])
    for i in range(len(contours)):
        c_area[i]= cv2.contourArea(contours[i]) 
        
    #Find Max Countour    
    cnts=contours[np.argmax(c_area)]
    x, y, w, h = cv2.boundingRect(cnts)
    
    #Posibble Square 
    roi = croping(data, x, y, w, h)
    
    # Absolute Square 
    roi=modification_cropping(roi)
    
    # Resize to 256x256 with Inter_Nearest
    roi=cv2.resize(roi,(256,256),interpolation=cv2.INTER_NEAREST)
    
    return roi,PatientName,PatientID,SOPInstanceUID,StudyDate,InstitutionAddress,PatientAge,PatientSex

def put_text_point(original_img,heatpoint):
    original_img =((original_img/np.max(original_img))*255).astype('uint8')
    color = (0, 51, 204)
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    for i in range(6):
      if heatpoint[i,0]<=0 and heatpoint[i,1]<=0:
          print("L"+str(i)+" There is no Point")
      else :
          if i>2:
              coordx=0
              coordy=-(i*3)
          else:
              coordx=-(i*3)
              coordy=+(i*3)+10
          img=cv2.putText(img, "L"+str(i),(heatpoint[i,0]+coordx,heatpoint[i,1]+coordy), cv2.FONT_HERSHEY_SIMPLEX,0.35, 	color, 1)   
          img = cv2.circle(img, (heatpoint[i,0],heatpoint[i,1]), radius=2, color=color, thickness=-1)
    return img

def get_vector(pt1,pt2):
    vec=np.zeros([2])
    vec[1]=(pt2[1]-pt1[1])
    vec[0]=(pt2[0]-pt1[0])
    return vec

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def getAngle(v1, v2):
  if length(v1)==0 or length(v2)==0:
      return "Failed"
  return math.degrees(math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))))

def bisector_vector(v1,v2):
    if length(v1)==0 or length(v2) ==0:
        return [0,0]
    v1=v1/(length(v1))
    v2=v2/(length(v2))
    v3=(v1+v2)
    return v3


#magnitude 50 length to l1 to l3
def angle_patellercongruence(heatpoint,magnitude=50):
    v1=get_vector(heatpoint[1,:],heatpoint[2,:])
    v2=get_vector(heatpoint[1,:],heatpoint[0,:])
    v3=get_vector(heatpoint[1,:],heatpoint[3,:])
    v4=bisector_vector(v1,v2)
    v=np.int32(v4*magnitude)
    coord=v+heatpoint[1,:]
    if length(v3)==0:
        return "Failed",[0,0]
    angle_patellercongruence=getAngle(v3/(length(v3)),v4)
    return angle_patellercongruence,coord

def angle_paraleltilt_displacement(heatpoint):
    v1=get_vector(heatpoint[4,:],heatpoint[5,:])
    v2=get_vector(heatpoint[0,:],heatpoint[2,:])
    angle_paraleltilt=getAngle(v1,v2)
    return angle_paraleltilt


def draw_angle(img,heatpoint):
    color = (255, 26, 26)
    color2=(255, 255, 0)
    color3=(51, 255, 51)
    if np.min(heatpoint[0:3,:])<=0:
      patellercongruence,angle_paraleltilt="Failed"
      return img
    if np.min(heatpoint[3:,:])<=0:
        angle_paraleltilt="Failed"
    v1=get_vector(heatpoint[1,:],heatpoint[2,:])
    v2=get_vector(heatpoint[1,:],heatpoint[0,:])
    angle=getAngle(v1,v2)
    patellercongruence,coord=angle_patellercongruence(heatpoint)
    angle_paraleltilt=angle_paraleltilt_displacement(heatpoint)
    img=cv2.line(img,tuple( (heatpoint[1,:])), tuple((heatpoint[2,:])), color, thickness=1, lineType=8)
    img=cv2.line(img, tuple((heatpoint[1,:])), tuple((heatpoint[0,:])), color, thickness=1, lineType=8)
    img=cv2.line(img, tuple((heatpoint[1,:])), tuple((heatpoint[3,:])), color2, thickness=1, lineType=8)
    img=cv2.line(img, tuple((heatpoint[4,:])), tuple((heatpoint[5,:])), color3, thickness=1, lineType=8)
    img=cv2.line(img, tuple((heatpoint[0,:])), tuple((heatpoint[2,:])), color3, thickness=1, lineType=8)
    img=cv2.line(img,tuple( (heatpoint[1,:])), tuple(coord), color2, thickness=1, lineType=8)
    img=cv2.putText(img,"Pateller Congruence Angle :"+str(round(patellercongruence,2)),(25,25), cv2.FONT_HERSHEY_SIMPLEX,0.35, color2, 1)
    img=cv2.putText(img,"Paralel Tilt Angle :"+str(round(angle_paraleltilt,2)),(50,50), cv2.FONT_HERSHEY_SIMPLEX,0.35, color3, 1)
    img=cv2.putText(img, "Angle :"+str(round(angle,2)),(heatpoint[1,0]+10,heatpoint[1,1]+15), cv2.FONT_HERSHEY_SIMPLEX,0.35, color,1)   
    return img,patellercongruence,angle_paraleltilt
      
def predict(img,model):
    #Normalization
    img=np.float32(img/(np.max(img)))
    img=np.reshape(img,(1,256,256,1))
    predictions=model.predict(img)
    #Get Final Prediction
    pre=predictions[-1]
    return pre





