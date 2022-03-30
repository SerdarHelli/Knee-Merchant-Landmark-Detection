



import streamlit as st 

import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from Utils import *
from io import BytesIO

model_path="C:/Users/10121335/Desktop/Merchant_Landmark_V1_30122021/30122021SavedModels/SavedModel300Epochs"
model = tf.keras.models.load_model(model_path,custom_objects=None)




st.subheader("Upload Merchant Knee View")
image_file = st.file_uploader("Upload Images", type=["dcm"])


if image_file is not None:
    st.text("Making A Prediction ....")

    
    try:
        data,PatientName,PatientID,SOPInstanceUID,StudyDate,InstitutionAddress,PatientAge,PatientSex=read_dicom(image_file,False,True)
    except:
        data,PatientName,PatientID,SOPInstanceUID,StudyDate,InstitutionAddress,PatientAge,PatientSex=read_dicom(image_file,True,True)
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
            roi= thresh
        
    else:
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
        
    pre=predict(roi,model)
    heatpoint=points_max_value(pre)
    output=put_text_point(roi,heatpoint)
    output,PatellerCongruenceAngle,ParalelTiltAngle=draw_angle(output,heatpoint)
    data_text = {'PatientID': PatientID, 'PatientName': PatientName,
            'Pateller_Congruence_Angle': PatellerCongruenceAngle,
            'Paralel_Tilt_Angle':ParalelTiltAngle,
            'SOP_Instance_UID':SOPInstanceUID,
            "StudyDate" :StudyDate,
            "InstitutionName" :InstitutionAddress,
            "PatientAge" :PatientAge ,
            "PatientSex" :PatientSex,
            }
    
    
    
    col1, col2 = st.columns(2)
    with col1:
        st.text("Original Dicom Image")

        st.image(np.uint8((data/np.max(data)*255)),width=350)

    
    with col2:
        st.text("Predicted Image ")

        st.image(np.uint8(output),width=350)

    
    
    st.write(data_text)
    
