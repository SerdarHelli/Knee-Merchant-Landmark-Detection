# -*- coding: utf-8 -*-
"""
@author: serdarhelli
"""

import pandas as pd
import os
from natsort import natsorted
import numpy as np
import cv2
import tensorflow as tf
from utils import *
import argparse
import sys 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Automation Landmark Merchant")
    parser.add_argument("--DicomsPath", type=str, required=True,help="Dicom Dosyalarının Bulunduğu Path")
    parser.add_argument("--OutputPath", type=str, required=True,help="Sonuçların Kaydedileceği  Path")
    parser.add_argument("--ModelPath", type=str, required=True,help="Modelin Bulunduğu Path")

    args = vars(parser.parse_args())
    path = args["DicomsPath"]
    result_path = args["OutputPath"]
    model_path = args["ModelPath"]


    if not os.path.isdir(result_path+"/results_images"):
        path_images = os.path.join(result_path, "results_images")
        os.mkdir(path_images)
    
    if os.path.exists(result_path+"/result.csv"):
        df=pd.read_csv(result_path+"/result.csv")
    else :
        df=pd.DataFrame(columns=[ 'PatientID','PatientName',"StudyDate","InstitutionName","PatientAge","PatientSex","Pateller_Congruence_Angle","Paralel_Tilt_Angle","SOP_Instance_UID","Done"])
    
    model = tf.keras.models.load_model(model_path,custom_objects=None)
    model.summary()
    print("##########################################################################################")

    dirs=natsorted(os.listdir(path))
    print(path+"/")

    for i in range(len(dirs)):

        try:
            img,PatientName,PatientID,SOPInstanceUID,StudyDate,InstitutionAddress,PatientAge,PatientSex=crop_resize(path+"/"+dirs[i])
        except:
             print("Read - Crop - Resize // Something is wrong in : " + path+"/"+dirs[i])
             print("Oops!", sys.exc_info()[0], "occurred.")
             pass
        try:
            if (SOPInstanceUID in df["SOP_Instance_UID"]) == False :
                pre=predict(img,model)
                heatpoint=points_max_value(pre)
                img2=put_text_point(img,heatpoint)
                img2,PatellerCongruenceAngle,ParalelTiltAngle=draw_angle(img2,heatpoint)
                data = {'PatientID': PatientID, 'PatientName': PatientName,
                        'Pateller_Congruence_Angle': PatellerCongruenceAngle,
                        'Paralel_Tilt_Angle':ParalelTiltAngle,
                        'SOP_Instance_UID':SOPInstanceUID,
                        "StudyDate" :StudyDate,
                        "InstitutionName" :InstitutionAddress,
                        "PatientAge" :PatientAge ,
                        "PatientSex" :PatientSex,
                        "Done" :"Done"}
                df=df.append(data,ignore_index=True)
                cv2.imwrite(result_path+"/results_images/"+np.str(SOPInstanceUID)+".png",img2)
                df.to_csv(result_path+"/result.csv", encoding='utf-8')
                print("Done : " + np.str(SOPInstanceUID))
        except:
                print("Prediction - Save // Something is wrong in : " + str(SOPInstanceUID))
                print("Oops!", sys.exc_info()[0], "occurred.")
                data = {'PatientID': PatientID, 'PatientName': PatientName,
                        'Pateller_Congruence_Angle': PatellerCongruenceAngle,
                        'Paralel_Tilt_Angle':ParalelTiltAngle,
                        'SOP_Instance_UID':SOPInstanceUID,
                        "StudyDate" :StudyDate,
                        "InstitutionName" :InstitutionAddress,
                        "PatientAge" :PatientAge ,
                        "PatientSex" :PatientSex,
                        "Done" :"Fail"}
                df=df.append(data,ignore_index=True)
                df.to_csv(result_path+"/result.csv", encoding='utf-8')
                pass


            