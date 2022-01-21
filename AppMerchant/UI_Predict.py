
import tkinter  as tk
from PIL import Image as Pil_image, ImageTk as Pil_imageTk
import numpy as np
from tensorflow.keras.models import load_model
from Utils import *
from tkinter import filedialog as fd
from tkzoom import *
import random
import math
import cv2 
import pydicom
import os
import sys
from tkinter import ttk


def resource_path(relative_path):
    try:
    # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class App(tk.Frame):
    def __init__(self, parent):       
        
        tk.Frame.__init__(self, parent)   
        self.parent = parent   
        self.parent.tk.call("source",resource_path("src/theme/azure.tcl"))
        self.parent.tk.call("set_theme", "light")
        self.image_name=""
        self.path_image=""
        self.model = load_model(resource_path("SavedModel/model.h5"))
        self.Sheet=tk.Canvas(self.parent,height=512, width=1280,bd=0)
        self.Sheet.pack(fill="both", expand=True)  
        self.Label_OriginalImage=tk.Label(self.Sheet, text='Original Dicom',font=("Segoe Ui",18,'bold'))
        self.Label_PredictImage=tk.Label(self.Sheet, text='Predicted',font=("Segoe Ui",18,'bold'))
        
        self.Label_Results=tk.LabelFrame(self.Sheet, text='Results',font=("Segoe Ui",18),bg="#000000",fg="#DC143C")
        self.Label_Info=tk.LabelFrame(self.Sheet, text='Information of Patient',pady=10,font=("Segoe Ui", 18,'bold'))
        
        self.Save_Button=ttk.Button(master=self.Sheet, text= "Save Predicted", style='Accent.TButton',command=self.SavePredictedButton)
        self.Learn_Button=ttk.Button(master=self.Sheet, text= "Model Diagram", style='Accent.TButton',command=self.openDiagram)
        self.Crop_Button=ttk.Button(master=self.Sheet, text= "Save Cropped and Resized", style='Accent.TButton',command=self.SaveCroppedButton)
        self.OriginalImageSave_Button=ttk.Button(master=self.Sheet, text= "Save Original Image", style='Accent.TButton',command=self.SaveOriginalImageButton)

        self.Theme_Button = ttk.Button(master=self.Sheet, text="Change theme", command=self.change_theme)

        self.canvas_dicomimg=None
        self.canvas_preimg=None
        self.original_image=None
        self.cropped_image=None
        self.image=None
        self.predictions=None
        self.SOPInstanceUID=None
        self.heatpoint=None
        self.info=None
        
        try :
            self.initUI()
        except Exception as e:
            tk.messagebox.showerror('Opps... ',str(e))
        


    def initUI(self):
        self.parent.title("Knee Skyline Merchant View with AI")
        self.pack(fill="both", expand=True)
        self.grid_columnconfigure(0, weight=1)
        menubar = tk.Menu(self.parent)
        self.parent.config(menu=menubar)
        fileMenu = tk.Menu(menubar)
        fileMenu.add_command(label="Predict", command=self.onPredict)
        menubar.add_cascade(label="File", menu=fileMenu)  
        self.Label_Knowledge=tk.Label(self.Sheet,text='App Version 0.1 - By S.Serdar Helli and Andaç Hamamcı - Contact: s.serdarhelli@gmail.com',font=("Segoe Ui", 9))
        self.Theme_Button.place(x=1100,y=465)

        
        
    def Show_Labels(self,PatientName,PatientID,SOPInstanceUID,StudyDate,InstitutionAddress,PatientAge,PatientSex,PatellerCongruenceAngle,ParalelTiltAngle):
        
        
        ttk.Label(self.Label_Info, text="Patient Name:",font=("-size", 12, "-weight", "bold")).grid(row=0, column=0,pady=5)
        tk.Label(self.Label_Info, text=PatientName,font=("Segoe Ui", 12),pady=5).grid(row=0, column=1)
        # w = tk.Text(self.Label_Info, height=1, borderwidth=0,font=("Segoe Ui", 10))
        # w.insert(1.0, PatientName)
        # w.grid(row=0, column=1,pady=5)
        # w.configure(state="disabled")
        # w.configure(bg=self.Label_Info.cget("bg"), relief="flat")
        
    
        ttk.Label(self.Label_Info, text="Patient Age:",font=("-size", 12, "-weight", "bold")).grid(row=1, column=0,pady=5)
        tk.Label(self.Label_Info, text=PatientAge,font=("Segoe Ui", 12),pady=5).grid(row=1, column=1)
        # w1 = tk.Text(self.Label_Info, height=1, borderwidth=0,font=("Segoe Ui", 10))
        # w1.insert(1.0, PatientAge)
        # w1.grid(row=1, column=1,pady=5)
        # w1.configure(state="disabled")
        # w1.configure(bg=self.Label_Info.cget("bg"), relief="flat")

        
        ttk.Label(self.Label_Info, text="Patient Sex:",font=("-size", 12, "-weight", "bold")).grid(row=2, column=0,pady=5)
        tk.Label(self.Label_Info, text=PatientSex,font=("Segoe Ui", 12),pady=5).grid(row=2, column=1)
        # w2 = tk.Text(self.Label_Info, height=1,font=("Segoe Ui", 10), borderwidth=0)
        # w2.insert(1.0, PatientAge)
        # w2.grid(row=2, column=1,pady=5)
        # w2.configure(state="disabled")
        # w2.configure(bg=self.Label_Info.cget("bg"), relief="flat")
        
        ttk.Label(self.Label_Info, text="Patient ID:",font=("-size", 12, "-weight", "bold")).grid(row=3, column=0,pady=5)
        tk.Label(self.Label_Info, text=PatientID,font=("Segoe Ui", 12),pady=5).grid(row=3, column=1)
        # w3 = tk.Text(self.Label_Info, height=1,font=("Segoe Ui", 10), borderwidth=0)
        # w3.insert(1.0, PatientID)
        # w3.grid(row=3, column=1,pady=5)
        # w3.configure(state="disabled")
        # w3.configure(bg=self.Label_Info.cget("bg"), relief="flat")
        
        
        ttk.Label(self.Label_Info, text="SOP Instance UID:",font=("-size", 12, "-weight", "bold")).grid(row=4, column=0,pady=5)
        tk.Label(self.Label_Info, text=SOPInstanceUID,font=("Segoe Ui", 10),pady=5).grid(row=4, column=1)
        # w4 = tk.Text(self.Label_Info, height=1,font=("Segoe Ui", 10), borderwidth=0)
        # w4.insert(1.0, SOPInstanceUID)
        # w4.grid(row=4, column=1,pady=5)
        # w4.configure(state="disabled")
        # w4.configure(bg=self.Label_Info.cget("bg"), relief="flat")
        
        
        ttk.Label(self.Label_Info, text="Institution Address:",font=("-size", 12, "-weight", "bold")).grid(row=5, column=0,pady=5)
        tk.Label(self.Label_Info, text=InstitutionAddress,font=("Segoe Ui", 10),pady=5).grid(row=5, column=1)
        # w5 = tk.Text(self.Label_Info, height=1,font=("Segoe Ui", 10), borderwidth=0)
        # w5.insert(1.0, InstitutionAddress)
        # w5.grid(row=5, column=1,pady=5,sticky = "ew")
        # w5.configure(state="disabled")
        # w5.configure(bg=self.Label_Info.cget("bg"), relief="flat")
        
        ttk.Label(self.Label_Info, text="Study Date:",font=("-size", 12, "-weight", "bold")).grid(row=6, column=0,pady=5)
        tk.Label(self.Label_Info, text=StudyDate,font=("Segoe Ui", 10),pady=5).grid(row=6, column=1)
        # w6 = tk.Text(self.Label_Info, height=1,font=("Segoe Ui", 10), borderwidth=0)
        # w6.insert(1.0, StudyDate)
        # w6.grid(row=6, column=1,pady=5,sticky = "ew")
        # w6.configure(state="disabled")
        # w6.configure(bg=self.Label_Info.cget("bg"), relief="flat")
        
        
        degree_sign = u"\N{DEGREE SIGN}"
        tk.Label(self.Label_Results, text="Pateller Congruence Angle:"+str(round(PatellerCongruenceAngle,2))+degree_sign,padx=15,bg="#000000",font=("Segoe Ui", 20),fg='#FFFF00').grid(row=0, column=0)
        tk.Label(self.Label_Results, text="Paralel Tilt Angle :"+str(round(ParalelTiltAngle,2))+degree_sign,padx=15,font=("Segoe Ui", 20),bg="#000000",fg='#33FF33').grid(row=0, column=1)
        
        self.Save_Button.place(x=750,y=325)
        self.Crop_Button.place(x=875,y=325)
        self.Learn_Button.place(x=1075,y=325)
        self.OriginalImageSave_Button.place(x=75,y=325)

        self.Label_OriginalImage.place(x=25,y=10)
        self.Label_PredictImage.place(x=450,y=10)
        self.Label_Info.place(x=750,y=20)
        self.Label_Results.place(x=150,y=376)
        self.Label_Knowledge.place(x=10,y=475)
        
    def openOrigImage(self,event):
        image = Pil_image.fromarray(self.original_image)
        newWindow = tk.Toplevel(self.parent)
        app = Zoom_Advanced(mainframe=newWindow, path=image,name="Original Dicom Image")
        newWindow.iconbitmap(resource_path("src/orthopedic.ico"))
        app.mainloop()
        
    def openPreImage(self,event):
        image = Pil_image.fromarray(self.image)
        newWindow = tk.Toplevel(self.parent)
        app = Zoom_Advanced(mainframe=newWindow, path=image,name="Predicted Image")
        newWindow.iconbitmap(resource_path("src/orthopedic.ico"))
        app.mainloop()
        
        
    def openDiagram(self):
        image = Pil_image.open(resource_path("src/model_diagram.png"))
        newWindow = tk.Toplevel(self.parent)
        app = Zoom_Advanced(mainframe=newWindow, path=image,name="Model Diagram")
        newWindow.iconbitmap(resource_path("src/orthopedic.ico"))
        app.mainloop()    
      
    def SavePredictedButton(self):
        
        message=[]   
        if self.original_image is None or self.image is None  or self.predictions is None or self.SOPInstanceUID is None:
            tk.messagebox.showerror("Error"," There is no image which is uploaded. Please Try to Predict")
        else :
            try:
                savefolder= fd.askdirectory()
                path=savefolder+"/"+self.SOPInstanceUID
                if not os.path.isdir(path):
                    path = os.path.join(savefolder, self.SOPInstanceUID)
                    os.mkdir(path)
            except:
                path=savefolder
                    
            try:
                if not cv2.imwrite(str(path)+"/"+"Predicted.png",self.image):
                    raise Exception("Could not write image")

            except Exception as e:
               tk.messagebox.showerror('Opps, Predicted Image Save Error ',str(e))
               message.append("Predicted Image ")
               pass
            try:
                np.savetxt(str(path)+"/"+"Heatpoints.txt", self.heatpoint, delimiter=',',fmt='%s') 
            except Exception as e:
               tk.messagebox.showerror('Opps, Predicted Heatpoints Save Error ',str(e))
               message.append("Predicted Heatpoints ")
               pass
            try:   
                with open(str(path)+"/"+"Data.txt", 'w',encoding="utf-8") as file:
                    file.write(str(self.info))
            except Exception as e:
                tk.messagebox.showerror('Opps, Patient Information Save Error ',str(e))
                message.append(" Patient Information ")
                pass
                 
            if len(message)!=0:
                tk.messagebox.showinfo("Info "," Succesfully Saved Except "+str(message) )
            else:
                tk.messagebox.showinfo("Info "," Succesfully Saved " )
            
    def SaveCroppedButton(self):
        if self.cropped_image is None :
            tk.messagebox.showerror("Error"," There is no image which is uploaded. Please Try to Predict")
        else :
            
            try:
                savefolder= fd.askdirectory()
                path=savefolder+"/C_"+self.SOPInstanceUID
                if not os.path.isdir(path):
                    path = os.path.join(savefolder, "C_"+self.SOPInstanceUID)
                    os.mkdir(path)
            except:
                path=savefolder
            try:
                if not cv2.imwrite(str(path)+"/"+"Cropped.png",self.cropped_image):
                    raise Exception("Could not write image")
                else:
                    tk.messagebox.showinfo("Info "," Succesfully Saved " )

            except Exception as e:
               tk.messagebox.showerror('Opps, Error Image Save Error ',str(e))
               pass

    def SaveOriginalImageButton(self):
        if self.original_image is None :
            tk.messagebox.showerror("Error"," There is no image which is uploaded. Please Try to Predict")
        else :
            
            try:
                savefolder= fd.askdirectory()
                path=savefolder+"/O_"+self.SOPInstanceUID
                if not os.path.isdir(path):
                    path = os.path.join(savefolder, "O_"+self.SOPInstanceUID)
                    os.mkdir(path)
            except:
                path=savefolder
            try:
                if not cv2.imwrite(str(path)+"/"+"OriginalImage.png",self.original_image):
                    raise Exception("Could not write image")
                else:
                    tk.messagebox.showinfo("Info "," Succesfully Saved " )

            except Exception as e:
               tk.messagebox.showerror('Opps, Error Image Save Error ',str(e))
               pass
        
    def clear_frame(self,frame):
        # destroy all widgets from frame
        for widget in frame.winfo_children():
           widget.destroy()
           
    def change_theme(self):
        # NOTE: The theme's real name is azure-<mode>
        if self.parent.tk.call("ttk::style", "theme", "use") == "azure-dark":
            # Set light theme
            self.parent.tk.call("set_theme", "light")
        else:
            # Set dark theme
            self.parent.tk.call("set_theme", "dark")
            
            
            
    def collab_resize(self,image):
        try:
            image=modification_cropping(image)
            image=cv2.resize(image,(256,256),interpolation=cv2.INTER_NEAREST)
        except:
            image=cv2.resize(image,(256,256),interpolation=cv2.INTER_NEAREST)

        return image
  
    def get_info(self,PatientName,PatientID,SOPInstanceUID,StudyDate,InstitutionAddress,PatientAge,PatientSex,PatellerCongruenceAngle,ParalelTiltAngle):
        info={"PatientName":PatientName,"PatientID":PatientID,"SOPInstanceUID":SOPInstanceUID,
              "StudyDate":StudyDate,"InstitutionAddress":InstitutionAddress,"PatientAge":PatientAge,"PatientSex":PatientSex,
              "PatellerCongruenceAngle":PatellerCongruenceAngle,
              "ParalelTiltAngle":ParalelTiltAngle
            }
        return info
        
  
    def onPredict(self):
        try:
            self.path_image= fd.askopenfile( filetypes =[('Dicom Files', '*.dcm')])
            try:
                self.original_image,_,_,self.SOPInstanceUID,_,_,_,_=read_dicom(self.path_image.name,False,True)

            except:
                self.original_image,_,_,self.SOPInstanceUID,_,_,_,_=read_dicom(self.path_image.name,True,True)

            
            self.original_image=((self.original_image/np.max(self.original_image))*255).astype("uint8")
            self.original_image1=self.collab_resize(self.original_image)
            self.original_image1 = Pil_image.fromarray(self.original_image1)
            self.original_image1 = Pil_imageTk.PhotoImage(self.original_image1) 
            
            
            if self.canvas_dicomimg is None:
                self.canvas_dicomimg=self.Sheet.create_image(25,50, image=self.original_image1,anchor="nw")
            else:
                self.Sheet.itemconfig(self.canvas_dicomimg,image=self.original_image1)
                
            self.image,PatientName,PatientID,SOPInstanceUID,StudyDate,InstitutionAddress,PatientAge,PatientSex=crop_resize(self.path_image.name) 
            self.cropped_image=np.copy(self.image)
            self.cropped_image=((self.cropped_image/np.max(self.cropped_image))*255).astype("uint8")

            self.predictions=predict(self.image,self.model)
            self.heatpoint=points_max_value(self.predictions)
            self.image=put_text_point(self.image,self.heatpoint)
            self.image,PatellerCongruenceAngle,ParalelTiltAngle=draw_angle(self.image,self.heatpoint)
            self.image1 = Pil_image.fromarray(self.image)
            self.image1 = Pil_imageTk.PhotoImage(self.image1)
            
            if self.canvas_preimg is None:
                self.canvas_preimg=self.Sheet.create_image(450,50, image=self.image1,anchor="nw")
            else:
                self.Sheet.itemconfig(self.canvas_preimg,image=self.image1)
            
            self.clear_frame(self.Label_Results)
            self.clear_frame(self.Label_Info)
            self.Show_Labels(PatientName,PatientID,SOPInstanceUID,StudyDate,InstitutionAddress,PatientAge,PatientSex,PatellerCongruenceAngle,ParalelTiltAngle)
            self.info=self.get_info(PatientName,PatientID,SOPInstanceUID,StudyDate,InstitutionAddress,PatientAge,PatientSex,PatellerCongruenceAngle,ParalelTiltAngle)
            self.Sheet.tag_bind(self.canvas_dicomimg,'<ButtonPress-1>', self.openOrigImage)
            self.Sheet.tag_bind(self.canvas_preimg,'<ButtonPress-1>', self.openPreImage)
        except Exception as e:
            tk.messagebox.showerror('Opps... ',str(e))
            pass
    



def main():
    root= tk.Tk() 
    ex = App(root)
    root.iconbitmap(resource_path("src/orthopedic.ico"))
    root.resizable(False, False)
    root.mainloop()  


if __name__ == '__main__':
    main()












