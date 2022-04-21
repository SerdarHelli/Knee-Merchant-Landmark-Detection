# Automatation-Drawing-Merchant-Lines-and-Angles-of-Knee


As soon as possible , the paper will be published 


### The Dataset

For training algorithm, the training data was prepared from [LERA - Lower Extremity Radiographs](https://aimi.stanford.edu/lera-lower-extremity-radiographs) . It has consists of Merchant view of the knee and their landmark heatmaps.

##### Dataset Description 
In this retrospective, HIPAA-compliant, IRB-approved study, we collected data from 182 patients who underwent a radiographic examination at the Stanford University Medical Center between 2003 and 2014. The dataset consists of images of the foot, knee, ankle, or hip associated with each patient.


[***On Hugging Face App*** , Check Out !](https://huggingface.co/spaces/SerdarHelli/Knee-View-Merchant-Landmark-Detection)


### Train Your Own Model
<img align="left" width="33%" src="https://github.com/ImagingYeditepe/Automatation-Drawing-Merchant-Lines-and-Angles-of-Knee/blob/main/Figures/example_data.png">

 Firstly , you need to load resized data as a numpy array . 
 ```python
 #For An Example
 import numpy as np
 
 images=np.load("To_Path/stanford_merchantdata256x256.npz")
 #X Merchant Views
 x=images.f.x
 #Y Heatmaps 
 y=images.f.y
 ```
 Then , you can train your own model with one click. For the code running smoothly ,  you should care of paths. For example , " np.load" , "model.save" etc... Before you run the code , you should check the code and its comments.
 
 

### MerchantApp

It has easy usage. After open app , you should click file , then click predict. When the file dialog is open , you can choose your dicom file.

[Download App](https://drive.google.com/file/d/10PoyChem-OduniI0AVLBPeHN3sHg_DI1/view?usp=sharing)

<img src="https://github.com/ImagingYeditepe/Automatation-Drawing-Merchant-Lines-and-Angles-of-Knee/blob/main/Figures/merchantapp.png" alt="merchantapp" width="%100" >


### On Command

For usage on command . In command, you can predict many dicoms file. 

-- DicomsPath  = Your Dicoms File of Path. The file must consist of dicoms .  It can have one more than. It only supports dicom extension. 

-- OutputPath = Your Results File of Path. The predictions will be saved into this file. It will consist of csv and the predicted images.

-- ModelPath = Your Trained Model of Path . 

```command
#Example For Usage 

cd Automatation-Drawing-Merchant-Lines-Angles-of-Knee

pip install -r requirement.txt

python main.py --DicomsPath "To_DicomsPath" --OutputPath "To_YourOutputFolderPath" --ModelPath "To_ModelPath"

```

### Methods

Trained Model will be shared as soon as ! 

##### Model   
<p align="center">

 <img src="https://github.com/ImagingYeditepe/Automatation-Drawing-Merchant-Lines-and-Angles-of-Knee/blob/main/Figures/model_diagram.png" alt="model_diagram" width="66%">
</p>

##### Adaptive Wing Loss

<p align="center">
  <img src="https://github.com/ImagingYeditepe/Automatation-Drawing-Merchant-Lines-and-Angles-of-Knee/blob/main/Figures/adaptive_wingloss.png" alt="awl" width="66%">
</p>


