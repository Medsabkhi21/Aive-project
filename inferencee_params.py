from pydantic import BaseModel
from typing import Optional


class InferenceParams(BaseModel):
    yolo_weights:Optional[str] = 'yolov5s.pt'
    reid_weights: Optional[str] ='osnet_x0_25_msmt17.pt'
    source: Optional[str]=''
    classes: Optional[int]=0 
    save_vid: Optional[int]= 1
    save_txt: Optional[int]=0
    
