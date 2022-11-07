import numpy as np
from track import run
import uvicorn
import shutil
import os

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.param_functions import Depends
from fastapi.responses import StreamingResponse

from inference_params import InferenceParams




app = FastAPI()

@app.post("/track")
def detect_with_url(request: Request,params: InferenceParams = Depends()):
    source, save_vid, classes,save_txt = set_tracking_params(params)

    run(source=source,classes= classes, save_vid=save_vid, save_txt=save_txt)
    if save_vid:
        return StreamingResponse(iterfile(), media_type="video/mp4")
    else:
        json = {"message":"video generated"}
        return JSONResponse(content=json)
@app.post("/upload_track")
def detect_with_upload(file: UploadFile=File(...), params: InferenceParams = Depends()):
    with open(f'{file.filename}',"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)
    
    video_path = file.filename
    source, save_vid, classes,save_txt = set_tracking_params(params)

    run(source=video_path,classes= classes, save_vid=save_vid, save_txt=save_txt)
    #run(source=buffer,classes= 0, save_vid=True, save_txt=False)
        
    

#helper function
def set_tracking_params(params:InferenceParams):
# Function that Ineference params and return them as variables
    params_dict = params.dict()
    if 'source' in params_dict :
        source = params_dict['source']
    if 'save_vid' in params_dict :
        save_vid = params_dict['save_vid']
    if 'classes' in params_dict :
        classes = params_dict['classes']
    if 'save_txt' in params_dict:
        save_txt = params_dict['save_txt']

    return source, save_vid, classes, save_txt

def iterfile():  # 
        #file_path = "os.environ.get('SAVE_PATH')"
        file_path = 'runs/track/exp2/watch_v_h4s0llOpKrU.mp4' 
        with open(file_path, mode="rb") as file_like: 
            yield from file_like 
        
