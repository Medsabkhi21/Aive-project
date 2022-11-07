FROM ultralytics/yolov5:latest
# the image will clone the yolov5 model into /usr/src/yolov5 
WORKDIR /usr/src/
RUN git clone --recurse-submodules https://github.com/Medsabkhi21/humans-tracker.git .  

RUN pip install -r requirements.txt
EXPOSE 8000

CMD ["uvicorn" ,"app:server" ,"--reload"]

# to use the tracker  directly: CMD ["python","tracker.py","--args"]