import streamlit as st
import requests
def main():
    st.title("Objects tracker for videos using Yolov5 and Deepsort")
    inference_msg = st.empty()
    st.header("Configuration")

    input_source = st.radio("Select input source",('Youtube', 'Local video'))
    
    save_output_video = st.radio("Save output video?",('Yes', 'No'))
    classes_dropdown = st.radio("which classes you want to track?",('humans', 'cars'))

    if save_output_video == 'Yes':
        save_vid = True
    else:
        save_vid = True

    if classes_dropdown == 'humans':
        classes = 0
    else:
        classes=3
    
    # ------------------------- LOCAL VIDEO ------------------------------
    if input_source == "Local video":
        video = st.file_uploader("Select input video", type=["mp4", "avi"], accept_multiple_files=False)
        
        if st.button("Start tracking"):
            pass

   
    # -------------------------- Youtube ------------------------------
    if input_source == "Youtube":
        url = st.text_input("Youtube Video URL", "https://www.youtube.com/watch?v=h4s0llOpKrU")
        if st.button("Start tracking"):
            params = {"source": url,"save_vid":save_vid,"classes":classes }
            stframe = st.empty()

            res = requests.post(f"backend:8000/track",json = params)
            json_str = res.json()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
