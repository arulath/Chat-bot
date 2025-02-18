import streamlit as st
from groq import Groq
from PIL import Image
import google.generativeai as genai
import pyperclip
import cv2
import io
import numpy as np
import time
from mss import mss  # For cross-platform screenshot

# Initialize clients
@st.cache_resource
def init_clients():
    groq_client = Groq(api_key="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")
    genai.configure(api_key="AIzaSyASS77AcDciKp6UoxXqq4VxyC-2v6cB910")
    return groq_client

groq_client = init_clients()  

# System messages and configurations
sys_msg = (
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)

@st.cache_resource
def init_model():
    generation_config = {
        'temperature': 0.7,
        'top_p': 1,
        'top_k': 1
    }

    safety_settings = [
        {
            'category': 'HARM_CATEGORY_HARASSMENT',
            'threshold': 'BLOCK_NONE'
        },
        {
            'category': 'HARM_CATEGORY_HATE_SPEECH',
            'threshold': 'BLOCK_NONE'
        },
        {
            'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
            'threshold': 'BLOCK_NONE'
        },
        {
            'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
            'threshold': 'BLOCK_NONE'
        }
    ]

    return genai.GenerativeModel('gemini-1.5-flash-latest',
                               generation_config=generation_config,
                               safety_settings=safety_settings)

model = init_model()

def groq_prompt(prompt, img_context):
    try:
        convo = [{'role': 'system', 'content': sys_msg}]
        if img_context:
            prompt = f'USER PROMPT: {prompt}\n\nIMAGE CONTEXT: {img_context}'
        convo.append({'role': 'user', 'content': prompt})
        chat_completion = groq_client.chat.completions.create(
            messages=convo, 
            model='llama3-70b-8192',
            max_tokens=2048
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error in groq_prompt: {str(e)}")
        return "Sorry, I encountered an error processing your request."

def function_call(prompt):
    try:
        sys_msg = (
            'You are an AI function calling model. You will determine whether extracting the users clipboard content, '
            'taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond '
            'to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
            'respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"] \n'
            'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
            'function call name exactly as I listed.'
        )

        function_convo = [{'role': 'system', 'content': sys_msg},
                         {'role': 'user', 'content': prompt}]

        chat_completion = groq_client.chat.completions.create(
            messages=function_convo, 
            model='llama3-70b-8192',
            max_tokens=2048
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error in function_call: {str(e)}")
        return "None"

def take_screenshot():
    try:
        with mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            screenshot = sct.grab(monitor)
            # Convert to PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG', quality=15)
            img_byte_arr.seek(0)
            return img_byte_arr
    except Exception as e:
        st.error(f"Error taking screenshot: {str(e)}")
        return None

def capture_webcam():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam")
            return None
        
        # Wait for camera to initialize
        time.sleep(0.5)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            # Save to bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            return img_byte_arr
        else:
            st.error("Failed to capture image")
            return None
    except Exception as e:
        st.error(f"Error capturing webcam: {str(e)}")
        return None

def vision_prompt(prompt, image_bytes):
    try:
        img = Image.open(image_bytes)
        prompt = (
            'You are the vision analysis AI that provides semantic meaning from images to provide context '
            'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
            'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
            'relevant to the user prompt. Then generate as much objective data about the image for the AI '
            f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
        )
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        st.error(f"Error in vision_prompt: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(page_title="AI Assistant", layout="wide")
st.title("AI Assistant")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create a container for the chat interface
chat_container = st.container()

# Input area at the bottom
with st.container():
    # User input
    user_input = st.text_input("Enter your message:", key="user_input")
    
    # Action buttons in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Take Screenshot", use_container_width=True):
            with st.spinner("Taking screenshot..."):
                screenshot_bytes = take_screenshot()
                if screenshot_bytes:
                    st.success("Screenshot captured!")
                    if user_input:
                        visual_context = vision_prompt(user_input, screenshot_bytes)
                        response = groq_prompt(user_input, visual_context)
                    else:
                        response = groq_prompt("Describe the screenshot.", vision_prompt("Describe the screenshot.", screenshot_bytes))
                    st.session_state.chat_history.append(("user", user_input or "Screenshot taken"))
                    st.session_state.chat_history.append(("assistant", response))
                        
    with col2:
        if st.button("Capture Webcam", use_container_width=True):
            with st.spinner("Capturing from webcam..."):
                webcam_bytes = capture_webcam()
                if webcam_bytes:
                    st.success("Webcam image captured!")
                    if user_input:
                        visual_context = vision_prompt(user_input, webcam_bytes)
                        response = groq_prompt(user_input, visual_context)
                    else:
                        response = groq_prompt("Describe the webcam image.", vision_prompt("Describe the webcam image.", webcam_bytes))
                    st.session_state.chat_history.append(("user", user_input or "Webcam image captured"))
                    st.session_state.chat_history.append(("assistant", response))

    with col3:
        if st.button("Send Message", use_container_width=True):
            if user_input:
                with st.spinner("Processing..."):
                    call = function_call(user_input)
                    if 'extract clipboard' in call:
                        try:
                            clipboard_content = pyperclip.paste()
                            user_input = f"{user_input}\n\n CLIPBOARD CONTENT: {clipboard_content}"
                        except Exception as e:
                            st.error(f"Error accessing clipboard: {str(e)}")
                    response = groq_prompt(user_input, None)
                    st.session_state.chat_history.append(("user", user_input))
                    st.session_state.chat_history.append(("assistant", response))
            else:
                st.warning("Please enter a message")

    with col4:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# Display chat history
with chat_container:
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"🧑 **You:** {message}")
        else:
            st.markdown(f"🤖 **Assistant:** {message}")