import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
import asyncio
import sys
import os
import time
import random
import string
import requests
import json
import base64
import tempfile
import re
import math
from google import genai

# API Client Libraries (ensure these are installed)
from playwright.sync_api import sync_playwright, Playwright
import boto3
from botocore.exceptions import NoCredentialsError
from google_images_search import GoogleImagesSearch
import openai # For DALL-E and potentially other OpenAI models
from openai import OpenAI as OpenAIClient # Explicitly for the client
from google import genai as google_genai # For Gemini
import anthropic
# Set up page config
st.set_page_config(layout="wide",page_title= "Creative Gen", page_icon="🎨")

# --- Asyncio setup for Playwright on Windows ---
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception as e:
        st.warning(f"Could not set WindowsProactorEventLoopPolicy for asyncio: {e}")

# --- Load Secrets (Ensure these are in your Streamlit secrets) ---
OPENAI_API_KEY_SECRET = st.secrets.get("GPT_API_KEY")
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] # Expects a list
GOOGLE_API_KEYS_SECRET = st.secrets.get("GOOGLE_API_KEY") # Expects a list
GOOGLE_CX_SECRET = st.secrets.get("GOOGLE_CX")
AWS_ACCESS_KEY_ID_SECRET = st.secrets.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY_SECRET = st.secrets.get("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME_SECRET = st.secrets.get("S3_BUCKET_NAME")
AWS_REGION_SECRET = st.secrets.get("AWS_REGION", "us-east-1")
GPT_API_KEY_SECRET = st.secrets.get("GPT_API_KEY") # Potentially different from OPENAI_API_KEY_SECRET
FLUX_API_KEYS_SECRET = st.secrets.get("FLUX_API_KEY", []) # Expects a list
ANTHROPIC_API_KEY_SECRET = st.secrets.get("ANTHROPIC_API_KEY")

# --- Initialize API Clients ---
# OpenAI Client (for DALL-E and general GPT if GPT_API_KEY_SECRET is the same)
if OPENAI_API_KEY_SECRET:
    openai.api_key = OPENAI_API_KEY_SECRET
    oai_client = OpenAIClient(api_key=OPENAI_API_KEY_SECRET)
else:
    oai_client = None
    st.error("OpenAI API Key (for DALL-E) is not configured in secrets.")

# Separate OpenAI client if GPT_API_KEY is different (for chatGPT function)
if GPT_API_KEY_SECRET:
    chatgpt_oai_client = OpenAIClient(api_key=GPT_API_KEY_SECRET)
else:
    chatgpt_oai_client = None # chatGPT function will need to handle this or use oai_client

# Gemini Client (will be initialized per call with random key)
# Anthropic Client (will be initialized per call)

# --- Global Variables / Constants ---
PREDICT_POLICY = """  Approved CTAs: Use calls-to-action like "Learn More" or "See Options" that clearly indicate leading to an article. Avoid CTAs like "Apply Now" or "Shop Now" "Today" that promise value not offered on the landing page.  \nProhibited Language: Do not use urgency terms ("Click now"), geographic suggestions ("Near you"), or superlatives ("Best") or "Limited Time" "Last Spots", "WARNING", "URGENT" "Get Today" .never use "Today"!. never use "We" or "Our"\n  \nEmployment/Education Claims: Do not guarantee employment benefits (like high pay or remote work) or education outcomes (like degrees or job placements).  \nFinancial Ad Rules: Do not guarantee loans, credit approval, specific investment returns, or debt relief. Do not offer banking, insurance, or licensed financial services. Avoid showing money bills \n"Free" Promotions: Generally avoid promoting services/products as "free". Exceptions require clarity: directing to an info article about a real free service, promoting a genuinely free course, or advertising free trials with clear terms. USE text on image, the most persuasive as you can you can.The above is regarding the text, for  visual elements and design use it  make up for the policy to make the design very enticing!!!!  .the above is NOT relevent to the visual aspect of the image!  The visual design must be extremely enticing to compensate for the strict text limitations """
GLOBAL_IMAGE_COLS_FOR_DATAFRAME = [] # To be populated before creating the final CSV

# --- Helper Functions (Copied from your script) ---

def shift_left_and_pad(row):
    """
    Utility function to left-shift non-null values in each row
    and pad with empty strings, preserving columns' order.
    """
    valid_values = [x for x in row if pd.notna(x)]  # Filter non-null values
    padded_values = valid_values + [''] * (len(image_cols) - len(valid_values))  # Pad with empty strings
    return pd.Series(padded_values[:len(image_cols)])  # Ensure correct length

def log_function_call(func):
    """
    Decorator to log function calls and return values.
    """
    def wrapper(*args, **kwargs):
        logger.info(f"CALL: {func.__name__} - args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        logger.info(f"RETURN: {func.__name__} -> {result}")
        return result
    return wrapper

#@log_function_call
def fetch_google_images(query, num_images=3, max_retries = 5 ):

    for trial in range(max_retries):
        
        """
        Fetch images from Google Images using google_images_search.
        Splits query by '~' to handle multiple search terms if needed.
        """
        terms_list = query.split('~')
        res_urls = []
        for term in terms_list:
            API_KEY = random.choice(st.secrets["GOOGLE_API_KEY"])

            CX = st.secrets["GOOGLE_CX"]

            gis = GoogleImagesSearch(API_KEY, CX)

            search_params = {
                'q': term,
                'num': num_images,
            }

            try:
                gis.search(search_params)
                image_urls = [result.url for result in gis.results()]
                res_urls.extend(image_urls)
            except Exception as e:
                st.text(f"Error fetching Google Images for '{query}': {e}")
                res_urls.append([])
                time.sleep(5)
        return list(set(res_urls))
def play_sound(audio_file_path):
    """Plays an audio file in the Streamlit app."""
    with open(audio_file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(md, unsafe_allow_html=True)




#@log_function_call
def install_playwright_browsers():
    """
    Install Playwright browsers (Chromium) if not installed yet.
    """
    try:
        os.system('playwright install-deps')
        os.system('playwright install chromium')
        return True
    except Exception as e:
        st.error(f"Failed to install Playwright browsers: {str(e)}")
        return False


# Install playwright if needed
if 'playwright_installed' not in st.session_state:
    st.session_state.playwright_installed = install_playwright_browsers()

# --------------------------------------------
# Load Secrets
# --------------------------------------------
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
S3_BUCKET_NAME = st.secrets["S3_BUCKET_NAME"]
AWS_REGION = st.secrets.get("AWS_REGION", "us-east-1")
GPT_API_KEY = st.secrets["GPT_API_KEY"]
FLUX_API_KEY = st.secrets["FLUX_API_KEY"]

# client = OpenAI(api_key=GPT_API_KEY)

#@log_function_call
def upload_pil_image_to_s3(
    image, 
    bucket_name, 
    aws_access_key_id, 
    aws_secret_access_key, 
    object_name='',
    region_name='us-east-1', 
    image_format='PNG'
):
    """
    Upload a PIL image to S3 in PNG (or other) format.
    """
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id.strip(),
            aws_secret_access_key=aws_secret_access_key.strip(),
            region_name=region_name.strip()
        )

        if not object_name:
            object_name = f"image_{int(time.time())}_{random.randint(1000, 9999)}.{image_format.lower()}"

        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format=image_format)
        img_byte_arr.seek(0)

        s3_client.put_object(
            Bucket=bucket_name.strip(),
            Key=object_name,
            Body=img_byte_arr,
            ContentType=f'image/{image_format.lower()}'
        )

        url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}"
        return url

    except Exception as e:
        st.error(f"Error in S3 upload: {str(e)}")
        return None


def gemini_text(
    prompt: str,
    
    api_key: str = random.choice(GEMINI_API_KEY),
    model_id: str = "gemini-2.5-pro-exp-03-25",
    api_endpoint: str = "generateContent"
) -> str | None:
    if is_pd_policy_global : prompt += PREDICT_POLICY

    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("Error: API key not provided and GEMINI_API_KEY environment variable not set.")
        return None

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:{api_endpoint}?key={api_key}"

    headers = {
        "Content-Type": "application/json",
    }



    request_data = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                ],
            },
        ],
        "generationConfig": {
            "responseMimeType": "text/plain",
        },
    }

    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=request_data,
            timeout=60
        )
        response.raise_for_status()

        st.text(response.json())

        return response.json()['candidates'][0]['content']['parts'][0]['text'].replace('```','')

    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        if 'response' in locals() and response is not None:
             print(f"Response status: {response.status_code}")
             print(f"Response text: {response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def gemini_text_lib(prompt,model ='gemini-2.5-pro-exp-03-25', is_with_file=False,file_url = None ):
    if is_pd_policy_global : prompt += PREDICT_POLICY

#st.text(prompt)


    client = genai.Client(api_key=random.choice(GEMINI_API_KEY))


    try:
        if is_with_file:
            file_extension ='jpg'
            with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix=file_extension or '.tmp') as temp_file:
                st.text(file_url)
                res = requests.get(file_url)
                res.raise_for_status()  
                temp_file.write(res.content)
                
                file = client.files.upload(file=temp_file.name, config={'mime_type' :'image/jpeg'})
                response = client.models.generate_content(
                    model=model, contents=  [prompt, file]

                )
        elif not is_with_file:
            response = client.models.generate_content(
                model=model, contents=  prompt
            )

        return response.text
    except Exception as e:
        st.text('gemini_text_lib error ' + str(e))
        time.sleep(4)
        return None




#@log_function_call
def chatGPT(prompt, model="gpt-4o", temperature=1.0,reasoning_effort=''):

    if is_pd_policy_global : prompt += PREDICT_POLICY
    try:
    
 
        st.write("Generating image description...")
        headers = {
            'Authorization': f'Bearer {GPT_API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': model,
            'temperature': temperature,
            "input" : prompt,
            'reasoning': {"effort": reasoning_effort}
        }

        if temperature == 0: data.pop('temperature')
        if reasoning_effort == '': data.pop('reasoning')

        if 'o1' in model or 'o3' in model:

            response = requests.post('https://api.openai.com/v1/responses', headers=headers, json=data)
            content = json.loads(response.content)['output'][1]['content'][0]['text']
            # st.text(content)
            return content



        else: 
            response = requests.post('https://api.openai.com/v1/responses', headers=headers, json=data)
            content = json.loads(response.content)['output'][0]['content'][0]['text']
            # st.text(content)
            return content

    except Exception as e:
        st.text(f"Error in chatGPT: {str(e)}")
        st.text(response.json())

        return None


def claude(prompt , model = "claude-3-7-sonnet-20250219", temperature=1 , is_thinking = False, max_retries = 10):
    if is_pd_policy_global : prompt +=   PREDICT_POLICY
    tries = 0

    while tries < max_retries:
        try:
        
        
        
            client = anthropic.Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=st.secrets["ANTHROPIC_API_KEY"])
        
            if is_thinking == False:
                    
                message = client.messages.create(
                    
                model=model,
                max_tokens=20000,
                temperature=temperature,
                
                top_p= 0.8,

                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
                return message.content[0].text
            if is_thinking == True:
                message = client.messages.create(
                    
                model=model,
                max_tokens=20000,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                thinking = { "type": "enabled",
                "budget_tokens": 16000}
            )
                return message.content[1].text
        
        
        
            print(message)
            return message.content[0].text

        except Exception as e:
            st.text(e)
            tries += 1 
            time.sleep(5)

#@log_function_call
def gen_flux_img(prompt, height=784, width=960):
    """
    Generate images using FLUX model from the Together.xyz API.
    """
    while True:
        try:
            url = "https://api.together.xyz/v1/images/generations"
            payload = {
                "prompt": prompt,
                "model": "black-forest-labs/FLUX.1-schnell-Free",
                "steps": 4,
                "n": 1,
                "height": height,
                "width": width,
            }
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {random.choice(FLUX_API_KEY)}"
            }
            response = requests.post(url, json=payload, headers=headers)
            return response.json()["data"][0]["url"]
        except Exception as e:
            print(e)
            if "NSFW" in str(e):
                return None
            time.sleep(2)

def gen_gemini_image(prompt, trys = 0):

    while trys < 40 :

        api = random.choice(GEMINI_API_KEY)


        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent?key={api}"

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": ( prompt
                                
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "" #INSERT_INPUT_HERE
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.65,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 8192,
                "responseMimeType": "text/plain",
                "responseModalities": ["image", "text"]
            }
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            res_json = response.json()
            try:
                image_b64 = res_json['candidates'][0]["content"]["parts"][0]["inlineData"]['data']
                image_data = base64.decodebytes(image_b64.encode())

                return Image.open(BytesIO(image_data))
            except Exception as e:
                trys +=1
                print("Failed to extract or save image:", e)
        else:
            trys +=1
            print("Error:")
            st.text(response.text)



def gen_flux_img_lora(prompt,height=784, width=960 ,lora_path="https://huggingface.co/ddh0/FLUX-Amateur-Photography-LoRA/resolve/main/FLUX-Amateur-Photography-LoRA-v2.safetensors?download=true"):
    retries =0
    while retries < 10:
        try:
           

            url = "https://api.together.xyz/v1/images/generations"
            headers = {
                "Authorization": f"Bearer {random.choice(FLUX_API_KEY)}",  # Replace with your actual API key
                "Content-Type": "application/json"
            }
            data = {
                "model": "black-forest-labs/FLUX.1-dev-lora",
                "prompt":"candid unstaged taken with iphone 8 : " +  prompt,
                "width": width,
                "height": height,
                "steps": 20,
                "n": 1,
                "response_format": "url",
                "image_loras": [
                    {
                        "path": "https://huggingface.co/ddh0/FLUX-Amateur-Photography-LoRA/resolve/main/FLUX-Amateur-Photography-LoRA-v2.safetensors?download=true",
                        "scale": 0.99
                    }
                ],
                "update_at": "2025-03-04T16:25:21.474Z"
            }

            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                # Assuming the response contains the image URL in the data
                response_data = response.json()
                image_url = response_data['data'][0]['url']
                print(f"Image URL: {image_url}")
                return image_url
            else:
                print(f"Request failed with status code {response.status_code}")

    
        except Exception as e:
            time.sleep(3)
            retries +=1
            st.text(e)



#@log_function_call
def capture_html_screenshot_playwright(html_content,width = 1000, height = 1000):
    """
    Use Playwright to capture a screenshot of the given HTML snippet.
    """
    if not st.session_state.playwright_installed:
        st.error("Playwright browsers not installed properly")
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            page = browser.new_page(viewport={'width': width, 'height': height})

            with NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
                f.write(html_content)
                temp_html_path = f.name

            page.goto(f'file://{temp_html_path}')
            page.wait_for_timeout(1000)
            screenshot_bytes = page.screenshot()

            browser.close()
            os.unlink(temp_html_path)

            return Image.open(BytesIO(screenshot_bytes))
    except Exception as e:
        st.error(f"Screenshot capture error: {str(e)}")
        return None

#@log_function_call
def save_html(headline, image_url, cta_text, template, tag_line='', output_file="advertisement.html"):
    """
    Returns an HTML string based on the chosen template ID (1..6, 41, 42, etc.).
    """
    # Template 1
    if template == 1:
        html_template = f"""
       <!DOCTYPE html>
       <html lang="en">
       <head>
           <meta charset="UTF-8">
           <meta name="viewport" content="width=device-width, initial-scale=1.0">
           <title>Ad Template</title>
           <style>
               body {{
                   font-family: 'Gisha', sans-serif;
                   font-weight: 550;
                   margin: 0;
                   padding: 0;
                   display: flex;
                   justify-content: center;
                   align-items: center;
                   height: 100vh;
               }}
               .ad-container {{
                   width: 1000px;
                   height: 1000px;
                   border: 1px solid #ddd;
                   border-radius: 20px;
                   box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
                   display: flex;
                   flex-direction: column;
                   justify-content: space-between;
                   align-items: center;
                   padding: 30px;
                   background: url('{image_url}') no-repeat center center/cover;
                   background-size: contain;
                   text-align: center;
               }}
               .ad-title {{
                   font-size: 3.2em;
                   margin-top: 10px;
                   color: #333;
                   background-color:white;
                   padding: 20px 40px;
                   border-radius: 20px;
               }}
               .cta-button {{
                   font-weight: 400;
                   display: inline-block;
                   padding: 40px 60px;
                   font-size: 3em;
                   color: white;
                   background-color: #FF5722;
                   border: none;
                   border-radius: 20px;
                   text-decoration: none;
                   cursor: pointer;
                   transition: background-color 0.3s ease;
                   margin-bottom: 20px;
               }}
               .cta-button:hover {{
                   background-color: #E64A19;
               }}
           </style>
       </head>
       <body>
           <div class="ad-container">
               <div class="ad-title">{headline}!</div>
               <a href="#" class="cta-button">{cta_text}</a>
           </div>
       </body>
       </html>
        """
    # Template 2
    elif template == 2:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template</title>
            <style>
                body {{
                    font-family: 'Gisha', sans-serif;
                    font-weight: 550;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }}
                .ad-container {{
                    width: 1000px;
                    height: 1000px;
                    border: 2px solid black;
                    box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
                    display: flex;
                    flex-direction: column;
                    overflow: hidden;
                    position: relative;
                }}
                .ad-title {{
                    font-size: 3.2em;
                    color: #333;
                    background-color: white;
                    padding: 20px;
                    text-align: center;
                    flex: 0 0 20%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
                .ad-image {{
                    flex: 1 1 80%;
                    background: url('{image_url}') no-repeat center center/cover;
                    background-size: fill;
                    position: relative;
                }}
                .cta-button {{
                    font-weight: 400;
                    display: inline-block;
                    padding: 20px 40px;
                    font-size: 3.2em;
                    color: white;
                    background-color: #FF5722;
                    border: none;
                    border-radius: 20px;
                    text-decoration: none;
                    cursor: pointer;
                    transition: background-color 0.3s ease;
                    position: absolute;
                    bottom: 10%;
                    left: 50%;
                    transform: translateX(-50%);
                    z-index: 10;
                }}
            </style>
        </head>
        <body>
            <div class="ad-container">
                <div class="ad-title">{headline}</div>
                <div class="ad-image">
                    <a href="#" class="cta-button">{cta_text}</a>
                </div>
            </div>
        </body>
        </html>
        """
    # Template 3
    elif template in  [3,7]:
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: 'Boogaloo', 'Segoe UI Emoji', 'Apple Color Emoji', sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    background: #f0f0f0;
                }}
                .container {{
                    position: relative;
                    width: 1000px;
                    height: 1000px;
                    margin: 0;
                    padding: 0;
                    overflow: hidden;
                    box-shadow: 0 0 20px rgba(0,0,0,0.2);
                }}
                .image {{
                    width: 1000px;
                    height: 1000px;
                    object-fit: cover;
                    filter: saturate(130%) contrast(110%);
                    transition: transform 0.3s ease;
                }}
                .image:hover {{
                    transform: scale(1.05);
                }}
                .overlay {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    min-height: 14%;
                    background: red;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    padding: 20px;
                    box-sizing: border-box;
                }}
                .overlay-text {{
                    color: #FFFFFF;
                    font-size: 4em;
                    text-align: center;
                    text-shadow: 2.5px 2.5px 2px #000000;
                    letter-spacing: 2px;
                    font-family: 'Boogaloo', 'Segoe UI Emoji', 'Apple Color Emoji', sans-serif;
                    margin: 0;
                    word-wrap: break-word;
                }}
                .cta-button {{
                    position: absolute;
                    bottom: 10%;
                    left: 50%;
                    transform: translateX(-50%);
                    padding: 20px 40px;
                    background: blue;
                    color: white;
                    border: none;
                    border-radius: 50px;
                    font-size: 3.5em;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    font-family: 'Boogaloo', 'Segoe UI Emoji', 'Apple Color Emoji', sans-serif;
                    text-transform: uppercase;
                    letter-spacing: 2px;
                    box-shadow: 0 5px 15px rgba(255,107,107,0.4);
                }}
                .cta-button:hover {{
                    background: #4ECDC4;
                    transform: translateX(-50%) translateY(-5px);
                    box-shadow: 0 8px 20px rgba(78,205,196,0.6);
                }}
                @keyframes shine {{
                    0% {{ left: -100%; }}
                    100% {{ left: 200%; }}
                }}
            </style>
            <link href="https://fonts.googleapis.com/css2?family=Boogaloo&display=swap" rel="stylesheet">
        </head>
        <body>
            <div class="container">
                <img src="{image_url}" class="image" alt="Health Image">
                <div class="overlay">
                    <h1 class="overlay-text">{headline}</h1>
                </div>
                <button class="cta-button">
                    {cta_text}
                    <div class="shine"></div>
                </button>
            </div>
        </body>
        </html>
        """
    # Template 4
    elif template == 4:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Nursing Careers in the UK</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap');
                @font-face {{
                    font-family: 'Calibre';
                    src: url('path-to-calibre-font.woff2') format('woff2');
                }}
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background-color: #F4F4F4;
                }}
                .container {{
                    position: relative;
                    width: 1000px;
                    height: 1000px;
                    background-image: url('{image_url}');
                    background-size: cover;
                    background-position: center;
                    border-radius: 10px;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
                }}
                .text-overlay {{
                    position: absolute;
                    width: 95%;
                    background-color: rgba(255, 255, 255, 1);
                    padding: 30px;
                    border-radius: 10px;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    text-align: center;
                }}
                .small-text {{
                    font-size: 36px;
                    font-weight: bold;
                    color: #333;
                    margin-bottom: 10px;
                    font-family: 'Calibre', Arial, sans-serif;
                }}
                .primary-text {{
                    font-size: 60px;
                    font-weight: bold;
                    color: #FF8C00;
                    font-family: 'Montserrat', sans-serif;
                    line-height: 1.2;
                    text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="text-overlay">
                    <div class="small-text">{cta_text}</div>
                    <div class="primary-text">{headline}</div>
                </div>
            </div>
        </body>
        </html>
        """
    # Template 5
    elif template == 5:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Landing Page Template</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@100..900&display=swap');
                @import url('https://fonts.googleapis.com/css2?family=Noto+Color+Emoji&display=swap');
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    width: 1000px;
                    height: 1000px;
                    margin: 0 auto;
                    font-family: 'Outfit', sans-serif;
                }}
                .container {{
                    width: 100%;
                    height: 100%;
                    display: flex;
                    flex-direction: column;
                    position: relative;
                    object-fit: fill;
                }}
                .image-container {{
                    width: 100%;
                    height: 60%;
                    background-color: #f0f0f0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .image-container img {{
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                }}
                .content-container {{
                    width: 100%;
                    height: 40%;
                    background-color: #121421;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 2rem;
                    gap: 2rem;
                }}
                .main-text {{
                    color: white;
                    font-size: 3.5rem;
                    font-weight: 700;
                    text-align: center;
                }}
                .cta-button {{
                    background-color: #ff0000;
                    color: white;
                    padding: 1rem 2rem;
                    font-size: 3.5rem;
                    font-weight: 700;
                    font-family: 'Outfit', sans-serif;
                    border: none;
                    font-style: italic;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: background-color 0.3s ease;
                }}
                .cta-button:hover {{
                    background-color: #cc0000;
                }}
                .intersection-rectangle {{
                    position: absolute;
                    max-width: 70%;
                    min-width: max-content;
                    height: 80px;
                    background-color: #121421;
                    left: 50%;
                    transform: translateX(-50%);
                    top: calc(60% - 40px);
                    border-radius: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 0 40px;
                }}
                .rectangle-text {{
                    font-family: 'Noto Color Emoji', sans-serif;
                    color: #66FF00;
                    font-weight: 700;
                    text-align: center;
                    font-size: 45px;
                    white-space: nowrap;
                }}
                .highlight {{
                    color: #FFFF00;
                    font-size: 3.5rem;
                    font-style: italic;
                    font-weight: 1000;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="image-container">
                    <img src="{image_url}" alt="Placeholder image">
                </div>
                <div class="intersection-rectangle">
                    <p class="rectangle-text">{tag_line.upper()}</p>
                </div>
                <div class="content-container">
                    <h1 class="main-text">{headline}</h1>
                    <button class="cta-button">{cta_text}</button>
                </div>
            </div>
        </body>
        </html>
        """
    # Template 41
    elif template == 41:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Nursing Careers in the UK</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap');
                @font-face {{
                    font-family: 'Calibre';
                    src: url('path-to-calibre-font.woff2') format('woff2');
                }}
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background-color: #F4F4F4;
                }}
                .container {{
                    position: relative;
                    width: 1000px;
                    height: 1000px;
                    background-image: url('{image_url}');
                    background-size: cover;
                    background-position: center;
                    border-radius: 10px;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
                }}
                .text-overlay {{
                    position: absolute;
                    width: 95%;
                    background-color: rgba(255, 255, 255, 1);
                    padding: 30px;
                    border-radius: 10px;
                    top: 15%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    text-align: center;
                }}
                .small-text {{
                    font-size: 36px;
                    font-weight: bold;
                    color: #333;
                    margin-bottom: 10px;
                    font-family: 'Calibre', Arial, sans-serif;
                }}
                .primary-text {{
                    font-size: 60px;
                    font-weight: bold;
                    color: #FF8C00;
                    font-family: 'Montserrat', sans-serif;
                    line-height: 1.2;
                    text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="text-overlay">
                    <div class="small-text">{cta_text}</div>
                    <div class="primary-text">{headline}</div>
                </div>
            </div>
        </body>
        </html>
        """
    # Template 42
    elif template == 42:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Nursing Careers in the UK</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap');
                @font-face {{
                    font-family: 'Calibre';
                    src: url('path-to-calibre-font.woff2') format('woff2');
                }}
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background-color: #F4F4F4;
                }}
                .container {{
                    position: relative;
                    width: 1000px;
                    height: 1000px;
                    background-image: url('{image_url}');
                    background-size: cover;
                    background-position: center;
                    border-radius: 10px;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
                }}
                .text-overlay {{
                    position: absolute;
                    width: 95%;
                    background-color: rgba(255, 255, 255, 1);
                    padding: 30px;
                    border-radius: 10px;
                    top: 90%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    text-align: center;
                }}
                .small-text {{
                    font-size: 36px;
                    font-weight: bold;
                    color: #333;
                    margin-bottom: 10px;
                    font-family: 'Calibre', Arial, sans-serif;
                }}
                .primary-text {{
                    font-size: 60px;
                    font-weight: bold;
                    color: #FF8C00;
                    font-family: 'Montserrat', sans-serif;
                    line-height: 1.2;
                    text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="text-overlay">
                    <div class="small-text">{cta_text}</div>
                    <div class="primary-text">{headline}</div>
                </div>
            </div>
        </body>
        </html>
        """
    # Template 6
    elif template == 6:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Nursing Careers in the UK</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap');
                @font-face {{
                    font-family: 'Calibre';
                    src: url('path-to-calibre-font.woff2') format('woff2');
                }}
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background-color: #F4F4F4;
                }}
                .container {{
                    position: relative;
                    width: 1000px;
                    height: 1000px;
                    background-image: url('{image_url}');
                    background-size: cover;
                    background-position: center;
                    border-radius: 10px;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="text-overlay">
                </div>
            </div>
        </body>
        </html>
        """
    elif template == 8:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Nursing Careers in the UK</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap');
                @font-face {{
                    font-family: 'Calibre';
                    src: url('path-to-calibre-font.woff2') format('woff2');
                }}
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background-color: #F4F4F4;
                }}
                .container {{
                    position: relative;
                    width: 999px;
                    height: 666px;
                    background-image: url('{image_url}');
                    background-size: cover;
                    background-position: center;
                    border-radius: 10px;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="text-overlay">
                </div>
            </div>
        </body>
        </html>
        """

    
    else:
        
        print('template not found')
        html_template = f"<p>Template {template} not found</p>"

        
    if template == 7:
        html_template = html_template.replace('button class="cta-button','button class="c1ta-button')
    return html_template

# NEW: Create DALLE Variation
#@log_function_call
def create_dalle_variation(image_url,count):
    """
    Downloads a Google image, converts it to PNG (resizing if needed to keep it under 4MB),
    then creates a DALL-E variation via OpenAI, returning the new image URL.
    """ 
    try:
        headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
        resp = requests.get(image_url,headers=headers)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))

        # Convert to PNG
        png_buffer = BytesIO()
        img.save(png_buffer, format="PNG")
        png_buffer.seek(0)

        # If >4MB, resize to 512x512
        if len(png_buffer.getvalue()) > 4 * 1024 * 1024:
            img = img.resize((512, 512))
            png_buffer = BytesIO()
            img.save(png_buffer, format="PNG")
            png_buffer.seek(0)

        response = client.images.create_variation(
            image=png_buffer,
            n=count,
            size="512x512"
        )
        return response.data
    except Exception as e:
        st.error(f"Error generating DALL-E variation: {e}")
        return None




# --- Page Config and Initial Playwright Check ---

if 'playwright_installed_successfully' not in st.session_state:
    with st.spinner("Checking Playwright installation..."):
        st.session_state.playwright_installed_successfully = install_playwright_browsers()


# --- Initialize Session State for Task Queues and Results ---
# Phase 1: Image Generation
if 'img_gen_task_queue' not in st.session_state: st.session_state.img_gen_task_queue = []
if 'img_gen_results_accumulator' not in st.session_state: st.session_state.img_gen_results_accumulator = [] # Stores {topic, lang, images: [img_obj, ...]}
if 'img_gen_processing_active' not in st.session_state: st.session_state.img_gen_processing_active = False
if 'img_gen_current_task_idx' not in st.session_state: st.session_state.img_gen_current_task_idx = 0
if 'img_gen_total_tasks' not in st.session_state: st.session_state.img_gen_total_tasks = 0
if 'img_gen_errors' not in st.session_state: st.session_state.img_gen_errors = []

# Phase 2: Ad Creation (HTML templates, screenshots)
if 'ad_creation_task_queue' not in st.session_state: st.session_state.ad_creation_task_queue = []
if 'ad_creation_results_list' not in st.session_state: st.session_state.ad_creation_results_list = [] # Stores final ad objects for CSV
if 'ad_creation_processing_active' not in st.session_state: st.session_state.ad_creation_processing_active = False
if 'ad_creation_current_task_idx' not in st.session_state: st.session_state.ad_creation_current_task_idx = 0
if 'ad_creation_total_tasks' not in st.session_state: st.session_state.ad_creation_total_tasks = 0
if 'ad_creation_errors' not in st.session_state: st.session_state.ad_creation_errors = []


# --- Streamlit UI ---
st.title("🎨 Creative Maker PRO 🚀 (Task Queue Version)")

# --- Expander for Template Examples ---
with st.expander("Click to see examples for templates", expanded=False):
    image_list = [
        {"image": "https://image-script.s3.us-east-1.amazonaws.com/image_1744112392_1276.png", "caption": "2"},
        # ... (add all your other example images here) ...
        {"image": "https://image-script.s3.us-east-1.amazonaws.com/image_1744716627_8304.png", "caption": "gemini7claude_simple"}
    ]
    num_example_cols = 6
    for i in range(0, len(image_list), num_example_cols):
        cols = st.columns(num_example_cols)
        for col_idx, item in enumerate(image_list[i : i + num_example_cols]):
            if item:
                with cols[col_idx]:
                    st.image(item["image"], use_container_width=True, caption=item["caption"])


st.subheader("Step 1: Define Topics for Image Generation")
with st.container(border=True):
    df_phase1_input = st.data_editor(
        pd.DataFrame({"topic": ["example_topic"],
                      "count": [2],
                      "lang": ["english"],
                      "template": [""], # Template 3 for google, 5 for product
                      "imgs_redraw":[""]}), # For gemini_redraw
        num_rows="dynamic",
        key="phase1_input_editor"
    )
    # Global flags for the entire batch of Phase 1
    is_pd_policy_global = st.checkbox("Apply PD Policy to Text Generation?", key="pd_policy_global", value=True)
    enhance_input_topic_global = st.checkbox("Enhance Input Topic (via LLM)?", key="enhance_topic_global", value=False)


# --- Phase 1: Image Generation Button & Processing Loop ---
if st.button("🚀 Start Phase 1: Generate Raw Images (Queued)", type="primary", use_container_width=True,
             disabled=st.session_state.get('img_gen_processing_active') or st.session_state.get('ad_creation_processing_active')):
    st.session_state.img_gen_task_queue = []
    st.session_state.img_gen_results_accumulator = []
    st.session_state.img_gen_errors = []
    st.session_state.img_gen_current_task_idx = 0
    st.session_state.img_gen_total_tasks = 0

    temp_tasks = []
    for idx, row in df_phase1_input.iterrows():
        topic = row['topic']
        count = int(row['count'])
        lang = row['lang']
        template_col_str = row["template"]
        redraw_sources = row["imgs_redraw"]

        for i in range(count):
            templates_available = [t.strip() for t in template_col_str.split(',') if t.strip()]
            if not templates_available:
                st.warning(f"No template specified for topic '{topic}', instance {i+1}. Skipping.")
                continue
            
            chosen_template = random.choice(templates_available) if len(templates_available) > 1 else templates_available[0]
            
            task = {
                "original_topic": topic, "current_topic": topic, "lang": lang,
                "chosen_template": chosen_template, "original_template_str": template_col_str,
                "redraw_sources": redraw_sources, "instance_num": i + 1,
                "unique_id": f"p1_{idx}_{i}_{''.join(random.choices(string.ascii_letters+string.digits , k=5))}", # For potential unique keys later
                "topic_enhanced_this_task": False # Flag to ensure topic is enhanced only once per task if needed
            }
            temp_tasks.append(task)

    st.session_state.img_gen_task_queue = temp_tasks
    st.session_state.img_gen_total_tasks = len(temp_tasks)
    if temp_tasks:
        st.session_state.img_gen_processing_active = True
        st.info(f"Queued {len(temp_tasks)} image generation tasks for Phase 1.")
        st.rerun()
    else:
        st.info("No tasks to queue for Phase 1.")


# Phase 1 Processing Loop
if st.session_state.get('img_gen_processing_active') and st.session_state.img_gen_task_queue:
    task_to_process = st.session_state.img_gen_task_queue[0]
    current_idx = st.session_state.img_gen_current_task_idx
    total_tasks = st.session_state.img_gen_total_tasks

    st.info(f"Phase 1: Processing task {current_idx + 1}/{total_tasks} - Topic: '{task_to_process['original_topic']}', Template: '{task_to_process['chosen_template']}'")
    st.progress((current_idx) / total_tasks if total_tasks > 0 else 0)

    try:
        # --- Process this single image generation task ---
        topic_for_api = task_to_process['current_topic']
        lang = task_to_process['lang']
        template = task_to_process['chosen_template']
        # Global flags (set by checkboxes)
        apply_pd_policy = is_pd_policy_global
        enhance_this_topic = enhance_input_topic_global

        # Optional: Enhance topic once per task if flag is set globally and not yet done for this task
        if enhance_this_topic and not task_to_process.get('topic_enhanced_this_task'):
            st.write(f"Enhancing topic: {topic_for_api}...")
            enhanced_topic = chatGPT(f"Rephrase for commercial appeal: '{topic_for_api}'", model="gpt-3.5-turbo") # Use appropriate model
            if enhanced_topic:
                topic_for_api = enhanced_topic
                task_to_process['current_topic'] = enhanced_topic # Update for this task only
            task_to_process['topic_enhanced_this_task'] = True


        img_url_result = None
        img_source_result = "unknown"

        if "google" in template.lower() or "google" in topic_for_api.lower(): # Simplified check
            clean_topic_for_google = topic_for_api.replace('google', '').strip()
            if '|' in clean_topic_for_google: # From original script
                clean_topic_for_google = re.sub("^.*\|", "", clean_topic_for_google)
            
            google_img_urls = fetch_google_images(clean_topic_for_google, num_images=1)
            if google_img_urls:
                img_url_result = google_img_urls[0]
                img_source_result = "google"
            else:
                raise ValueError(f"Google image fetch failed for '{clean_topic_for_google}'.")

        elif "gemini" in template.lower():
            # Complex Gemini prompt logic from original script
            gemini_api_prompt = f"Prompt for Gemini: Topic '{topic_for_api}', Lang '{lang}', Template '{template}'" # Placeholder
            if template == 'gemini2':
                gemini_api_prompt = chatGPT(f"Short square image prompt for '{topic_for_api}' ({lang}) with CTA 'Learn More Here >>'. Start with 'square image 1:1 of'.", model="gpt-4o")
            elif template == 'gemini7claude':
                 gemini_api_prompt = claude(f"Short square image prompt for '{topic_for_api}' ({lang}) with CTA 'Learn More Here >>', low quality, enticing. Start with 'square image 1:1 of'. No specific promises.")
            elif template == 'geminicandid':
                 gemini_api_prompt = claude(f"Image prompt: candid smartphone photo of regular person with '{topic_for_api}'. Organic, perplexing, high energy. No text overlay. Start with 'Square photo 1:1 iphone 12 photo uploaded to reddit:'.", is_thinking_enabled=True)
            elif template == 'gemini_redraw':
                redraw_img_url_list = [url.strip() for url in task_to_process['redraw_sources'].split('|') if url.strip()]
                if not redraw_img_url_list: raise ValueError("Redraw images list is empty for gemini_redraw.")
                chosen_redraw_url = random.choice(redraw_img_url_list)
                redraw_prompt = f"Describe this image visually in detail for regeneration: {chosen_redraw_url}. Start 'square image 1:1 of'. Include text overlays in original language and CTA."
                gemini_api_prompt = gemini_text_lib(redraw_prompt, model_name="gemini-1.5-pro-latest", is_with_file=True, file_url=chosen_redraw_url) # model supporting vision
            # ... Add all other Gemini template conditions from your script ...
            else: # Default Gemini prompt
                gemini_api_prompt = chatGPT(f"Default square image prompt for '{topic_for_api}' ({lang}) with CTA 'Learn More Here >>', low quality, enticing. Start with 'square image 1:1 of'.", model="gpt-4o", temperature=1.0)

            if not gemini_api_prompt: raise ValueError("Failed to generate Gemini API prompt.")
            st.write(f"Gemini API Prompt: {gemini_api_prompt}")
            pil_image = gen_gemini_image(gemini_api_prompt)
            if pil_image:
                img_url_result = upload_pil_image_to_s3(pil_image, S3_BUCKET_NAME_SECRET, AWS_ACCESS_KEY_ID_SECRET, AWS_SECRET_ACCESS_KEY_SECRET, region_name=AWS_REGION_SECRET)
                img_source_result = template # Use the specific gemini template name as source
            else:
                raise ValueError("Gemini image generation or S3 upload failed.")
        
        # For templates that are numbers (Flux image + HTML template)
        # This assumes raw Flux/Google images are NOT the final product for these templates.
        # If a template like "3" means "use Google image in HTML template 3",
        # then the `img_url_result` from a "google" task above would be passed to Phase 2.
        # For simplicity in Phase 1, we'll assume numeric templates mean "Flux image that will LATER be used in HTML".
        # So, Phase 1 for numeric templates only generates the Flux URL.
        elif template.isdigit() or template in ["geminicandid", "geministock"]: # These might be base images for HTML templates OR direct use
            flux_topic_prompt = topic_for_api # Default
            if template == "geminicandid": # Flux LORA
                flux_topic_prompt = claude(f"Image prompt of a candid unstaged photo of a regular joe showing off his/her {topic_for_api}. Smartphone quality, reddit style, perplexing, high energy. NO TEXT OVERLAY.", is_thinking_enabled=True)
                if flux_topic_prompt:
                     img_url_result = gen_flux_img_lora(flux_topic_prompt)
                else: raise ValueError("Failed to generate prompt for flux lora.")

            elif template == "geministock": # Regular Flux for stock-like
                flux_topic_prompt = chatGPT(f"Short image prompt for {topic_for_api}, no text on image. High-quality, realistic, well-lit, marketing/editorial style.", model="gpt-4o")
                if flux_topic_prompt:
                    img_url_result = gen_flux_img(flux_topic_prompt)
                else: raise ValueError("Failed to generate prompt for flux stock.")
            
            # If it's just a number, it implies a Flux image for a later HTML template.
            # The actual HTML templating happens in Phase 2. So Phase 1 just gets the Flux image URL.
            elif template.isdigit():
                # Default Flux image prompt if not candid or stock
                flux_topic_prompt = chatGPT(f"Visually enticing image description (15 words max) for {topic_for_api}.", model="gpt-4o")
                if flux_topic_prompt:
                    img_url_result = gen_flux_img(flux_topic_prompt)
                else: raise ValueError("Failed to generate prompt for flux (numeric template).")

            if img_url_result == "nsfw_detected":
                st.error(f"NSFW content detected by Flux for topic '{topic_for_api}'. Task aborted.")
                img_url_result = None # Ensure it's None so error is caught
                raise ValueError("NSFW detected")
            elif not img_url_result:
                raise ValueError("Flux image generation failed.")
            img_source_result = f"flux_for_template_{template}"


        # --- Accumulate result for Phase 1 ---
        if img_url_result:
            # Find or create group for this topic+lang
            group = next((g for g in st.session_state.img_gen_results_accumulator if g['topic'] == task_to_process['original_topic'] and g['lang'] == lang), None)
            if not group:
                group = {"topic": task_to_process['original_topic'], "lang": lang, "images": []}
                st.session_state.img_gen_results_accumulator.append(group)
            
            group['images'].append({
                'url': img_url_result,
                'template_used': template, # The specific template chosen for this instance
                'source_type': img_source_result,
                'original_topic': task_to_process['original_topic'], # Keep for reference
                'processed_topic': topic_for_api, # Potentially enhanced topic
                'lang': lang,
                'selected_count': 0, # For Phase 2 selection
                'dalle_variation_requested': False
            })
        else:
            raise ValueError("Image URL was not generated for the task.")

        # Task successful
        st.session_state.img_gen_task_queue.pop(0)
        st.session_state.img_gen_current_task_idx += 1

    except Exception as e:
        st.error(f"Error in Phase 1 Task: {task_to_process['original_topic']} (Tmpl: {template}): {str(e)}")
        failed_task = st.session_state.img_gen_task_queue.pop(0)
        st.session_state.img_gen_errors.append({"task": failed_task, "error": str(e)})
        st.session_state.img_gen_current_task_idx += 1 # Count as processed attempt

    # Rerun for next task or to finish Phase 1
    if not st.session_state.img_gen_task_queue: # If queue is now empty
        st.session_state.img_gen_processing_active = False
        st.success(f"🎉 Phase 1: All {total_tasks} image generation tasks processed!")
        if total_tasks > 0: play_sound("audio/bonus-points-190035.mp3") # Ensure path is correct
        # No rerun here, let UI update below for results display
    else:
        st.rerun() # Continue to next task in Phase 1

# --- Display Phase 1 Results & Allow Selection for Phase 2 ---
if not st.session_state.get('img_gen_processing_active') and st.session_state.img_gen_results_accumulator:
    st.subheader("✅ Phase 1 Complete: Review Generated Images")
    st.caption("Set 'Ad Count' for images you want to process into final HTML ad creatives in Phase 2.")
    
    if st.session_state.img_gen_errors:
        with st.expander("Show Phase 1 Errors", expanded=False):
            for err in st.session_state.img_gen_errors:
                st.error(f"Failed Task: {err['task']['original_topic']} (Template: {err['task']['chosen_template']}) - Error: {err['error']}")

    auto_select_all_phase2 = st.checkbox("Auto-set Ad Count to 1 for all Phase 1 images?", key="auto_select_phase2")

    for group_idx, result_group in enumerate(st.session_state.img_gen_results_accumulator):
        with st.container(border=True):
            st.markdown(f"#### Topic: {result_group['topic']} (Language: {result_group['lang']})")
            images_in_group = result_group['images']
            cols_per_row = st.slider("Images per row (Phase 1 Display)", 2, 8, 4, key=f"cols_p1_{group_idx}")
            
            for i in range(0, len(images_in_group), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx, img_obj in enumerate(images_in_group[i : i + cols_per_row]):
                    if col_idx < len(cols): # Ensure we don't go out of bounds for columns
                        with cols[col_idx]:
                            img_unique_key = f"p1_img_{group_idx}_{i+col_idx}"
                            st.image(img_obj['url'], caption=f"Source: {img_obj['source_type']}, Tmpl: {img_obj['template_used']}", use_container_width=True)
                            
                            default_ad_count = 1 if auto_select_all_phase2 else img_obj.get('selected_count', 0)
                            ad_count = st.number_input("Ad Count (for Phase 2)", min_value=0, max_value=5, value=default_ad_count, key=f"ad_count_{img_unique_key}")
                            img_obj['selected_count'] = ad_count # Update the object in session state

                            # DALL-E Variation (simplified, runs synchronously here, could be a separate task system too)
                            if img_obj.get("source_type") == "google" and not img_obj.get("dalle_variation_requested"):
                                if st.button("Request DALL-E Variations", key=f"dalle_{img_unique_key}", help="Generates DALL-E variations for this Google image based on 'Ad Count'. Adds them to this list."):
                                    if ad_count > 0 and oai_client:
                                        with st.spinner(f"Generating {ad_count} DALL-E variations..."):
                                            variation_data_list = create_dalle_variation(img_obj['url'], ad_count)
                                            if variation_data_list:
                                                img_obj["dalle_variation_requested"] = True
                                                for var_idx, var_data in enumerate(variation_data_list):
                                                    images_in_group.append({ # Add to current display group
                                                        'url': var_data.url, 'template_used': img_obj['template_used'],
                                                        'source_type': f"dalle_variation_{var_idx+1}",
                                                        'original_topic': img_obj['original_topic'], 'processed_topic': img_obj['processed_topic'],
                                                        'lang': img_obj['lang'], 'selected_count': 1, # Default selected
                                                        'dalle_variation_requested': True
                                                    })
                                                st.success(f"Added {len(variation_data_list)} DALL-E variations.")
                                                st.rerun() # Refresh display
                                    elif not oai_client:
                                        st.error("OpenAI client for DALL-E not configured.")
                                    else:
                                        st.warning("Set 'Ad Count > 0' to generate DALL-E variations.")
    st.markdown("---")


# --- Phase 2: Ad Creation (HTML Templating) Button & Processing Loop ---
st.subheader("Step 2: Create Final Ad Creatives from Selected Images")
if st.button("🚀 Start Phase 2: Create HTML Ad Creatives (Queued)", type="primary", use_container_width=True,
             disabled=st.session_state.get('img_gen_processing_active') or st.session_state.get('ad_creation_processing_active') or not st.session_state.img_gen_results_accumulator):
    
    st.session_state.ad_creation_task_queue = []
    st.session_state.ad_creation_results_list = [] # This will store the final {Topic, Language, Image_X_Y} dicts
    st.session_state.ad_creation_errors = []
    st.session_state.ad_creation_current_task_idx = 0
    st.session_state.ad_creation_total_tasks = 0

    temp_ad_tasks = []
    for result_group in st.session_state.img_gen_results_accumulator:
        for img_obj in result_group['images']:
            if img_obj.get('selected_count', 0) > 0:
                # Check if this image type is suitable for HTML templating
                # Gemini images are usually final. Google/Flux images are bases for HTML.
                # Dalle variations are also usually final.
                # The original script implied Gemini images were just added to the list.
                # Numeric templates (1-8 excluding google/gemini ones) were for HTML.
                
                is_html_template_task = False
                template_for_html_str = img_obj['template_used']

                if template_for_html_str.isdigit(): # Templates 1-8 are HTML based
                    is_html_template_task = True
                elif img_obj['source_type'] == "google" and not template_for_html_str.lower().startswith("google"):
                    # If it's a google image BUT the template_used is a numeric HTML template
                    # (e.g., user input "google_image_search,3" -> template_used becomes "3" for the google image)
                    if template_for_html_str.isdigit():
                         is_html_template_task = True


                if is_html_template_task:
                    for i in range(img_obj['selected_count']):
                        ad_task = {
                            "base_image_url": img_obj['url'],
                            "original_topic": img_obj['original_topic'],
                            "processed_topic": img_obj['processed_topic'],
                            "lang": img_obj['lang'],
                            "html_template_id_str": template_for_html_str, # The numeric template ID as string
                            "instance_num": i + 1,
                            "unique_id": f"p2_{img_obj['url'][-10:]}_{i}_{''.join(random.choices(string.ascii_letters+string.digits , k=5))}"
                        }
                        temp_ad_tasks.append(ad_task)
                else: # Direct use image (Gemini, DALL-E variation, or Google if template was 'google_image_search')
                    # Add these directly to the ad_creation_results_list for the final CSV
                    # as they don't need further HTML processing.
                    for i in range(img_obj['selected_count']):
                         # This structure needs to match what shift_left_and_pad expects later.
                         # Original `final_results` was a list of dicts, where each dict is a row.
                         # We need to build these row-like dicts.
                         # We'll do this after Phase 2 processing completes. For now, just note them.
                        st.session_state.ad_creation_results_list.append({
                            # This structure needs to be transformed for the final CSV.
                            # Let's store them as "pre_csv_objects"
                            "type": "direct_image",
                            "Topic": img_obj['original_topic'],
                            "Language": img_obj['lang'],
                            "S3_URL": img_obj['url'], # This is the direct S3 URL
                            "Source_Template_Phase1" : img_obj['template_used']
                        })


    st.session_state.ad_creation_task_queue = temp_ad_tasks
    st.session_state.ad_creation_total_tasks = len(temp_ad_tasks)
    if temp_ad_tasks:
        st.session_state.ad_creation_processing_active = True
        st.info(f"Queued {len(temp_ad_tasks)} HTML ad creation tasks for Phase 2.")
        st.rerun()
    elif st.session_state.ad_creation_results_list: # Only direct images were selected
        st.session_state.ad_creation_processing_active = False # No HTML tasks
        st.success("Phase 2: No HTML ads to create. Direct images collected.")
        st.rerun() # To trigger final CSV display
    else:
        st.info("No images selected or suitable for HTML ad creation in Phase 2.")


# Phase 2 Ad Creation Processing Loop
if st.session_state.get('ad_creation_processing_active') and st.session_state.ad_creation_task_queue:
    ad_task_to_process = st.session_state.ad_creation_task_queue[0]
    current_ad_idx = st.session_state.ad_creation_current_task_idx
    total_ad_tasks = st.session_state.ad_creation_total_tasks

    st.info(f"Phase 2: Processing Ad Task {current_ad_idx + 1}/{total_ad_tasks} - Topic: '{ad_task_to_process['original_topic']}', HTML Template: '{ad_task_to_process['html_template_id_str']}'")
    st.progress((current_ad_idx) / total_ad_tasks if total_ad_tasks > 0 else 0)

    try:
        base_img_url = ad_task_to_process['base_image_url']
        topic = ad_task_to_process['processed_topic'] # Use the (potentially enhanced) topic
        lang = ad_task_to_process['lang']
        html_template_id = int(ad_task_to_process['html_template_id_str']) # Convert to int for save_html

        # Global flags
        apply_pd_policy = is_pd_policy_global # From checkbox

        # --- Generate text components for HTML (from original "Process Selected Images") ---
        headline_text_for_ad, cta_text_for_ad, tag_line_for_ad = "", "Learn More", "" # Defaults

        # This logic was quite specific in your original script. Replicating main branches.
        clean_topic_for_text = re.sub(r'\|.*','', ad_task_to_process['original_topic']).strip() # Use original topic for text gen

        if html_template_id in [1, 2]:
            headline_prompt = f"Short text (max 20 words) to promote article about {clean_topic_for_text} ({lang}). Concise, compelling."
            headline_text_for_ad = chatGPT(headline_prompt, model="gpt-4o")
        elif html_template_id == 3: # also for 7 if styling is same
            headline_prompt = f"1 statement, no quotes, for {clean_topic_for_text} ({lang}). Examples: 'Surprising Travel Perks...'. Don't use 'Hidden' or 'Unlock'. Max 6 words."
            headline_text_for_ad = chatGPT(headline_prompt)
        elif html_template_id == 5:
            headline_prompt = f"1 statement, ALL CAPS, for {clean_topic_for_text} ({lang}). Wrap 1-2 urgent words in <span class='highlight'>...</span>. Max 60 chars. Drive curiosity."
            headline_text_for_ad = chatGPT(headline_prompt, model="gpt-4o")
            tag_line_prompt = f"Short tagline for {clean_topic_for_text} ({lang}), max 25 chars, ALL CAPS, emoji ok. Don't mention topic. Drive action."
            tag_line_for_ad = chatGPT(tag_line_prompt, model="gpt-4o").strip('"').strip("'").strip("!")
            if headline_text_for_ad: headline_text_for_ad = headline_text_for_ad.replace(r"</span>", r"</span> ") # Formatting fix
        elif html_template_id == 7: # Similar to 3 but punchier?
            headline_prompt = f"Short punchy 1 sentence for article on {clean_topic_for_text} ({lang}). Casual, sharp, concise, ill-tempered language. No 'you'. Max 70 CHARS, Title Case. No dark themes."
            headline_text_for_ad = chatGPT(headline_prompt, model="gpt-4o")
        elif html_template_id in [4, 41, 42]: # Topic is headline, CTA is "Read more about"
            headline_text_for_ad = topic # Use the processed topic
            cta_text_for_ad = chatGPT(f"Translate 'Read more about' to {lang}.", model="gpt-4o").replace('"','')
        elif html_template_id in [6, 8]: # Image only, no text generated for overlay
            headline_text_for_ad = ""
            cta_text_for_ad = ""
        else: # Default headline
            headline_text_for_ad = chatGPT(f"Concise headline for {clean_topic_for_text} ({lang}), no quotes.", model="gpt-4o")
        
        # Default CTA if not set by template-specific logic
        if not cta_text_for_ad and html_template_id not in [6,8]: # Templates 6,8 have no cta
            cta_text_for_ad = chatGPT(f"Translate 'Learn More' to {lang}.", model="gpt-4o").replace('"','')

        if not headline_text_for_ad and html_template_id not in [6,8]: headline_text_for_ad = clean_topic_for_text # Fallback headline

        # --- Create HTML, Screenshot, Upload ---
        html_ad_content = save_html(headline_text_for_ad, base_img_url, cta_text_for_ad, html_template_id, tag_line_for_ad)
        
        screenshot_width, screenshot_height = 1000, 1000
        if html_template_id == 8: screenshot_width, screenshot_height = 999, 666 # Specific dimensions

        screenshot_img_bytes = capture_html_screenshot_playwright(html_ad_content, width=screenshot_width, height=screenshot_height)
        if not screenshot_img_bytes:
            raise ValueError("Failed to capture HTML screenshot.")

        final_ad_s3_url = upload_pil_image_to_s3(
            screenshot_img_bytes, S3_BUCKET_NAME_SECRET, AWS_ACCESS_KEY_ID_SECRET, AWS_SECRET_ACCESS_KEY_SECRET,
            region_name=AWS_REGION_SECRET, image_format='PNG' # Screenshots are PNG
        )
        if not final_ad_s3_url:
            raise ValueError("Failed to upload final ad screenshot to S3.")

        # Add to results list (for CSV generation)
        st.session_state.ad_creation_results_list.append({
            "type": "html_ad",
            "Topic": ad_task_to_process['original_topic'],
            "Language": lang,
            "S3_URL": final_ad_s3_url,
            "Source_Template_Phase1": ad_task_to_process['html_template_id_str'], # Or could store original template from phase 1 if different
            "HTML_Template_Used": html_template_id
        })
        
        # Task successful
        st.session_state.ad_creation_task_queue.pop(0)
        st.session_state.ad_creation_current_task_idx += 1

    except Exception as e:
        st.error(f"Error in Phase 2 Ad Creation Task: {ad_task_to_process['original_topic']} (HTML Tmpl: {ad_task_to_process['html_template_id_str']}): {str(e)}")
        failed_ad_task = st.session_state.ad_creation_task_queue.pop(0)
        st.session_state.ad_creation_errors.append({"task": failed_ad_task, "error": str(e)})
        st.session_state.ad_creation_current_task_idx += 1
    
    # Rerun for next ad task or to finish Phase 2
    if not st.session_state.ad_creation_task_queue:
        st.session_state.ad_creation_processing_active = False
        st.success(f"🎉 Phase 2: All {total_ad_tasks} ad creation tasks processed!")
        if total_ad_tasks > 0 or st.session_state.ad_creation_results_list : play_sound("audio/bonus-points-190035.mp3")
        # No rerun, let UI update below for final results display
    else:
        st.rerun()


# --- Display Final Results Table & CSV Download ---
if not st.session_state.get('img_gen_processing_active') and \
   not st.session_state.get('ad_creation_processing_active') and \
   st.session_state.ad_creation_results_list: # Check if there's anything to display

    st.subheader("✅ All Processing Complete: Final Ad Creatives")
    if st.session_state.ad_creation_errors:
        with st.expander("Show Phase 2 Errors", expanded=False):
            for err in st.session_state.ad_creation_errors:
                st.error(f"Failed Ad Task: {err['task']['original_topic']} (HTML Tmpl: {err['task']['html_template_id_str']}) - Error: {err['error']}")

    # Prepare data for the final DataFrame (as in your original script)
    final_df_rows = []
    # Group results by Topic and Language for the CSV structure
    grouped_for_csv = {}
    for item in st.session_state.ad_creation_results_list:
        key = (item['Topic'], item['Language'])
        if key not in grouped_for_csv:
            grouped_for_csv[key] = {'Topic': item['Topic'], 'Language': item['Language'], 'images': []}
        grouped_for_csv[key]['images'].append(item['S3_URL'])
    
    max_images_per_row = 0
    for data in grouped_for_csv.values():
        final_df_rows.append(data) # Keep the structure with 'images' as a list for now
        if len(data['images']) > max_images_per_row:
            max_images_per_row = len(data['images'])

    # Create columns like Image_1, Image_2, ...
    GLOBAL_IMAGE_COLS_FOR_DATAFRAME.clear() # Clear before repopulating
    for i in range(max_images_per_row):
        GLOBAL_IMAGE_COLS_FOR_DATAFRAME.append(f"Image_{i+1}")

    # Now create the DataFrame and apply shift_left_and_pad
    output_data_for_df = []
    for row_data_with_list in final_df_rows:
        new_row = {'Topic': row_data_with_list['Topic'], 'Language': row_data_with_list['Language']}
        for i in range(max_images_per_row):
            col_name = GLOBAL_IMAGE_COLS_FOR_DATAFRAME[i]
            new_row[col_name] = row_data_with_list['images'][i] if i < len(row_data_with_list['images']) else ''
        output_data_for_df.append(new_row)
    
    if output_data_for_df:
        output_df = pd.DataFrame(output_data_for_df)
        # The shift_left_and_pad was more for when columns were not fixed.
        # Here, columns are fixed. If you need to ensure no empty cells in between,
        # you might re-apply a modified shift_left_and_pad or ensure the image list is dense.
        # For now, this structure should be fine.
        # If you need to apply it:
        # image_data_cols = [col for col in output_df.columns if col.startswith("Image_")]
        # if image_data_cols: # Check if any image columns exist
        # GLOBAL_IMAGE_COLS_FOR_DATAFRAME = image_data_cols # Update global for the function
        # output_df[image_data_cols] = output_df[image_data_cols].apply(shift_left_and_pad, axis=1)

        st.dataframe(output_df)
        csv_data = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Final Results as CSV",
            data=csv_data,
            file_name='creative_maker_final_ads.csv',
            mime='text/csv',
            use_container_width=True
        )
    else:
        st.info("No final ad creatives were generated or collected for the CSV output.")
