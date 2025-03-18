import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
import boto3
from botocore.exceptions import NoCredentialsError
import random
import string
import requests
import os
import time
from playwright.sync_api import sync_playwright
from tempfile import NamedTemporaryFile
import re
from google_images_search import GoogleImagesSearch
import openai  # NEW: For DALL-E variations
import logging
from openai import OpenAI
# Configure logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s', 
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# Set your OpenAI key for DALL-E
openai.api_key = st.secrets.get("OPENAI_API_KEY")

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

@log_function_call
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

@log_function_call
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

# Set up page config
st.set_page_config(layout="wide")

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

client = OpenAI(api_key=GPT_API_KEY)

@log_function_call
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

@log_function_call
def chatGPT(prompt, model="gpt-4o", temperature=1.0):
    """
    Call OpenAI's Chat Completion (GPT) to generate text.
    """
    st.write("Generating image description...")
    headers = {
        'Authorization': f'Bearer {GPT_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': model,
        'temperature': temperature,
        'messages': [{'role': 'user', 'content': prompt}]
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    content = response.json()['choices'][0]['message']['content'].strip()
    return content

@log_function_call
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

@log_function_call
def capture_html_screenshot_playwright(html_content):
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
            page = browser.new_page(viewport={'width': 1000, 'height': 1000})

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

@log_function_call
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
    
    if template == 7:
        html_template = html_template.replace('button class="cta-button','button class="c1ta-button')
    else:
        
        print('template not found')
        html_template = f"<p>Template {template} not found</p>"

    return html_template

# NEW: Create DALLE Variation
@log_function_call
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

# --------------------------------------------
# Streamlit UI
# --------------------------------------------
st.title("AI Image Generation and Upload App")

# Initialize session state for storing generated images
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = {}

st.subheader("Enter Topics for Image Generation")
df = st.data_editor(
    pd.DataFrame({"topic": ["example_topic"], "count": [1], "lang": ["english"], "template": ["1,2,3,4,41,42,5,6 use , for multi"]}),
    num_rows="dynamic",
    key="table_input"
)

# Step 1: Generate Images
if st.button("Generate Images"):
    st.session_state.generated_images = []  # Clear previous images
    processed_combinations = set()

    for _, row in df.iterrows():
        topic = row['topic']
        count = int(row['count'])
        lang = row['lang']
        combo = f"{topic}_{lang}"
        template_str = row["template"]

        if combo not in processed_combinations:
            processed_combinations.add(combo)
            st.subheader(f"Generating images for: {topic}")
            topic_images = []
            temp_topic = topic

        if "google" in topic.lower():
            # If "google" is in the topic, fetch from Google
            topic = topic.replace('google', ' ')
            if '|' in topic:
                topic_for_google = re.sub("^.*\|", "", topic)
                st.markdown(topic_for_google)
            else:
                topic_for_google = topic

            google_image_urls = fetch_google_images(topic_for_google, num_images=int(count))
            for img_url in google_image_urls:
                topic_images.append({
                    'url': img_url,
                    'selected': False,
                    'template': random.choice([int(x) for x in template_str.split(",")]),
                    'source': 'google',       # Mark as Google
                    'dalle_generated': False  # For tracking DALL-E generation
                })

        else:
            # Otherwise, use FLUX to generate
            for i in range(count):
                topic = temp_topic
                if '^' in topic:
                    topic = random.choice(topic.split("^"))

                new_prompt = False
                if "," in template_str:
                    template = random.choice([int(x) for x in template_str.split(",")])
                elif "*" in template_str:
                    new_prompt = random.choice([True, False])
                    template_str = template_str.replace("*", "")
                    template = int(template_str)
                else:
                    template = int(template_str)

                with st.spinner(f"Generating image {i + 1} for '{topic}'..."):
                    if template == 5:
                        rand_prompt = f"""Generate a concise visual image description (15 words MAX) for {topic}.
                        Be wildly creative, curious, and push the limits of imagination—while staying grounded in real-life scenarios!
                        Depict an everyday, highly relatable yet dramatically eye-catching scene that sparks immediate curiosity within 3 seconds.
                        Ensure the image conveys the value of early detection (e.g., saving money, time, improving health, or education) in a sensational but simple way.
                        The scene must feature one person, clearly illustrating the topic without confusion.
                        Avoid surreal or abstract elements; instead, focus on relatable yet RANDOM high-energy moments from daily life.
                        Do not include any text in the image.
                        Your final output should be 8-13 words, written as if describing a snapshot from a camera.
                        Make sure the offer’s value is unmistakably clear and visually intriguing"""
                        image_prompt = chatGPT(rand_prompt, model='gpt-4', temperature=1.2)
                        st.markdown(image_prompt)

                    elif not new_prompt:
                        image_prompt = chatGPT(
                            f"""Generate a  visual image description  15 words MAX for  {topic}.
                            Be creative, show the value of the offer (saving money, time, health, etc.) in a sensational yet simplistic scene.
                            Include one person and do not include text in the image. 
                            Output is up to 5 words. Think like a camera snapshot!""",
                            model='gpt-4',
                            temperature=1.15
                        )
                    else:
                        image_prompt = chatGPT(
                            f"""Generate a  visual image description 15 words MAX for {topic}.
                            Use a visually enticing style with high CTR, avoid obvious descriptions.""",
                            model='o1-mini'
                        )

                    # Generate with FLUX
                    if template == 5:
                        image_url = gen_flux_img(
                            f"{random.choice(['cartoony clipart of ', 'cartoony clipart of ', '', ''])}{image_prompt}",
                            width=688,
                            height=416
                        )
                    if template == 7:
                        image_url = gen_flux_img(
                            f"{image_prompt}",
                            
                        ) 
                    else:
                        image_url = gen_flux_img(
                            f"{random.choice(['cartoony clipart of ', 'cartoony clipart of ', '', ''])}{image_prompt}"
                        )

                    if image_url:
                        topic_images.append({
                            'url': image_url,
                            'selected': False,
                            'template': template,
                            'source': 'flux',            # Mark as flux
                            'dalle_generated': False     # Not relevant for flux, but keep structure
                        })

        # Append the images for this topic
        st.session_state.generated_images.append({
            "topic": topic,
            "lang": lang,
            "images": topic_images
        })

# Step 2: Display generated images in a grid
if st.session_state.generated_images:
    st.subheader("Select Images to Process")
    zoom = st.slider("Zoom Level", min_value=50, max_value=500, value=300, step=50)

    for entry in st.session_state.generated_images:
        topic = entry["topic"]
        lang = entry["lang"]
        images = entry["images"]

        st.write(f"### {topic} ({lang})")

        num_columns = 6
        rows = (len(images) + num_columns - 1) // num_columns

        for row in range(rows):
            cols = st.columns(num_columns)
            for col, img in zip(cols, images[row * num_columns:(row + 1) * num_columns]):
                with col:
                    st.image(img['url'], width=zoom)
                    unique_key = f"num_select_{topic}_{lang}_{img['url']}"
                    img['selected_count'] = st.number_input(
                        f"Count for {img['url'][-5:]}",
                        min_value=0, max_value=10, value=0, key=unique_key
                    )

                    # DALL-E Variation button for Google images
                    if img.get("source") == "google" and not img.get("dalle_generated", False):
                        if st.button("Get DALL-E Variation", key=f"dalle_button_{topic}_{img['url']}"):
                            dalle_url = create_dalle_variation(img['url'],img.get("selected_count"))
                            if dalle_url:
                                for dalle_img in dalle_url:
                                        
                                    st.success("DALL-E variation generated!")
                                    img["dalle_generated"] = True
                                    # Append the new DALL-E image
                                    entry["images"].append({
                                        "url": dalle_img.url,
                                        "selected": False,
                                        "template": img["template"],
                                        "source": "dalle",
                                        "dalle_generated": True
                                    })
                                    # st.experimental_rerun()

# Step 3: Process selected images -> generate HTML, screenshot, upload to S3
if st.button("Process Selected Images"):
    final_results = []

    for entry in st.session_state.generated_images:
        topic = entry["topic"]
        lang = entry["lang"]
        images = entry["images"]

        res = {'Topic': topic, 'Language': lang}
        selected_images = [img for img in images if img['selected_count'] > 0]

        # We'll store CTA text per language in a dict to avoid repeated calls
        cta_texts = {}

        for idx, img in enumerate(selected_images):
            for i in range(img['selected_count']):
                template = img['template']

                # Decide which prompt to use for headline
                if template == 1 or template == 2:
                    headline_prompt = (
                        f"write a short text (up to 20 words) to promote an article about {topic} in {lang}. "
                        f"Goal: be concise yet compelling to click."
                    )
                elif template in [3]:
                    headline_prompt = (
                        f"write 1 statement, same length, no quotes, for {re.sub('\\|.*','',topic)} in {lang}."
                        f"Examples:\n'Surprising Travel Perks You Might Be Missing'\n"
                        f"'Little-Known Tax Tricks to Save Big'\n"
                        f"Dont mention 'Hidden' or 'Unlock'."
                    )
                elif template in [5]:
                    headline_prompt = (
                        f"write 1 statement, same length, no quotes, for {re.sub('\\|.*','',topic)} in {lang}. "
                        f"ALL IN CAPS. wrap the 1-2 most urgent words in <span class='highlight'>...</span>."
                        f"Make it under 60 chars total, to drive curiosity."
                    )
                elif template in [7]:
                    headline_prompt = (f"write short punchy 1 sentence text to   this article: \n casual and sharp and consice\nuse ill-tempered language\n don't address the reader (don't use 'you' and etc)\n informal casual friends texting style and tone\nAvoid dark themes like drugs, death etc..\n MAX 70 CHARS in land {lang} for: {re.sub('\\|.*','',topic)}")
                        

                else:
                    headline_prompt = f"Write a concise headline for {topic} in {lang}, no quotes."

                if lang in cta_texts:
                    cta_text = cta_texts[lang]
                else:
                    # e.g. "Learn More" in that language
                    cta_trans = chatGPT(
                        f"Return EXACTLY the text 'Learn More' in {lang} (no quotes)."
                    ).replace('"', '')
                    cta_texts[lang] = cta_trans
                    cta_text = cta_trans

                # For certain templates, override cta_text or headline
                if template in [4, 41, 42]:
                    headline_text = topic
                    cta_text = chatGPT(
                        f"Return EXACTLY 'Read more about' in {lang} (no quotes)."
                    ).replace('"', '')
                elif template == 6:
                    headline_text = ''
                else:
                    # Generate the main headline with GPT
                    headline_text = chatGPT(
                        prompt=headline_prompt,
                        model='gpt-4'
                    ).strip('"').strip("'")

                    st.markdown(headline_text)

                # If template=5, generate a "tag_line"
                if template == 5:
                    tag_line = chatGPT(
                        f"Write a short tagline for {re.sub('\\|.*','',topic)} in {lang}, "
                        f"to drive action, max 25 chars, ALL CAPS, possibly with emoji. "
                        f"Do NOT mention the topic explicitly."
                    ).strip('"').strip("'").strip("!")
                    # Minor formatting fix to keep <span> spacing
                    headline_text = headline_text.replace(r"</span>", r"</span>   ")
                else:
                    tag_line = ''

                # Build final HTML
                html_content = save_html(
                    headline=headline_text,
                    image_url=img['url'],
                    cta_text=cta_text,
                    template=template,
                    tag_line=tag_line
                )

                # Capture screenshot
                screenshot_image = capture_html_screenshot_playwright(html_content)

                if screenshot_image:
                    st.image(screenshot_image, caption=f"Generated Advertisement for {topic}", width=600)
                    # Upload to S3
                    s3_url = upload_pil_image_to_s3(
                        image=screenshot_image,
                        bucket_name=S3_BUCKET_NAME,
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                        region_name=AWS_REGION
                    )
                    if s3_url:
                        # e.g. "Image_1__1"
                        res[f'Image_{idx + 1}__{i + 1}'] = s3_url

        final_results.append(res)

    if final_results:
        output_df = pd.DataFrame(final_results)

        # Reorganize and flatten image links
        global image_cols
        image_cols = [col for col in output_df.columns if "Image_" in col]
        output_df[image_cols] = output_df[image_cols].apply(shift_left_and_pad, axis=1)

        st.dataframe(output_df.drop_duplicates())

        st.subheader("Final Results")
        st.dataframe(output_df)

        # Download CSV
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name='final_results.csv',
            mime='text/csv',
        )
