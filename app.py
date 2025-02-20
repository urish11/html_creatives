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
import openai
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set OpenAI key
openai.api_key = st.secrets.get("OPENAI_API_KEY")

# Utility Functions
def shift_left_and_pad(row):
    """Utility function to left-shift non-null values in each row and pad with empty strings."""
    valid_values = [x for x in row if pd.notna(x)]
    padded_values = valid_values + [''] * (len(image_cols) - len(valid_values))
    return pd.Series(padded_values[:len(image_cols)])

def log_function_call(func):
    """Decorator to log function calls and return values."""
    def wrapper(*args, **kwargs):
        logger.info(f"CALL: {func.__name__} - args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        logger.info(f"RETURN: {func.__name__} -> {result}")
        return result
    return wrapper

@log_function_call
def fetch_google_images(query, num_images=3):
    """Fetch images from Google Images using google_images_search."""
    terms_list = query.split('~')
    res_urls = []
    for term in terms_list:
        API_KEY = random.choice(st.secrets["GOOGLE_API_KEY"])
        CX = st.secrets["GOOGLE_CX"]
        gis = GoogleImagesSearch(API_KEY, CX)
        search_params = {'q': term, 'num': num_images}
        try:
            gis.search(search_params)
            image_urls = [result.url for result in gis.results()]
            res_urls.extend(image_urls)
        except Exception as e:
            st.error(f"Error fetching Google Images for '{query}': {e}")
            res_urls.append([])
    return res_urls

@log_function_call
def install_playwright_browsers():
    """Install Playwright browsers (Chromium) if not installed yet."""
    try:
        os.system('playwright install-deps')
        os.system('playwright install chromium')
        return True
    except Exception as e:
        st.error(f"Failed to install Playwright browsers: {str(e)}")
        return False

@log_function_call
def upload_pil_image_to_s3(image, bucket_name, aws_access_key_id, aws_secret_access_key, object_name='', region_name='us-east-1', image_format='PNG'):
    """Upload a PIL image to S3 in PNG (or other) format."""
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

@st.cache_data
@log_function_call
def chatGPT(prompt, model="gpt-4o", temperature=1.0):
    """Call OpenAI's Chat Completion to generate text."""
    headers = {'Authorization': f'Bearer {st.secrets["GPT_API_KEY"]}', 'Content-Type': 'application/json'}
    data = {'model': model, 'temperature': temperature, 'messages': [{'role': 'user', 'content': prompt}]}
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    content = response.json()['choices'][0]['message']['content'].strip()
    return content

@st.cache_data
@log_function_call
def gen_flux_img(prompt, height=784, width=960):
    """Generate images using FLUX model from Together.xyz API."""
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
                "authorization": f"Bearer {random.choice(st.secrets['FLUX_API_KEY'])}"
            }
            response = requests.post(url, json=payload, headers=headers)
            return response.json()["data"][0]["url"]
        except Exception as e:
            if "NSFW" in str(e):
                return None
            time.sleep(2)

@log_function_call
def capture_html_screenshot_playwright(html_content):
    """Use Playwright to capture a screenshot of the given HTML snippet."""
    if not st.session_state.playwright_installed:
        st.error("Playwright browsers not installed properly")
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
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
def save_html(headline, image_url, cta_text, template, tag_line=''):
    """Returns an HTML string based on the chosen template ID."""
    if template == 1:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template</title>
            <style>
                body {{font-family: 'Gisha', sans-serif; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh;}}
                .ad-container {{width: 1000px; height: 1000px; border: 1px solid #ddd; border-radius: 20px; box-shadow: 0 8px 12px rgba(0,0,0,0.2); display: flex; flex-direction: column; justify-content: space-between; align-items: center; padding: 30px; background: url('{image_url}') no-repeat center center/cover; text-align: center;}}
                .ad-title {{font-size: 3.2em; margin-top: 10px; color: #333; background-color:white; padding: 20px 40px; border-radius: 20px;}}
                .cta-button {{font-weight: 400; padding: 40px 60px; font-size: 3em; color: white; background-color: #FF5722; border: none; border-radius: 20px; text-decoration: none; cursor: pointer; transition: background-color 0.3s ease; margin-bottom: 20px;}}
                .cta-button:hover {{background-color: #E64A19;}}
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
    elif template == 2:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template</title>
            <style>
                body {{font-family: 'Gisha', sans-serif; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh;}}
                .ad-container {{width: 1000px; height: 1000px; border: 2px solid black; box-shadow: 0 8px 12px rgba(0,0,0,0.2); display: flex; flex-direction: column; overflow: hidden; position: relative;}}
                .ad-title {{font-size: 3.2em; color: #333; background-color: white; padding: 20px; text-align: center; flex: 0 0 20%; display: flex; justify-content: center; align-items: center;}}
                .ad-image {{flex: 1 1 80%; background: url('{image_url}') no-repeat center center/cover; position: relative;}}
                .cta-button {{font-weight: 400; padding: 20px 40px; font-size: 3.2em; color: white; background-color: #FF5722; border: none; border-radius: 20px; text-decoration: none; cursor: pointer; transition: background-color 0.3s ease; position: absolute; bottom: 10%; left: 50%; transform: translateX(-50%); z-index: 10;}}
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
    elif template == 3:
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{margin: 0; padding: 0; font-family: 'Boogaloo', sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; background: #f0f0f0;}}
                .container {{position: relative; width: 1000px; height: 1000px; margin: 0; padding: 0; overflow: hidden; box-shadow: 0 0 20px rgba(0,0,0,0.2);}}
                .image {{width: 1000px; height: 1000px; object-fit: cover; filter: saturate(130%) contrast(110%); transition: transform 0.3s ease;}}
                .image:hover {{transform: scale(1.05);}}
                .overlay {{position: absolute; top: 0; left: 0; width: 100%; min-height: 14%; background: red; display: flex; justify-content: center; align-items: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); padding: 20px; box-sizing: border-box;}}
                .overlay-text {{color: #FFFFFF; font-size: 4em; text-align: center; text-shadow: 2.5px 2.5px 2px #000000; letter-spacing: 2px; margin: 0; word-wrap: break-word;}}
                .cta-button {{position: absolute; bottom: 10%; left: 50%; transform: translateX(-50%); padding: 20px 40px; background: blue; color: white; border: none; border-radius: 50px; font-size: 3.5em; cursor: pointer; transition: all 0.3s ease; text-transform: uppercase; letter-spacing: 2px; box-shadow: 0 5px 15px rgba(255,107,107,0.4);}}
                .cta-button:hover {{background: #4ECDC4; transform: translateX(-50%) translateY(-5px); box-shadow: 0 8px 20px rgba(78,205,196,0.6);}}
            </style>
            <link href="https://fonts.googleapis.com/css2?family=Boogaloo&display=swap" rel="stylesheet">
        </head>
        <body>
            <div class="container">
                <img src="{image_url}" class="image" alt="Ad Image">
                <div class="overlay">
                    <h1 class="overlay-text">{headline}</h1>
                </div>
                <button class="cta-button">{cta_text}</button>
            </div>
        </body>
        </html>
        """
    elif template == 4:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap');
                body {{margin: 0; padding: 0; font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #F4F4F4;}}
                .container {{position: relative; width: 1000px; height: 1000px; background-image: url('{image_url}'); background-size: cover; background-position: center; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.3);}}
                .text-overlay {{position: absolute; width: 95%; background-color: rgba(255,255,255,1); padding: 30px; border-radius: 10px; top: 50%; left: 50%; transform: translate(-50%,-50%); text-align: center;}}
                .small-text {{font-size: 36px; font-weight: bold; color: #333; margin-bottom: 10px; font-family: 'Calibre', Arial, sans-serif;}}
                .primary-text {{font-size: 60px; font-weight: bold; color: #FF8C00; font-family: 'Montserrat', sans-serif; line-height: 1.2; text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000;}}
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
                * {{margin: 0; padding: 0; box-sizing: border-box;}}
                body {{width: 1000px; height: 1000px; margin: 0 auto; font-family: 'Outfit', sans-serif;}}
                .container {{width: 100%; height: 100%; display: flex; flex-direction: column; position: relative; object-fit: fill;}}
                .image-container {{width: 100%; height: 60%; background-color: #f0f0f0; display: flex; align-items: center; justify-content: center;}}
                .image-container img {{width: 100%; height: 100%; object-fit: cover;}}
                .content-container {{width: 100%; height: 40%; background-color: #121421; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2rem; gap: 2rem;}}
                .main-text {{color: white; font-size: 3.5rem; font-weight: 700; text-align: center;}}
                .cta-button {{background-color: #ff0000; color: white; padding: 1rem 2rem; font-size: 3.5rem; font-weight: 700; font-family: 'Outfit', sans-serif; border: none; font-style: italic; border-radius: 8px; cursor: pointer; transition: background-color 0.3s ease;}}
                .cta-button:hover {{background-color: #cc0000;}}
                .intersection-rectangle {{position: absolute; max-width: 70%; min-width: max-content; height: 80px; background-color: #121421; left: 50%; transform: translateX(-50%); top: calc(60% - 40px); border-radius: 10px; display: flex; align-items: center; justify-content: center; padding: 0 40px;}}
                .rectangle-text {{font-family: 'Noto Color Emoji', sans-serif; color: #66FF00; font-weight: 700; text-align: center; font-size: 45px; white-space: nowrap;}}
                .highlight {{color: #FFFF00; font-size: 3.5rem; font-style: italic; font-weight: 1000; text-align: center;}}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="image-container">
                    <img src="{image_url}" alt="Ad Image">
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
    elif template == 41:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap');
                body {{margin: 0; padding: 0; font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #F4F4F4;}}
                .container {{position: relative; width: 1000px; height: 1000px; background-image: url('{image_url}'); background-size: cover; background-position: center; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.3);}}
                .text-overlay {{position: absolute; width: 95%; background-color: rgba(255,255,255,1); padding: 30px; border-radius: 10px; top: 15%; left: 50%; transform: translate(-50%,-50%); text-align: center;}}
                .small-text {{font-size: 36px; font-weight: bold; color: #333; margin-bottom: 10px; font-family: 'Calibre', Arial, sans-serif;}}
                .primary-text {{font-size: 60px; font-weight: bold; color: #FF8C00; font-family: 'Montserrat', sans-serif; line-height: 1.2; text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000;}}
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
    elif template == 42:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap');
                body {{margin: 0; padding: 0; font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #F4F4F4;}}
                .container {{position: relative; width: 1000px; height: 1000px; background-image: url('{image_url}'); background-size: cover; background-position: center; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.3);}}
                .text-overlay {{position: absolute; width: 95%; background-color: rgba(255,255,255,1); padding: 30px; border-radius: 10px; top: 90%; left: 50%; transform: translate(-50%,-50%); text-align: center;}}
                .small-text {{font-size: 36px; font-weight: bold; color: #333; margin-bottom: 10px; font-family: 'Calibre', Arial, sans-serif;}}
                .primary-text {{font-size: 60px; font-weight: bold; color: #FF8C00; font-family: 'Montserrat', sans-serif; line-height: 1.2; text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000;}}
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
    elif template == 6:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap');
                body {{margin: 0; padding: 0; font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #F4F4F4;}}
                .container {{position: relative; width: 1000px; height: 1000px; background-image: url('{image_url}'); background-size: cover; background-position: center; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.3);}}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="text-overlay"></div>
            </div>
        </body>
        </html>
        """
    else:
        html_template = f"<p>Template {template} not found</p>"
    return html_template

@log_function_call
def create_dalle_variation(image_url, count):
    """Downloads a Google image, converts it to PNG, creates a DALL-E variation."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(image_url, headers=headers)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        png_buffer = BytesIO()
        img.save(png_buffer, format="PNG")
        png_buffer.seek(0)
        if len(png_buffer.getvalue()) > 4 * 1024 * 1024:
            img = img.resize((512, 512))
            png_buffer = BytesIO()
            img.save(png_buffer, format="PNG")
            png_buffer.seek(0)
        response = client.images.create_variation(image=png_buffer, n=count, size="512x512")
        return response.data
    except Exception as e:
        st.error(f"Error generating DALL-E variation: {e}")
        return None

# Load Secrets
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
S3_BUCKET_NAME = st.secrets["S3_BUCKET_NAME"]
AWS_REGION = st.secrets.get("AWS_REGION", "us-east-1")
GPT_API_KEY = st.secrets["GPT_API_KEY"]
FLUX_API_KEY = st.secrets["FLUX_API_KEY"]

client = OpenAI(api_key=GPT_API_KEY)

# Set page config
st.set_page_config(page_title="AI Image Generator", layout="wide")

# Install Playwright if needed
if 'playwright_installed' not in st.session_state:
    st.session_state.playwright_installed = install_playwright_browsers()

# --------------------------------------------
# Streamlit UI
# --------------------------------------------
st.title("✨ AI Image Generation & Upload")

# Help section
with st.expander("How to Use This App"):
    st.markdown("""
    1. Enter a topic, number of images, language, and templates below.
    2. Click "Generate Images" to create or fetch images.
    3. Select desired images using sliders and adjust as needed.
    4. Confirm and process to generate ads and upload to S3.
    """)

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = {}

# Input Section
st.subheader("📋 Generate Images")
col1, col2 = st.columns(2)
with col1:
    topic = st.text_input("Topic", value="example_topic", help="e.g., 'Nature' or 'Google | Nature'")
    count = st.number_input("Number of Images", min_value=1, max_value=10, value=1)
with col2:
    lang = st.selectbox("Language", options=["english", "spanish", "french", "german"], help="Choose output language")
    templates = st.multiselect("Templates", options=["1", "2", "3", "4", "41", "42", "5", "6"], default=["1"], 
                              help="Select template IDs for image styles")

template_str = ",".join(templates)

# Generate Images Button
if st.button("Generate Images", key="gen_btn"):
    st.session_state.generated_images = []
    processed_combinations = set()
    combo = f"{topic}_{lang}"

    if combo not in processed_combinations:
        processed_combinations.add(combo)
        st.info(f"Generating images for '{topic}'...")
        topic_images = []
        temp_topic = topic

        if "google" in topic.lower():
            topic = topic.replace('google', ' ')
            topic_for_google = re.sub("^.*\|", "", topic) if '|' in topic else topic
            google_image_urls = fetch_google_images(topic_for_google, num_images=int(count))
            for idx, img_url in enumerate(google_image_urls):
                with st.spinner(f"Fetching Google image {idx + 1}..."):
                    topic_images.append({
                        'url': img_url,
                        'selected_count': 0,
                        'template': random.choice([int(x) for x in templates]),
                        'source': 'google',
                        'dalle_generated': False
                    })
        else:
            for i in range(count):
                topic = random.choice(temp_topic.split("^")) if '^' in temp_topic else temp_topic
                template = random.choice([int(x) for x in templates]) if "," in template_str else int(template_str.replace("*", ""))
                new_prompt = "*" in template_str

                with st.spinner(f"Generating image {i + 1} of {count} for '{topic}'..."):
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
                    elif not new_prompt:
                        image_prompt = chatGPT(
                            f"""Generate a visual image description 15 words MAX for {topic}.
                            Be creative, show the value of the offer (saving money, time, health, etc.) in a sensational yet simplistic scene.
                            Include one person and do not include text in the image. 
                            Output is up to 5 words. Think like a camera snapshot!""",
                            model='gpt-4', temperature=1.15
                        )
                    else:
                        image_prompt = chatGPT(
                            f"""Generate a visual image description 15 words MAX for {topic}.
                            Use a visually enticing style with high CTR, avoid obvious descriptions.""",
                            model='o1-mini'
                        )
                    image_url = gen_flux_img(
                        f"{random.choice(['cartoony clipart of ', '', ''])}{image_prompt}",
                        width=688 if template == 5 else 960,
                        height=416 if template == 5 else 784
                    )
                    if image_url:
                        topic_images.append({
                            'url': image_url,
                            'selected_count': 0,
                            'template': template,
                            'source': 'flux',
                            'dalle_generated': False
                        })

        st.session_state.generated_images.append({"topic": topic, "lang": lang, "images": topic_images})
        st.success(f"Generated {len(topic_images)} images for '{topic}'!")

# Display and Select Images
if st.session_state.generated_images:
    st.subheader("🖼️ Select Images")
    st.divider()
    zoom = st.slider("Image Size", 50, 500, 300, 50, help="Adjust display size of images")

    for entry in st.session_state.generated_images:
        st.markdown(f"#### {entry['topic']} ({entry['lang']})")
        cols = st.columns(3)
        for idx, img in enumerate(entry["images"]):
            with cols[idx % 3]:
                st.image(img['url'], width=zoom, caption=f"Template {img['template']}")
                img['selected_count'] = st.slider(
                    f"Count for Image {idx + 1}",
                    0, 10, 0, key=f"sel_{entry['topic']}_{img['url']}"
                )
                if img['source'] == "google" and not img.get("dalle_generated", False):
                    if st.button("DALL-E Variation", key=f"dalle_{entry['topic']}_{img['url']}"):
                        dalle_urls = create_dalle_variation(img['url'], img['selected_count'])
                        if dalle_urls:
                            for dalle_img in dalle_urls:
                                entry["images"].append({
                                    "url": dalle_img.url,
                                    "selected_count": 0,
                                    "template": img["template"],
                                    "source": "dalle",
                                    "dalle_generated": True
                                })
                            st.success("DALL-E variation added!")
        st.divider()

# Process Selected Images
if st.button("Process Selected Images", key="proc_btn"):
    if st.checkbox("Confirm Processing", help="Check to proceed with generating ads and uploading to S3"):
        final_results = []
        for entry in st.session_state.generated_images:
            selected_images = [img for img in entry["images"] if img['selected_count'] > 0]
            if not selected_images:
                continue

            res = {'Topic': entry['topic'], 'Language': entry['lang']}
            cta_texts = {}
            for idx, img in enumerate(selected_images):
                for i in range(img['selected_count']):
                    template = img['template']
                    cta_text = cta_texts.get(entry['lang'], chatGPT(
                        f"Return EXACTLY 'Learn More' in {entry['lang']} (no quotes)."
                    ).replace('"', ''))
                    cta_texts[entry['lang']] = cta_text

                    if template in [4, 41, 42]:
                        headline_text = entry['topic']
                        cta_text = chatGPT(f"Return EXACTLY 'Read more about' in {entry['lang']} (no quotes).").replace('"', '')
                    elif template == 6:
                        headline_text = ''
                    else:
                        headline_prompt = (
                            f"Write a concise headline for {entry['topic']} in {entry['lang']}, no quotes."
                            if template not in [1, 2, 3, 5] else
                            f"Write a short text (up to 20 words) to promote an article about {entry['topic']} in {entry['lang']}..."
                        )
                        headline_text = chatGPT(headline_prompt, model='gpt-4').strip('"').strip("'")

                    tag_line = chatGPT(
                        f"Write a short tagline for {re.sub('\\|.*','',entry['topic'])} in {entry['lang']}, max 25 chars, ALL CAPS...",
                        model='gpt-4'
                    ).strip('"').strip("'") if template == 5 else ''

                    html_content = save_html(headline_text, img['url'], cta_text, template, tag_line)
                    screenshot_image = capture_html_screenshot_playwright(html_content)
                    if screenshot_image:
                        s3_url = upload_pil_image_to_s3(
                            screenshot_image, S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
                        )
                        if s3_url:
                            res[f'Image_{idx + 1}__{i + 1}'] = s3_url
                            st.image(screenshot_image, caption=f"Ad for {entry['topic']}", width=600)

            final_results.append(res)

        if final_results:
            output_df = pd.DataFrame(final_results)
            global image_cols
            image_cols = [col for col in output_df.columns if "Image_" in col]
            output_df[image_cols] = output_df[image_cols].apply(shift_left_and_pad, axis=1)
            st.subheader("📊 Results")
            st.dataframe(output_df)
            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "results.csv", "text/csv")
    else:
        st.warning("Please confirm processing to proceed.")
