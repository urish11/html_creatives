import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
import boto3
import random
import requests
import os
import time
from playwright.sync_api import sync_playwright
import re
from google_images_search import GoogleImagesSearch
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set OpenAI API Key
# openai.api_key = st.secrets.get("OPENAI_API_KEY")

# AWS Credentials
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
S3_BUCKET_NAME = st.secrets["S3_BUCKET_NAME"]
AWS_REGION = st.secrets.get("AWS_REGION", "us-east-1")

client = OpenAI()

# Function to log calls
def log_function_call(func):
    def wrapper(*args, **kwargs):
        logger.info(f"CALL: {func.__name__} - args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        logger.info(f"RETURN: {func.__name__} -> {result}")
        return result
    return wrapper

@log_function_call
def fetch_google_images(query, num_images=3):
    """Fetch images from Google Search API."""
    API_KEY = random.choice(st.secrets["GOOGLE_API_KEY"])
    CX = st.secrets["GOOGLE_CX"]
    gis = GoogleImagesSearch(API_KEY, CX)

    search_params = {
        'q': query,
        'num': num_images,
    }

    try:
        gis.search(search_params)
        image_urls = [result.url for result in gis.results()]
        return image_urls
    except Exception as e:
        st.error(f"Error fetching Google Images for '{query}': {e}")
        return []

@log_function_call
def create_dalle_variation(image_url):
    """Create a variation of a Google Image using DALL-E."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        # Convert to PNG (resize if > 4MB)
        png_buffer = BytesIO()
        image.save(png_buffer, format="PNG")
        png_buffer.seek(0)

        if len(png_buffer.getvalue()) > 4 * 1024 * 1024:
            image = image.resize((512, 512))
            png_buffer = BytesIO()
            image.save(png_buffer, format="PNG")
            png_buffer.seek(0)

        # Call OpenAI API
        response = client.images.create_variation(image=png_buffer, n=1, size="512x512")
        return response["data"][0]["url"]
    except Exception as e:
        st.error(f"Error generating DALL-E variation: {e}")
        return None

@log_function_call
def upload_pil_image_to_s3(image, bucket_name, aws_access_key_id, aws_secret_access_key, object_name='', region_name='us-east-1'):
    """Upload an image to S3."""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id.strip(),
            aws_secret_access_key=aws_secret_access_key.strip(),
            region_name=region_name.strip()
        )

        object_name = object_name or f"image_{int(time.time())}_{random.randint(1000, 9999)}.png"

        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        s3_client.put_object(Bucket=bucket_name, Key=object_name, Body=img_byte_arr, ContentType='image/png')

        return f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}"
    except Exception as e:
        st.error(f"Error in S3 upload: {e}")
        return None

@log_function_call
def capture_html_screenshot_playwright(html_content):
    """Capture a screenshot of the generated HTML using Playwright."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
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
        st.error(f"Screenshot capture error: {e}")
        return None

# --------------------------------------------
# Streamlit UI
# --------------------------------------------
st.title("AI Image Generation and Upload App")

# Initialize session state for storing images
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = {}

# Table Input for Topics
st.subheader("Enter Topics for Image Generation")
df = st.data_editor(
    pd.DataFrame({"topic": ["example_topic"], "count": [1], "lang": ["english"]}),
    num_rows="dynamic",
    key="table_input"
)

if st.button("Generate Images"):
    st.session_state.generated_images = []
    for _, row in df.iterrows():
        topic = row['topic']
        count = int(row['count'])
        lang = row['lang']

        topic_images = []
        google_image_urls = fetch_google_images(topic, num_images=count)

        for img_url in google_image_urls:
            topic_images.append({
                'url': img_url,
                'source': 'google',
                'dalle_generated': False
            })

        st.session_state.generated_images.append({
            "topic": topic,
            "lang": lang,
            "images": topic_images
        })

# Display Images
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
                    img['selected_count'] = st.number_input(
                        f"Count for {img['url'][-5:]}", min_value=0, max_value=10, value=0
                    )

                    if img.get("source") == "google" and not img.get("dalle_generated", False):
                        if st.button("Get DALL-E Variation", key=f"dalle_button_{topic}_{img['url']}"):
                            dalle_url = create_dalle_variation(img['url'])
                            if dalle_url:
                                st.success("DALL-E variation generated!")
                                img["dalle_generated"] = True
                                entry["images"].append({
                                    "url": dalle_url,
                                    "source": "dalle"
                                })
                                st.experimental_rerun()

# --------------------------------------------
# Process Selected Images
# --------------------------------------------
if st.button("Process Selected Images"):
    final_results = []
    for entry in st.session_state.generated_images:
        topic = entry["topic"]
        lang = entry["lang"]
        images = entry["images"]

        res = {'Topic': topic, 'Language': lang}
        selected_images = [img for img in images if img['selected_count'] > 0]

        for idx, img in enumerate(selected_images):
            screenshot_image = capture_html_screenshot_playwright(f"<img src='{img['url']}'>")
            if screenshot_image:
                s3_url = upload_pil_image_to_s3(screenshot_image, S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
                if s3_url:
                    res[f'Image_{idx + 1}'] = s3_url
        final_results.append(res)

    st.dataframe(pd.DataFrame(final_results))
