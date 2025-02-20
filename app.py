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
            st.error(f"Error fetching Google Images for '{query}': Please ensure your API keys are valid and try again.")
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
        st.error(f"Failed to install Playwright browsers: {str(e)}. Please ensure proper permissions.")
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
        st.error(f"Error in S3 upload: {str(e)}. Please check your AWS credentials.")
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
                st.error(f"Failed to generate image: Content flagged as NSFW.")
                return None
            st.error(f"Error generating image: {str(e)}. Retrying...")
            time.sleep(2)

@log_function_call
def capture_html_screenshot_playwright(html_content):
    """Use Playwright to capture a screenshot of the given HTML snippet."""
    if not st.session_state.playwright_installed:
        st.error("Playwright browsers not installed properly. Cannot capture screenshot.")
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
        st.error(f"Screenshot capture error: {str(e)}. Ensure Playwright is installed correctly.")
        return None

@log_function_call
def generate_custom_template(wizard_data):
    """Generate a custom HTML template based on wizard input."""
    layout = wizard_data['layout']
    body_font = wizard_data['body_font']
    heading_font = wizard_data['heading_font']
    cta_font = wizard_data['cta_font']
    background_color = wizard_data['background_color']
    text_color = wizard_data['text_color']
    cta_bg_color = wizard_data['cta_bg_color']
    tagline = wizard_data.get('tagline', '')

    if layout == "Full-screen image with text overlay":
        template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Custom Ad</title>
            <style>
                body {{font-family: {body_font}; margin: 0; padding: 0; background-color: {background_color}; display: flex; justify-content: center; align-items: center; height: 100vh;}}
                .ad-container {{width: 1000px; height: 1000px; background: url('{{image_url}}') no-repeat center center/cover; position: relative; text-align: center;}}
                .text-overlay {{position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: {text_color};}}
                .headline {{font-family: {heading_font}; font-size: 3.5em; margin: 0;}}
                .cta-button {{font-family: {cta_font}; font-size: 2.5em; padding: 20px 40px; background-color: {cta_bg_color}; color: white; border: none; border-radius: 10px; text-decoration: none; cursor: pointer; transition: background-color 0.3s ease; display: inline-block;}}
                .cta-button:hover {{background-color: #666;}}
                .tagline {{font-family: {body_font}; font-size: 1.5em; margin-top: 10px;}}
            </style>
        </head>
        <body>
            <div class="ad-container">
                <div class="text-overlay">
                    <h1 class="headline">{{headline}}</h1>
                    <p class="tagline">{tagline}</p>
                    <a href="#" class="cta-button">{{cta_text}}</a>
                </div>
            </div>
        </body>
        </html>
        """
    elif layout == "Image on top, text below":
        template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Custom Ad</title>
            <style>
                body {{font-family: {body_font}; margin: 0; padding: 0; background-color: {background_color}; display: flex; justify-content: center; align-items: center; height: 100vh;}}
                .ad-container {{width: 1000px; height: 1000px; display: flex; flex-direction: column;}}
                .image-section {{height: 60%; background: url('{{image_url}}') no-repeat center center/cover;}}
                .text-section {{height: 40%; text-align: center; padding: 20px; color: {text_color};}}
                .headline {{font-family: {heading_font}; font-size: 3em; margin: 0;}}
                .cta-button {{font-family: {cta_font}; font-size: 2em; padding: 15px 30px; background-color: {cta_bg_color}; color: white; border: none; border-radius: 10px; text-decoration: none; cursor: pointer; transition: background-color 0.3s ease;}}
                .cta-button:hover {{background-color: #666;}}
                .tagline {{font-family: {body_font}; font-size: 1.2em; margin: 10px 0;}}
            </style>
        </head>
        <body>
            <div class="ad-container">
                <div class="image-section"></div>
                <div class="text-section">
                    <h1 class="headline">{{headline}}</h1>
                    <p class="tagline">{tagline}</p>
                    <a href="#" class="cta-button">{{cta_text}}</a>
                </div>
            </div>
        </body>
        </html>
        """
    elif layout == "Text on left, image on right":
        template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Custom Ad</title>
            <style>
                body {{font-family: {body_font}; margin: 0; padding: 0; background-color: {background_color}; display: flex; justify-content: center; align-items: center; height: 100vh;}}
                .ad-container {{width: 1000px; height: 1000px; display: flex;}}
                .text-section {{width: 50%; padding: 40px; text-align: center; color: {text_color}; display: flex; flex-direction: column; justify-content: center;}}
                .image-section {{width: 50%; background: url('{{image_url}}') no-repeat center center/cover;}}
                .headline {{font-family: {heading_font}; font-size: 3em; margin: 0;}}
                .cta-button {{font-family: {cta_font}; font-size: 2em; padding: 15px 30px; background-color: {cta_bg_color}; color: white; border: none; border-radius: 10px; text-decoration: none; cursor: pointer; transition: background-color 0.3s ease;}}
                .cta-button:hover {{background-color: #666;}}
                .tagline {{font-family: {body_font}; font-size: 1.2em; margin: 10px 0;}}
            </style>
        </head>
        <body>
            <div class="ad-container">
                <div class="text-section">
                    <h1 class="headline">{{headline}}</h1>
                    <p class="tagline">{tagline}</p>
                    <a href="#" class="cta-button">{{cta_text}}</a>
                </div>
                <div class="image-section"></div>
            </div>
        </body>
        </html>
        """
    return template

@log_function_call
def save_html(headline, image_url, cta_text, template, tag_line=''):
    """Returns an HTML string based on the chosen template ID or custom template."""
    if isinstance(template, str) and template.startswith('custom_'):
        # Handle custom template
        custom_template = st.session_state['custom_templates'].get(template)
        if custom_template:
            return custom_template.format(image_url=image_url, headline=headline, cta_text=cta_text)
        else:
            return "<p>Custom template not found</p>"
    # Existing predefined templates
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
    # Add other predefined templates (3, 4, 41, 42, 5, 6) as needed for brevity
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
        st.error(f"Error generating DALL-E variation: {e}. Check image URL or API key.")
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

# Initialize session state for custom templates
if 'custom_templates' not in st.session_state:
    st.session_state['custom_templates'] = {}
if 'wizard_step' not in st.session_state:
    st.session_state['wizard_step'] = 0
if 'wizard_data' not in st.session_state:
    st.session_state['wizard_data'] = {}

# --------------------------------------------
# Streamlit UI
# --------------------------------------------
st.title("✨ AI Image Generation & Upload")

# Help section
with st.expander("How to Use This App"):
    st.markdown("""
    1. Enter a topic, number of images, language, and templates below.
    2. Create custom templates using the wizard (optional).
    3. Generate images, select desired ones, and preview ads.
    4. Confirm and process to generate ads and upload to S3.
    """)

# Template Wizard
if st.button("Launch Template Wizard"):
    st.session_state['wizard_step'] = 1

if st.session_state['wizard_step'] > 0:
    st.subheader("🛠️ Template Wizard")
    if st.session_state['wizard_step'] == 1:
        st.write("Step 1: Choose Layout Type")
        layout = st.selectbox("Select Layout", ["Full-screen image with text overlay", "Image on top, text below", "Text on left, image on right"])
        if st.button("Next"):
            st.session_state['wizard_data']['layout'] = layout
            st.session_state['wizard_step'] = 2
    elif st.session_state['wizard_step'] == 2:
        st.write("Step 2: Choose Fonts")
        body_font = st.selectbox("Body Font", ["Arial", "Times New Roman", "Open Sans", "Montserrat"])
        heading_font = st.selectbox("Heading Font", ["Arial", "Times New Roman", "Open Sans", "Montserrat"])
        cta_font = st.selectbox("CTA Font", ["Arial", "Times New Roman", "Open Sans", "Montserrat"])
        if st.button("Next"):
            st.session_state['wizard_data']['body_font'] = body_font
            st.session_state['wizard_data']['heading_font'] = heading_font
            st.session_state['wizard_data']['cta_font'] = cta_font
            st.session_state['wizard_step'] = 3
        if st.button("Back"):
            st.session_state['wizard_step'] = 1
    elif st.session_state['wizard_step'] == 3:
        st.write("Step 3: Choose Colors")
        background_color = st.color_picker("Background Color", "#FFFFFF")
        text_color = st.color_picker("Text Color", "#000000")
        cta_bg_color = st.color_picker("CTA Background Color", "#FF0000")
        if st.button("Next"):
            st.session_state['wizard_data']['background_color'] = background_color
            st.session_state['wizard_data']['text_color'] = text_color
            st.session_state['wizard_data']['cta_bg_color'] = cta_bg_color
            st.session_state['wizard_step'] = 4
        if st.button("Back"):
            st.session_state['wizard_step'] = 2
    elif st.session_state['wizard_step'] == 4:
        st.write("Step 4: Customize Specific Elements")
        if st.session_state['wizard_data']['layout'] in ["Full-screen image with text overlay", "Image on top, text below", "Text on left, image on right"]:
            tagline = st.text_input("Tagline (optional)", "")
        if st.button("Next"):
            st.session_state['wizard_data']['tagline'] = tagline
            st.session_state['wizard_step'] = 5
        if st.button("Back"):
            st.session_state['wizard_step'] = 3
    elif st.session_state['wizard_step'] == 5:
        st.write("Step 5: Review and Save")
        custom_template = generate_custom_template(st.session_state['wizard_data'])
        st.code(custom_template, language="html")
        placeholder_html = custom_template.format(image_url="https://via.placeholder.com/1000", headline="Sample Headline", cta_text="Click Me")
        preview_image = capture_html_screenshot_playwright(placeholder_html)
        if preview_image:
            st.image(preview_image, caption="Template Preview", width=600)
        template_name = st.text_input("Template Name", "custom_template")
        if st.button("Save Template"):
            custom_id = f"custom_{len(st.session_state['custom_templates']) + 1}"
            st.session_state['custom_templates'][custom_id] = custom_template
            st.success(f"Template saved as '{custom_id}'!")
            st.session_state['wizard_step'] = 0
        if st.button("Back"):
            st.session_state['wizard_step'] = 4

# Input Section
st.subheader("📋 Generate Images")
col1, col2 = st.columns(2)
with col1:
    topic = st.text_input("Topic", value="example_topic", help="e.g., 'Nature' or 'Google | Nature'")
    count = st.number_input("Number of Images", min_value=1, max_value=10, value=1)
with col2:
    lang = st.selectbox("Language", options=["english", "spanish", "french", "german"], help="Choose output language")
    all_templates = ["1", "2"] + list(st.session_state['custom_templates'].keys())  # Add more predefined templates as needed
    templates = st.multiselect("Templates", options=all_templates, default=["1"], help="Select template IDs or custom templates")

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
                        'selected': False,
                        'selected_count': 0,
                        'template': random.choice(templates),
                        'source': 'google',
                        'dalle_generated': False
                    })
        else:
            for i in range(count):
                topic = random.choice(temp_topic.split("^")) if '^' in temp_topic else temp_topic
                template = random.choice(templates) if "," in template_str else template_str.replace("*", "")
                new_prompt = "*" in template_str

                with st.spinner(f"Generating image {i + 1} of {count} for '{topic}'..."):
                    if template == "5":
                        image_prompt = chatGPT(
                            f"""Generate a concise visual image description (15 words MAX) for {topic}...""",
                            model='gpt-4', temperature=1.2
                        )
                    elif not new_prompt:
                        image_prompt = chatGPT(
                            f"""Generate a visual image description 15 words MAX for {topic}...""",
                            model='gpt-4', temperature=1.15
                        )
                    else:
                        image_prompt = chatGPT(
                            f"""Generate a visual image description 15 words MAX for {topic}...""",
                            model='o1-mini'
                        )
                    image_url = gen_flux_img(
                        f"{random.choice(['cartoony clipart of ', '', ''])}{image_prompt}",
                        width=688 if template == "5" else 960,
                        height=416 if template == "5" else 784
                    )
                    if image_url:
                        topic_images.append({
                            'url': image_url,
                            'selected': False,
                            'selected_count': 0,
                            'template': template,
                            'source': 'flux',
                            'dalle_generated': False
                        })

        st.session_state.generated_images = [{"topic": topic, "lang": lang, "images": topic_images}]
        st.success(f"Generated {len(topic_images)} images for '{topic}'!")

# Display and Select Images
if 'generated_images' in st.session_state and st.session_state.generated_images:
    st.subheader("🖼️ Select Images")
    st.divider()
    zoom = st.slider("Image Size", 50, 500, 300, 50, help="Adjust display size of images")

    for entry in st.session_state.generated_images:
        st.markdown(f"#### {entry['topic']} ({entry['lang']})")
        cols = st.columns(3)
        for idx, img in enumerate(entry["images"]):
            with cols[idx % 3]:
                st.image(img['url'], width=zoom, caption=f"Template {img['template']}")
                include = st.checkbox("Include", key=f"include_{entry['topic']}_{img['url']}")
                if include:
                    img['selected'] = True
                    img['selected_count'] = st.number_input("Count", min_value=1, max_value=10, value=1, key=f"count_{entry['topic']}_{img['url']}")
                else:
                    img['selected'] = False
                    img['selected_count'] = 0
                if img['source'] == "google" and not img.get("dalle_generated", False):
                    if st.button("DALL-E Variation", key=f"dalle_{entry['topic']}_{img['url']}"):
                        dalle_urls = create_dalle_variation(img['url'], img['selected_count'])
                        if dalle_urls:
                            for dalle_img in dalle_urls:
                                entry["images"].append({
                                    "url": dalle_img.url,
                                    "selected": False,
                                    "selected_count": 0,
                                    "template": img["template"],
                                    "source": "dalle",
                                    "dalle_generated": True
                                })
                            st.success("DALL-E variation added!")
                if st.button("View Ad", key=f"view_{entry['topic']}_{img['url']}"):
                    headline_text = chatGPT(f"Write a concise headline for {entry['topic']} in {entry['lang']}, no quotes.", model='gpt-4')
                    cta_text = chatGPT(f"Return EXACTLY 'Learn More' in {entry['lang']} (no quotes).").replace('"', '')
                    html_content = save_html(headline_text, img['url'], cta_text, img['template'], '')
                    st.components.v1.html(html_content, height=600, scrolling=True)

        st.divider()

# Text Customization
st.subheader("✍️ Text Customization")
text_option = st.selectbox("Text Generation", ["AI-Generated", "Custom Text"])
custom_headline = ""
custom_cta_text = ""
if text_option == "Custom Text":
    custom_headline = st.text_input("Custom Headline", "")
    custom_cta_text = st.text_input("Custom CTA Text", "")

# Process Selected Images
if st.button("Process Selected Images", key="proc_btn"):
    if st.checkbox("Confirm Processing", help="Check to proceed with generating ads and uploading to S3"):
        final_results = []
        for entry in st.session_state.generated_images:
            selected_images = [img for img in entry["images"] if img['selected']]
            if not selected_images:
                continue

            res = {'Topic': entry['topic'], 'Language': entry['lang']}
            cta_texts = {}
            for idx, img in enumerate(selected_images):
                for i in range(img['selected_count']):
                    template = img['template']
                    if text_option == "Custom Text" and custom_headline and custom_cta_text:
                        headline_text = custom_headline
                        cta_text = custom_cta_text
                    else:
                        cta_text = cta_texts.get(entry['lang'], chatGPT(
                            f"Return EXACTLY 'Learn More' in {entry['lang']} (no quotes)."
                        ).replace('"', ''))
                        cta_texts[entry['lang']] = cta_text
                        if template in ["4", "41", "42"]:
                            headline_text = entry['topic']
                            cta_text = chatGPT(f"Return EXACTLY 'Read more about' in {entry['lang']} (no quotes).").replace('"', '')
                        elif template == "6":
                            headline_text = ''
                        else:
                            headline_prompt = (
                                f"Write a concise headline for {entry['topic']} in {entry['lang']}, no quotes."
                                if template not in ["1", "2", "3", "5"] else
                                f"Write a short text (up to 20 words) to promote an article about {entry['topic']} in {entry['lang']}..."
                            )
                            headline_text = chatGPT(headline_prompt, model='gpt-4').strip('"').strip("'")

                    tag_line = chatGPT(
                        f"Write a short tagline for {re.sub('\\|.*','',entry['topic'])} in {entry['lang']}, max 25 chars, ALL CAPS...",
                        model='gpt-4'
                    ).strip('"').strip("'") if template == "5" else ''

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
            st.success("Ads processed and uploaded to S3!")
    else:
        st.warning("Please confirm processing to proceed.")
