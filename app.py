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


import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def log_function_call(func):
    def wrapper(*args, **kwargs):
        logger.info(f"CALL: {func.__name__} - args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        logger.info(f"RETURN: {func.__name__} -> {result}")
        return result
    return wrapper





# --------------------------------------------
# Setup and Installation 
# --------------------------------------------
st.set_page_config(layout="wide")

@log_function_call
def install_playwright_browsers():
    try:
        os.system('playwright install-deps')
        os.system('playwright install chromium')
        return True
    except Exception as e:
        st.error(f"Failed to install Playwright browsers: {str(e)}")
        return False


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


# --------------------------------------------
# Utility Functions
# --------------------------------------------
@log_function_call
def upload_pil_image_to_s3(image, bucket_name, aws_access_key_id, aws_secret_access_key, object_name='',
                           region_name='us-east-1', image_format='PNG'):
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
    st.write("Generating image description...")
    headers = {
        'Authorization': f'Bearer {GPT_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}]
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    content = response.json()['choices'][0]['message']['content'].strip()
    return content

@log_function_call
def gen_flux_img(prompt):
    while True:
        try:
            url = "https://api.together.xyz/v1/images/generations"
            payload = {
                "prompt": prompt,
                "model": "black-forest-labs/FLUX.1-schnell-Free",
                "steps": 4,
                "n": 1,
                "height": 480 * 2,
                "width": 480 * 2,
            }
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {FLUX_API_KEY}"
            }
        
            response = requests.post(url, json=payload, headers=headers)
            print(response)
            return response.json()["data"][0]["url"]
        except Exception as e:
            print(e)
            if "NSFW" in str(e):
                return None
            time.sleep(2)

@log_function_call
def capture_html_screenshot_playwright(html_content):
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
def save_html(headline, image_url, cta_text, template, output_file="advertisement.html"):
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

    if template == 2:
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

    if template == 3:

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
                    //border-radius: 10px;
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
                    /* Remove fixed height */
                    min-height: 14%; /* Set minimum height instead */
                    background: red;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    /* Add padding for better text containment */
                    padding: 20px;
                    /* Add box-sizing to include padding in height calculation */
                    box-sizing: border-box;
                }}

                .overlay-text {{
                    color: #FFFFFF;
                    font-size: 4em;
                    text-align: center;
                    text-shadow: 2.5px 2.5px 2px #000000;
                    letter-spacing: 2px;
                    font-family: 'Boogaloo', 'Segoe UI Emoji', 'Apple Color Emoji', sans-serif;
                    /* Remove any margin that might affect sizing */
                    margin: 0;
                    /* Add word-wrap for very long text */
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

    else:

        print('template not found')

    return html_template


# --------------------------------------------
# Streamlit UI
# --------------------------------------------
st.title("AI Image Generation and Upload App")

# Initialize session state for storing generated images
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = {}

# Table Input for Topics
st.subheader("Enter Topics for Image Generation")
df = st.data_editor(
    pd.DataFrame({"topic": ["example_topic"], "count": [1], "lang": ["english"], "template": ["1,2,3 use , for multi"]}),
    num_rows="dynamic",
    key="table_input"
)

# Step 1: Generate Images
# Step 1: Generate Images
if st.button("Generate Images"):
    st.session_state.generated_images = []  # Clear previous images

    # Create a set to track unique topic-lang combinations
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

            for i in range(count):

                if "," in template_str:
                    template = random.choice([int(x) for x in template_str.split(",")])
                else:
                    template = int(template_str)

                with st.spinner(f"Generating image {i + 1} for '{topic}'..."):
                    image_prompt = chatGPT(
                        f"""Generate a  visual image description  15 words MAX for  {topic}  . Be   creative and intriguing,think of and show the value of the offer like (examples, use whatever is relevant if relevant, or others in the same vibe, must be relevant to the offer): saving money, time, be healthier, more educated etc.. show a SENSATIONAL AND DRAMATIC SCENE  ,  don't include text in the image. make sure the offer is conveyed clearly. output is 5 words MAX, use a person in image, write what is seen like a camera! show a SENSATIONAL AND DRAMATIC SCENE VERY SIMPLISTIC SCENE, SHOW TOPIC EXPLICITLY  """,
                        model='gpt-4', temperature=1.15)  # Your existing prompt
                    image_url = gen_flux_img(
                        f"{random.choice(['cartoony clipart of ', 'cartoony clipart of ', ''])}  {image_prompt}")

                    if image_url:
                        topic_images.append({
                            'url': image_url,
                            'selected': False,
                            'template': template

                        })

            st.session_state.generated_images.append({
                "topic": topic,
                "lang": lang,
                "images": topic_images
            })

# Display generated images in a grid
if st.session_state.generated_images:
    st.subheader("Select Images to Process")

    for entry in st.session_state.generated_images:
        topic = entry["topic"]
        lang = entry["lang"]
        images = entry["images"]

        st.write(f"### {topic} ({lang})")
        cols = st.columns(len(images))

        for idx, (col, img) in enumerate(zip(cols, images)):
            with col:
                st.image(img['url'], use_container_width=True)
                unique_key = f"checkbox_{topic}_{lang}_{idx}"
                img['selected'] = st.checkbox(
                    f"Select image {idx + 1}",
                    key=unique_key
                )

    # Step 2: Process Selected Images
    if st.button("Process Selected Images"):
        final_results = []

        for entry in st.session_state.generated_images:
            topic = entry["topic"]
            lang = entry["lang"]
            images = entry["images"]

            res = {'Topic': topic, 'Language': lang}
            selected_images = [img for img in images if img['selected']]

            for idx, img in enumerate(selected_images):  # Generate HTML with the selected image
                template = img['template']

                if template == 1 or template == 2:
                    headline_prompt = f"write a short text (up to 20 words) for a creative to promote an article containing information about {topic} in language{lang} , your goal is to be concise but convenience users to enter the article"
                    
                elif template == 3 : 
                    headline_prompt = f"write  statement SAME LENGTH, no quotation marks, for {topic} in {lang} like 'Surprising Medicare Benefits You Might Be Missing'"

                html_content = save_html(
                    headline=chatGPT(headline_prompt).replace('"', ''),

                    image_url=img['url'],
                    cta_text=chatGPT(
                        f"return EXACTLY JUST THE TEXT the text 'Learn More' in the following language {lang} even if it is English").replace(
                        '"', ''),
                    template=template
                )

                # Capture screenshot
                screenshot_image = capture_html_screenshot_playwright(html_content)

                if screenshot_image:
                    st.image(screenshot_image, caption=f"Generated Advertisement for {topic}")

                    # Upload to S3
                    s3_url = upload_pil_image_to_s3(
                        image=screenshot_image,
                        bucket_name=S3_BUCKET_NAME,
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                        region_name=AWS_REGION
                    )

                    if s3_url:
                        res[f'Image_{idx + 1}'] = s3_url

            final_results.append(res)

        # Display Final Results
        if final_results:
            output_df = pd.DataFrame(final_results)
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
