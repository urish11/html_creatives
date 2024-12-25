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
import base64
from tempfile import NamedTemporaryFile

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Then use logger.info() throughout your code
logger.info("Starting image generation...")

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

def capture_html_screenshot_playwright(html_content):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1000, 'height': 1000})

        # Create a temporary HTML file
        with NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
            f.write(html_content)
            temp_html_path = f.name

        # Navigate to the HTML file
        page.goto(f'file://{temp_html_path}')

        # Wait for any animations/loading
        page.wait_for_timeout(1000)

        # Capture screenshot
        screenshot_bytes = page.screenshot()

        # Clean up
        browser.close()
        os.unlink(temp_html_path)

        return Image.open(BytesIO(screenshot_bytes))


def upload_pil_image_to_s3(image, bucket_name, aws_access_key_id, aws_secret_access_key, object_name='',
                           region_name='us-east-1', image_format='PNG'):
    try:
        # Create S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id.strip(),  # Remove any potential whitespace
            aws_secret_access_key=aws_secret_access_key.strip(),  # Remove any potential whitespace
            region_name=region_name.strip()  # Remove any potential whitespace
        )

        # Generate random object name if not provided
        if not object_name:
            object_name = f"image_{int(time.time())}_{random.randint(1000, 9999)}.{image_format.lower()}"

        # Convert image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format=image_format)
        img_byte_arr.seek(0)

        # Upload to S3
        try:
            s3_client.put_object(
                Bucket=bucket_name.strip(),  # Remove any potential whitespace
                Key=object_name,
                Body=img_byte_arr,
                ContentType=f'image/{image_format.lower()}'
            )

            # Generate the URL
            url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}"
            print(f"Successfully uploaded image to: {url}")
            return url

        except Exception as e:
            print(f"Error during S3 upload: {str(e)}")
            print(f"Bucket: {bucket_name}")
            print(f"Region: {region_name}")
            print(f"Object name: {object_name}")
            return None

    except Exception as e:
        print(f"Error creating S3 client: {str(e)}")
        return None


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


def gen_flux_img(prompt):
    print('start' + prompt)
    while True:
        try:
            url = "https://api.together.xyz/v1/images/generations"

            payload = {
                "prompt": f"{prompt}",
                "model": "black-forest-labs/FLUX.1-schnell-Free",
                "steps": 4,
                "n": 1,
                "height": 480 * 2,
                "width": 480 * 2,
                # "seed": 0
            }
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {FLUX_API_KEY}"
            }

            response = requests.post(url, json=payload, headers=headers)
            print(response.json())
            return response.json()["data"][0]["url"]
        except Exception as e:
            if "NSFW" in str(e):
                return " "
            print(e)
            time.sleep(2)


def test_background_remover(image_url, output_path="output_image.png"):
    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch image from {image_url}")

    temp_input_path = "temp_input_image.png"
    with open(temp_input_path, "wb") as f:
        f.write(response.content)

    img = Image.open(temp_input_path)
    img.save(output_path)
    os.remove(temp_input_path)
    return output_path


def save_html(headline, main_text, image_url, cta_text, output_file="advertisement.html"):
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
            font-weight: 450;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background: url('{image_url}') no-repeat center center/cover;
        }}
        .ad-container {{
            width: 1000px;
            height: 1000px;
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid #ddd;
            border-radius: 20px;
            box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            align-items: center;
            padding: 30px;
        }}
        .ad-title {{
            font-size: 3em;
            margin-top: 20px;
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
            border-radius: 10px;
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
        <div class="ad-title">Transform Your Experience</div>
        <a href="#" class="cta-button">learn more</a>
    </div>
</body>
</html>


    """
    with open(output_file, "w") as f:
        f.write(html_template)
    return html_template


# --------------------------------------------
# Streamlit UI
# --------------------------------------------

st.title("AI Image Generation and Upload App")
st.sidebar.header("Configuration")

# Table Input for Topics
st.subheader("Enter Topics for Image Generation")
df = st.data_editor(
    pd.DataFrame({"topic": ["example_topic"], "count": [1], "lang": ["en"]}),
    num_rows="dynamic",
    key="table_input"
)

if st.button("Generate Images and Upload"):
    try:

        if df.empty:
            st.warning("Please fill in the table before proceeding.")
        else:
            final_results = []

            for _, row in df.iterrows():
                topic = row['topic']
                count = int(row['count'])
                lang = row['lang']

                st.subheader(f"Processing: {topic}")
                for i in range(count):
                    st.write(f"Generating image {i + 1} for '{topic}'...")

                    # Step 1: Get Image Description
                    image_prompt = chatGPT(f"Generate a visual description for {topic}")

                    # Step 2: Generate Image
                    image_url = gen_flux_img(image_prompt)

                    if image_url:
                        # Generate HTML with the Flux image
                        html_content = save_html(
                            headline="Transform Your Experience",
                            main_text="Your Main Text",
                            image_url=image_url,
                            cta_text="Learn More"
                        )

                        try:
                            # Capture screenshot of the HTML
                            screenshot_image = capture_html_screenshot_playwright(html_content)

                            # Display the screenshot in Streamlit
                            st.image(screenshot_image, caption="Generated Advertisement")

                            # Upload screenshot to S3
                            s3_url = upload_pil_image_to_s3(
                                image=screenshot_image,
                                bucket_name=S3_BUCKET_NAME,
                                aws_access_key_id=AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                region_name=AWS_REGION
                            )

                            if s3_url:
                                final_results.append({
                                    'Topic': topic,
                                    'Count': count,
                                    'Language': lang,
                                    'Image URL': s3_url
                                })

                        except Exception as e:
                            st.error(f"Error capturing screenshot: {str(e)}")

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
    except Exception as e:
        st.error(f'Error: {str(e)}')


