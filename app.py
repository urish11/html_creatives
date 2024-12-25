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

# --------------------------------------------
# Setup and Installation
# --------------------------------------------
st.set_page_config(layout="wide")

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
def upload_pil_image_to_s3(image, bucket_name, aws_access_key_id, aws_secret_access_key, object_name='', region_name='us-east-1', image_format='PNG'):
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

def chatGPT(prompt, model="gpt-4", temperature=1.0):
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
    while True:
        try:
            url = "https://api.together.xyz/v1/images/generations"
            payload = {
                "prompt": prompt,
                "model": "black-forest-labs/FLUX.1-schnell-Free",
                "steps": 4,
                "n": 1,
                "height": 480*2,
                "width": 480*2,
            }
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {FLUX_API_KEY}"
            }

            response = requests.post(url, json=payload, headers=headers)
            return response.json()["data"][0]["url"]
        except Exception as e:
            if "NSFW" in str(e):
                return None
            time.sleep(2)

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
    pd.DataFrame({"topic": ["example_topic"], "count": [1], "lang": ["en"]}),
    num_rows="dynamic",
    key="table_input"
)

# Step 1: Generate Images
if st.button("Generate Images"):
    st.session_state.generated_images = {}
    
    for _, row in df.iterrows():
        topic = row['topic']
        count = int(row['count'])
        
        st.subheader(f"Generating images for: {topic}")
        topic_images = []
        
        for i in range(count):
            with st.spinner(f"Generating image {i + 1} for '{topic}'..."):
                image_prompt = chatGPT(f"Generate a visual description for {topic}")
                image_url = gen_flux_img(image_prompt)
                
                if image_url:
                    topic_images.append({
                        'url': image_url,
                        'selected': False  # Add selection state
                    })
        
        st.session_state.generated_images[topic] = topic_images

# Display generated images in a grid
if st.session_state.generated_images:
    st.subheader("Select Images to Process")
    
    for topic, images in st.session_state.generated_images.items():
        st.write(f"### {topic}")
        cols = st.columns(len(images))
        
        for idx, (col, img) in enumerate(zip(cols, images)):
            with col:
                st.image(img['url'], use_column_width=True)
                images[idx]['selected'] = st.checkbox(f"Select {topic} image {idx + 1}", key=f"{topic}_{idx}")

    # Step 2: Process Selected Images
    if st.button("Process Selected Images"):
        final_results = []
        
        for topic, images in st.session_state.generated_images.items():
            selected_images = [img for img in images if img['selected']]
            
            for img in selected_images:
                # Generate HTML with the selected image
                html_content = save_html(
                    headline="Transform Your Experience",
                    main_text="Your Main Text",
                    image_url=img['url'],
                    cta_text="Learn More"
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
                        final_results.append({
                            'Topic': topic,
                            'Image URL': s3_url
                        })
        
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
