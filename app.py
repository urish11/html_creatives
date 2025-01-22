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

import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s', 
    level=logging.DEBUG
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
def chatGPT(prompt, model="gpt-4o", temperature=1.0) :
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
    return  content

@log_function_call
def gen_flux_img(prompt,height = 784 , width =960 ):
    while True:
        try:
            url = "https://api.together.xyz/v1/images/generations"
            payload = {
                "prompt": prompt,
                "model": "black-forest-labs/FLUX.1-schnell-Free",
                "steps": 4,
                "n": 1,
                "height":  height,
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
def save_html(headline, image_url, cta_text, template,tag_line = '', output_file="advertisement.html"):
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

    if template == 4:

    

        html_template=    f"""
        <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nursing Careers in the UK</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap'); /* Import Bebas Neue and Montserrat Bold fonts */
        @font-face {{
            font-family: 'Calibre';
            src: url('path-to-calibre-font.woff2') format('woff2'); /* Replace with Calibre font path */
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
            width: 95%; /* Almost full width */
            background-color: rgba(255, 255, 255, 1); /* Transparent white background */
            padding: 30px; /* Reduced padding */
            border-radius: 10px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }}
        .small-text {{
            font-size: 36px; /* Bigger "Read more about" text */
            font-weight: bold; /* Bold styling for "Read more about" */
            color: #333;
            margin-bottom: 10px; /* Reduced margin */
            font-family: 'Calibre', Arial, sans-serif; /* Fallback to Arial */
        }}
        .primary-text {{
            font-size: 60px; /* Much larger font for primary text */
            font-weight: bold; /* Bold weight */
            color: #FF8C00; /* Orange color */
            font-family: 'Montserrat', sans-serif; /* Using Montserrat Bold */
            line-height: 1.2; /* Tighter line spacing */
            text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000; /* Black outline */
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
    
    if template == 5 :

        html_template=f"""
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
    object-fit: cover;}}



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
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
        }}
        .cta-button {{
            background-color: #ff0000;
            color: white;
            padding: 1rem 2rem;
            font-size: 3rem;
            font-weight: 700;
            
            font-family: 'Outfit', sans-serif;
            border: none;
            font-style :italic;
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

            color: yellow;
            font-weight: 700;
            text-align: center;
            font-size: 39px;
            white-space: nowrap;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="image-container">
            <img src="{image_url}" alt="Placeholder image">
        </div>
        <div class="intersection-rectangle">
            <p class="rectangle-text">⛔{tag_line}⛔</p>
        </div>
        <div class="content-container">
            <h1 class="main-text">{headline.upper()}</h1>
            <button class="cta-button">{cta_text}</button>
        </div>
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
    pd.DataFrame({"topic": ["example_topic"], "count": [1], "lang": ["english"], "template": ["1,2,3,4 use , for multi"]}),
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
            temp_topic = topic

            for i in range(count):
                topic = temp_topic


                if '^' in topic:
                    topic = random.choice(topic.split("^"))





                new_prompt = False

                if "," in template_str:
                    template = random.choice([int(x) for x in template_str.split(",")])

                elif "*" in template_str:
                    new_prompt= random.choice([True,False])
                    # new_prompt = True
                    template_str = template_str.replace("*","")
                    template = int(template_str)
                else:
                    template = int(template_str)




                    

                with st.spinner(f"Generating image {i + 1} for '{topic}'..."):

                    if template== 5 :

                        #rand_prompt = random.choice([f"""Generate a  visual image description  15 words MAX for  {topic}  . Be   creative and intriguing,think of and show the value of the offer like (examples, use whatever is relevant if relevant, or others in the same vibe, must be relevant to the offer): saving money, time, be healthier, more educated etc.. show a SENSATIONAL AND DRAMATIC SCENE  ,  don't include text in the image. make sure the offer is conveyed clearly. output is 5 words MAX, use a person in image, write what is seen like a camera! show a SENSATIONAL AND DRAMATIC SCENE VERY SIMPLISTIC SCENE, SHOW TOPIC EXPLICITLY  """,f"""Generate a visual image description (15 words MAX) for {topic}.  \nBe extremely creative, curious , and unhinged—push the limits of imagination!!!!!!!!!!!!! \nShow the value of the offer (e.g., saving money, time, improving health, or education) in a sensational, dramatic, yet very simplistic scene. The image should take 3 secs to understand what's going on while captivating the viewer \nInclude one person in the image, and clearly depict the topic. \nDo not include any text in the image. \nYour final output must be exactly 10 words, written like a camera snapshot. \nEnsure the offer is unmistakably conveyed.\n"""])
                        rand_prompt = f"""Generate a visual image description (15 words MAX) for {topic}.  \nBe extremely creative, curious , and unhinged—push the limits of imagination!!!!!!!!!!!!! \nShow the value of the offer (e.g., saving money, time, improving health, or education) in a sensational, dramatic, yet very simplistic scene. The image should take 3 secs to understand what's going on while captivating the viewer \nInclude one person in the image, and clearly depict the topic. \nDo not include any text in the image. \nYour final output must be exactly 10 words, written like a camera snapshot. \nEnsure the offer is unmistakably conveyed.\n"""
                        image_prompt = chatGPT(rand_prompt,model='gpt-4', temperature=1.02)
                        st.markdown(image_prompt)



                    elif not new_prompt:
                        image_prompt = chatGPT(
                            f"""Generate a  visual image description  15 words MAX for  {topic}  . Be   creative and intriguing,think of and show the value of the offer like (examples, use whatever is relevant if relevant, or others in the same vibe, must be relevant to the offer): saving money, time, be healthier, more educated etc.. show a SENSATIONAL AND DRAMATIC SCENE  ,  don't include text in the image. make sure the offer is conveyed clearly. output is 5 words MAX, use a person in image, write what is seen like a camera! show a SENSATIONAL AND DRAMATIC SCENE VERY SIMPLISTIC SCENE, SHOW TOPIC EXPLICITLY  """,
                            model='gpt-4', temperature=1.15)  # Your existing prompt

                    elif new_prompt : 

                        image_prompt = chatGPT(
                                f"""Generate a  visual image description  15 words MAX for  {topic}  . think of a visually very enticing way of prompting the topic!! i want very high CTR. use very  engaging ideas. dont do the obvious  descriptions """,
                                model='o1-mini')

                    if template ==5:
                        image_url = gen_flux_img(
                        f"{random.choice(['cartoony clipart of ', 'cartoony clipart of ', '',''])}  {image_prompt}",width=688,height=416)
                    else:
                        image_url = gen_flux_img(
                        f"{random.choice(['cartoony clipart of ', 'cartoony clipart of ', '',''])}  {image_prompt}")

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
                st.image(img['url'], use_container_width =True)
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

            cta_texts = {}

            for idx, img in enumerate(selected_images):  # Generate HTML with the selected image
                template = img['template']

                if template == 1 or template == 2:
                    headline_prompt = f"write a short text (up to 20 words) for a creative to promote an article containing information about {topic} in language{lang} , your goal is to be concise but convenience users to enter the article"
                    
                elif template in [3,5]  : 
                    #headline_prompt = f"write  statement SAME LENGTH, no quotation marks, for {topic} in {lang} like 'Surprising Medicare Benefits You Might Be Missing'"
                    headline_prompt = f"write 1  statement SAME LENGTH, no quotation marks, for {re.sub('\|.*','',topic)} in {lang} like examples output:\n'Surprising Travel Perks You Might Be Missing'\n'Little-Known Tax Tricks to Save Big'\n'Dont Miss Out on These Credit Card Extras'\n'Why Most Shoppers Miss These Loyalty Rewards'\n'Home Improvement Hacks Youll Wish You Knew Sooner' \n\n\n dont use Hidden, Unlock \n  "
                    
                if lang in cta_texts:
                    cta_text = cta_texts[lang]
                else:
                   cta_texts[lang]  = chatGPT(
                        f"return EXACTLY JUST THE TEXT the text 'Learn More' in the following language {lang} even if it is English").replace(
                        '"', '')
                   cta_text = cta_texts[lang]
                if template == 4:

                    headline_text = topic
                    cta_text = chatGPT(f"Retrun JUST 'Read more about' in {lang} JUST THE TEXT NO INTROS ").replace('"','')
                else:

                    headline_text = chatGPT(prompt = headline_prompt, model='gpt-4').strip('"').strip("'")


                if template == 5 :
                    tag_line = chatGPT(f'write a tag line for {topic} in language {lang}, short and consice, to drive action. For example "⛔ Never Ignore These ⛔"\ndont mention the topic explicitly, rather drive action').strip('"').strip("'").strip("!")
                else : tag_line = ''

                html_content = save_html(
                    headline = headline_text ,

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
