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


# --------------------------------------------
# Load Secrets
# --------------------------------------------
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
S3_BUCKET_NAME = st.secrets["S3_BUCKET_NAME"]
AWS_REGION = st.secrets.get("AWS_REGION", "us-east-1")
GPT_API_KEY = st.secrets["GPT_API_KEY"]


# --------------------------------------------
# Utility Functions
# --------------------------------------------

def upload_pil_image_to_s3(image, bucket_name=S3_BUCKET_NAME, aws_access_key_id=AWS_ACCESS_KEY_ID,
                           aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name='us-east-1', image_format='PNG'):
    try:
        object_name = "".join(random.choices(string.ascii_letters + string.digits, k=8)) + ".png"
        output_byte_arr = BytesIO()
        image.save(output_byte_arr, format=image_format)
        output_byte_arr.seek(0)

        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

        s3_client.put_object(Bucket=bucket_name, Key=object_name, Body=output_byte_arr, ContentType=f'image/{image_format.lower()}')
        url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}"
        return url
    except NoCredentialsError:
        st.error("AWS credentials not available")
        return None
    except Exception as e:
        st.error(f"An error occurred while uploading: {e}")
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
    url = "https://api.together.xyz/v1/images/generations"
    payload = {
        "prompt": f"{prompt}",
        "model": "black-forest-labs/FLUX.1-schnell-Free",
        "steps": 4,
        "n": 1,
        "height": 960,
        "width": 960
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {IMGBB_API_KEY}"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["data"][0]["url"]
    else:
        st.error("Error generating image.")
        return None


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
        <title>Ad</title>
    </head>
    <body>
        <h1>{headline}</h1>
        <img src="{image_url}" alt="Generated Image" width="500">
        <p>{main_text}</p>
        <button>{cta_text}</button>
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
                    image = Image.open(requests.get(image_url, stream=True).raw)
                    st.image(image, caption=f"Generated Image for '{topic}'")

                    # Step 3: Upload to S3
                    s3_url = upload_pil_image_to_s3(
                        image=image,
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
