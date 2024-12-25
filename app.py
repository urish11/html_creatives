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

def upload_pil_image_to_s3(image, bucket_name="image-script", aws_access_key_id=AWS_ACCESS_KEY_ID,
                           aws_secret_access_key=AWS_SECRET_ACCESS_KEY, object_name='',
                           region_name='us-east-1', image_format='PNG'):
    """
    Upload an in-memory PIL image to an S3 bucket.

    :param image: PIL Image object or URL to upload
    :param bucket_name: Bucket to upload to
    :param aws_access_key_id: AWS access key ID
    :param aws_secret_access_key: AWS secret access key
    :param object_name: S3 object name (path in the bucket)
    :param region_name: AWS region where the bucket is located (default is 'us-east-1')
    :param image_format: Format to save the image as before uploading (default is 'JPEG')
    :return: URL of the uploaded image if successful, else None
    """
    # Generate a random object name if not provided
    if not object_name:
        object_name = "".join(random.choices(string.ascii_letters + string.digits, k=8)) + ".jpg"

    # Check if input is a URL, file path, or PIL Image
    if isinstance(image, str):
        if os.path.isfile(image):  # Local file path
            img = Image.open(image)
        elif image.startswith('http://') or image.startswith('https://'):  # URL
            image_down = requests.get(image, allow_redirects=True)
            img = Image.open(BytesIO(image_down.content))
        else:
            raise ValueError("Invalid image input. Must be a URL, file path, or PIL Image object.")
    elif isinstance(image, Image.Image):  # PIL Image
        img = image
    else:
        raise ValueError("Invalid image input. Must be a URL, file path, or PIL Image object.")


    # Convert the image to a bytes object
    output_byte_arr = BytesIO()
    img.save(output_byte_arr, format=image_format)
    output_byte_arr.seek(0)

    # Create an S3 client with the provided credentials
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

    # Upload the image
    try:
        s3_client.put_object(Bucket=bucket_name, Key=object_name, Body=output_byte_arr,
                             ContentType=f'image/{image_format.lower()}')
    except NoCredentialsError:
        print("Credentials not available")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    # Return the URL of the uploaded image
    url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}"
    print(f"Uploaded Image URL: {url}")
    return url


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
                "prompt": f"{prompt}" ,
                "model": "black-forest-labs/FLUX.1-schnell-Free",
                "steps": 4,
                "n": 1,
                "height": 480*2,
                "width": 480*2,
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
	                    image = Image.open(requests.get(image_url, stream=True).raw)
	                   # st.image(image, caption=f"Generated Image for '{topic}'")

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
	except Exception as e:
		print(f'error try {e}')            
