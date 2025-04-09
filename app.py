# pip install httpx aioboto3 openai anthropic google-generativeai

import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
# import boto3  # Keep for sync reference if needed, but use aioboto3
import aioboto3 # <--- New async version
from botocore.exceptions import NoCredentialsError
import random
import string
import requests # Keep for sync functions or replace where possible
import httpx # <--- New async HTTP client
from google import genai
# import anthropic # Keep for sync reference if needed, but use async client
from anthropic import AsyncAnthropic # <--- New async version
import json
import base64
import os
import time
from playwright.sync_api import sync_playwright # Playwright remains sync for now
from tempfile import NamedTemporaryFile
import re
import math
# from google_images_search import GoogleImagesSearch # Keep sync version for now, wrap if needed
import logging
# import openai # Keep for sync reference if needed, but use async client
from openai import AsyncOpenAI # <--- New async version
import asyncio # <--- Core async library

# Configure logging (optional but helpful)
# logging.basicConfig(
#     format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
#     level=logging.DEBUG
# )
# logger = logging.getLogger(__name__)

# --- Secrets ---
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
S3_BUCKET_NAME = st.secrets["S3_BUCKET_NAME"]
AWS_REGION = st.secrets.get("AWS_REGION", "us-east-1")
GPT_API_KEY = st.secrets["GPT_API_KEY"] # This seems to be OpenAI key based on usage
FLUX_API_KEY = st.secrets["FLUX_API_KEY"]
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") # Ensure it's a list or handle single key
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = st.secrets.get("GPT_API_KEY") # Explicit OpenAI key

# --- Initialize Async Clients (outside functions for reuse) ---
# Use context managers (async with) inside functions where possible for specific requests
# Or create global clients if managing lifecycle carefully (less recommended in Streamlit)
async_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
async_anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
# google-generativeai doesn't have a stable async client yet (as of mid-2024),
# but we can wrap its sync calls or use httpx for direct API calls if needed.
# For simplicity, we might wrap the sync gemini_text_lib call.


# --- Utility Functions (Keep sync for now unless they become bottlenecks) ---
def shift_left_and_pad(row):
    valid_values = [x for x in row if pd.notna(x)]
    padded_values = valid_values + [''] * (len(st.session_state.get('image_cols',[])) - len(valid_values)) # Use state
    return pd.Series(padded_values[:len(st.session_state.get('image_cols',[]))])

def log_function_call(func):
    def wrapper(*args, **kwargs):
        # logger.info(f"CALL: {func.__name__} - args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        # logger.info(f"RETURN: {func.__name__} -> {result}")
        return result
    return wrapper

def play_sound(audio_file_path):
    try:
        with open(audio_file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
            """
            st.markdown(md, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Audio file not found: {audio_file_path}")
    except Exception as e:
        st.error(f"Error playing sound: {e}")

def install_playwright_browsers():
    try:
        print("Attempting to install Playwright browsers...")
        os.system('playwright install-deps')
        os.system('playwright install chromium')
        print("Playwright browsers installation commands executed.")
        # Add a check if possible, e.g., try launching playwright briefly
        return True
    except Exception as e:
        st.error(f"Failed to install Playwright browsers: {str(e)}")
        return False

# --- Async Network Functions ---

# @log_function_call # Decorator needs adjustment for async if logging awaited results
async def async_upload_pil_image_to_s3(
    image,
    bucket_name,
    aws_access_key_id,
    aws_secret_access_key,
    object_name='',
    region_name='us-east-1',
    image_format='PNG'
):
    """
    Upload a PIL image to S3 asynchronously using aioboto3.
    """
    try:
        session = aioboto3.Session()
        # Use async with for context management of the client
        async with session.client(
            's3',
            aws_access_key_id=aws_access_key_id.strip(),
            aws_secret_access_key=aws_secret_access_key.strip(),
            region_name=region_name.strip()
        ) as s3_client:

            if not object_name:
                object_name = f"image_{int(time.time())}_{random.randint(1000, 9999)}.{image_format.lower()}"

            img_byte_arr = BytesIO()
            # image.save is sync, run in thread executor if it becomes a bottleneck
            await asyncio.to_thread(image.save, img_byte_arr, format=image_format)
            # image.save(img_byte_arr, format=image_format)
            img_byte_arr.seek(0)

            await s3_client.put_object(
                Bucket=bucket_name.strip(),
                Key=object_name,
                Body=img_byte_arr,
                ContentType=f'image/{image_format.lower()}'
            )

            url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}"
            return url

    except NoCredentialsError:
        st.error("AWS credentials not found. Please configure them.")
        return None
    except Exception as e:
        st.error(f"Error in async S3 upload: {str(e)}")
        # logger.exception("Async S3 Upload Error") # Log detailed traceback
        return None

async def async_gemini_text(
    prompt: str,
    api_key: str = None, # Will select random inside if None
    model_id: str = "gemini-1.5-flash", # Updated model name
    api_endpoint: str = "generateContent"
) -> str | None:
    global is_pd_policy, predict_policy # Access global state

    if is_pd_policy : prompt += predict_policy

    if api_key is None:
        # Ensure GEMINI_API_KEY is a list in secrets
        api_keys = st.secrets.get("GEMINI_API_KEY", [])
        if not api_keys:
             st.error("Error: GEMINI_API_KEY not found in secrets or is empty.")
             return None
        if isinstance(api_keys, str): # Handle if it's accidentally a single string
             api_keys = [api_keys]
        api_key = random.choice(api_keys)


    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:{api_endpoint}?key={api_key}"

    headers = {
        "Content-Type": "application/json",
    }

    request_data = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                ],
            },
        ],
        "generationConfig": {
            "responseMimeType": "text/plain",
             # Add other configs like temperature if needed
             "temperature": 0.7,
        },
    }

    try:
        # Use httpx for async request
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                api_url,
                headers=headers,
                json=request_data,
            )
            response.raise_for_status() # Raise an exception for bad status codes

            response_json = response.json()
            # st.text(response_json) # Maybe log instead of printing to UI during async task

            # Safer access to nested structure
            candidates = response_json.get('candidates', [])
            if candidates and 'content' in candidates[0] and 'parts' in candidates[0]['content']:
                parts = candidates[0]['content']['parts']
                if parts and 'text' in parts[0]:
                    return parts[0]['text'].replace('```','')

            st.warning(f"Unexpected response structure from Gemini: {response_json}")
            return None

    except httpx.RequestError as e:
        st.error(f"HTTP Error calling Gemini API: {e}")
        # logger.exception("Gemini API HTTP Error")
        return None
    except httpx.HTTPStatusError as e:
         st.error(f"Gemini API returned error status {e.response.status_code}: {e.response.text}")
         # logger.exception(f"Gemini API Status Error {e.response.status_code}")
         return None
    except Exception as e:
        st.error(f"An unexpected error occurred in async_gemini_text: {e}")
        # logger.exception("Unexpected Gemini Error")
        return None

# Wrap the synchronous google.genai library call
async def async_gemini_text_lib(prompt, model='gemini-1.5-flash'):
    global is_pd_policy, predict_policy # Access global state
    if is_pd_policy : prompt += predict_policy

    api_keys = st.secrets.get("GEMINI_API_KEY", [])
    if not api_keys:
        st.error("Error: GEMINI_API_KEY not found in secrets or is empty.")
        return None
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    api_key = random.choice(api_keys)

    try:
        # Configure the library client (do this once if possible, but random key requires it here)
        genai.configure(api_key=api_key)
        # Get the model
        gemini_model = genai.GenerativeModel(model)

        # Run the synchronous call in a thread
        response = await asyncio.to_thread(
            gemini_model.generate_content,
            contents=prompt
        )
        return response.text
    except Exception as e:
        st.error(f'async_gemini_text_lib error: {e}')
        # logger.exception("async_gemini_text_lib error")
        await asyncio.sleep(4) # Async sleep
        return None


async def async_chatGPT(prompt, model="gpt-4o", temperature=1.0, reasoning_effort=''):
    global is_pd_policy, predict_policy # Access global state
    if is_pd_policy: prompt += predict_policy

    try:
        # Use the async client
        messages = [{"role": "user", "content": prompt}]
        # Note: OpenAI API doesn't have a direct 'reasoning' parameter like this.
        # The older /v1/responses endpoint might be custom or deprecated.
        # Using the standard chat completions endpoint:
        completion_params = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
        }
        # Only include temperature if not default (or handle 0 case specifically if needed)
        # The API usually defaults temperature around 0.7-1.0 if not provided.
        # if temperature != 1.0: # Or some other default check
        #     completion_params['temperature'] = temperature

        response = await async_openai_client.chat.completions.create(**completion_params)

        content = response.choices[0].message.content
        return content

    except Exception as e:
        st.error(f"Error in async_chatGPT: {str(e)}")
        # logger.exception("async_chatGPT Error")
        # Attempt to log response body if available in exception
        # if hasattr(e, 'response') and hasattr(e.response, 'text'):
        #     st.text(f"Error Response Body: {e.response.text}")
        return None

async def async_claude(prompt, model="claude-3-sonnet-20240229", temperature=1, is_thinking=False, max_retries=10):
    global is_pd_policy, predict_policy # Access global state
    if is_pd_policy : prompt += predict_policy
    tries = 0

    while tries < max_retries:
        try:
            # Use the async client
            request_params = {
                 "model": model,
                 "max_tokens": 4000, # Reduced max_tokens slightly
                 "temperature": temperature,
                 "messages": [{"role": "user", "content": prompt}]
                 # Anthropic doesn't have 'thinking' or 'top_p' in standard messages API
                 # If using older APIs or specific beta features, adjust accordingly.
            }

            # Add thinking if applicable (check Anthropic documentation for current support)
            # if is_thinking:
            #     request_params["thinking"] = {"type": "enabled", "budget_tokens": 16000} # Hypothetical

            message = await async_anthropic_client.messages.create(**request_params)

            # Accessing content depends on the response structure
            if message.content and isinstance(message.content, list):
                 # Standard Claude 3 structure
                 text_blocks = [block.text for block in message.content if block.type == 'text']
                 return "\n".join(text_blocks) if text_blocks else None
            else:
                 st.warning(f"Unexpected Claude response structure: {message}")
                 return None # Or try accessing older structures if needed

        except Exception as e:
            st.error(f"Error in async_claude (Try {tries+1}/{max_retries}): {e}")
            # logger.exception(f"async_claude Error (Try {tries+1})")
            tries += 1
            if tries >= max_retries:
                 st.error("Max retries reached for Claude.")
                 return None
            await asyncio.sleep(5) # Async sleep


async def async_gen_flux_img(prompt, height=784, width=960):
    """
    Generate images using FLUX model from the Together.xyz API asynchronously.
    """
    url = "https://api.together.xyz/v1/images/generations"
    payload = {
        "prompt": prompt,
        "model": "black-forest-labs/FLUX.1-schnell", # Use schnell or dev based on need
        "steps": 4, # Schnell uses fewer steps
        "n": 1,
        "height": height,
        "width": width,
        "response_format": "url", # Ensure URL is requested
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {random.choice(st.secrets['FLUX_API_KEY'])}" # Assumes FLUX_API_KEY is list
    }
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client: # Increased timeout
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                if data.get("data") and len(data["data"]) > 0 and data["data"][0].get("url"):
                    return data["data"][0]["url"]
                else:
                    st.warning(f"Flux response missing URL: {data}")
                    return None # Or retry
        except httpx.RequestError as e:
            st.warning(f"Flux HTTP Request Error (Try {retries+1}): {e}")
        except httpx.HTTPStatusError as e:
             # Check for NSFW or other specific errors
             if "NSFW" in e.response.text:
                 st.warning(f"Flux NSFW content detected for prompt: {prompt[:50]}...")
                 return None # Don't retry NSFW
             st.warning(f"Flux HTTP Status Error {e.response.status_code} (Try {retries+1}): {e.response.text}")
        except Exception as e:
            st.warning(f"Flux Unexpected Error (Try {retries+1}): {e}")

        retries += 1
        if retries >= max_retries:
            st.error(f"Max retries reached for Flux image generation.")
            return None
        await asyncio.sleep(random.uniform(3, 7)) # Exponential backoff might be better


async def async_gen_gemini_image(prompt, max_retries=5):
    api_keys = st.secrets.get("GEMINI_API_KEY", [])
    if not api_keys:
        st.error("GEMINI_API_KEY missing.")
        return None
    if isinstance(api_keys, str): api_keys = [api_keys]

    trys = 0
    while trys < max_retries :
        api = random.choice(api_keys)
        # Use the correct model name for image generation (check Google AI Studio/Docs)
        # It might be 'gemini-pro-vision' for input, but generation uses different models.
        # Let's assume a hypothetical image generation endpoint model or use Vertex AI's Imagen.
        # Using the Imagen 2 endpoint via Vertex AI SDK might be more robust.
        # Sticking to the user's provided endpoint for now, assuming 'gemini-1.0-pro-exp-image-generation' exists.
        # UPDATE: Found the user used `gemini-2.0-flash-exp-image-generation`. Keep that.
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent?key={api}"

        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}], # Simplified contents
            "generationConfig": {
                "temperature": 0.65, "topK": 40, "topP": 0.95,
                "maxOutputTokens": 1024, # Reduced for image gen focus
                "responseMimeType": "application/json", # Expect JSON for image data
                "responseModalities": ["image"] # Request only image
            },
             "safety_settings": [ # Add safety settings if needed
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, headers=headers, json=data) # Use json=data

                if response.status_code == 200:
                    res_json = response.json()
                    candidates = res_json.get('candidates')
                    if candidates and 'content' in candidates[0] and 'parts' in candidates[0]['content']:
                         parts = candidates[0]['content']['parts']
                         # Find the image part
                         image_part = next((p for p in parts if 'inlineData' in p and p['inlineData'].get('mimeType','').startswith('image/')), None)
                         if image_part:
                             image_b64 = image_part['inlineData']['data']
                             image_data = base64.b64decode(image_b64) # Use standard b64decode
                             # Convert sync BytesIO/Image.open to thread if needed
                             pil_image = await asyncio.to_thread(Image.open, BytesIO(image_data))
                             # pil_image = Image.open(BytesIO(image_data))
                             return pil_image
                         else:
                            st.warning(f"Gemini Image Gen: Image data not found in response parts. {res_json}")
                    else:
                        st.warning(f"Gemini Image Gen: Unexpected response structure. {res_json}")
                        # Check for safety blocks
                        if res_json.get("promptFeedback", {}).get("blockReason"):
                            st.warning(f"Gemini image blocked: {res_json['promptFeedback']['blockReason']}")
                            return None # Don't retry blocked content


                else:
                    st.warning(f"Gemini Image Gen Error {response.status_code} (Try {trys+1}): {response.text}")

        except httpx.RequestError as e:
             st.warning(f"Gemini Image Gen HTTP Request Error (Try {trys+1}): {e}")
        except httpx.HTTPStatusError as e:
             st.warning(f"Gemini Image Gen HTTP Status Error {e.response.status_code} (Try {trys+1}): {e.response.text}")
        except Exception as e:
             st.warning(f"Gemini Image Gen Unexpected Error (Try {trys+1}): {e}")
             # logger.exception("Gemini Image Gen Unexpected Error")


        trys += 1
        if trys >= max_retries:
             st.error(f"Max retries ({max_retries}) reached for Gemini image generation.")
             return None
        await asyncio.sleep(random.uniform(3, 7)) # Sleep before retrying

    return None # Should not be reached if loop finishes


async def async_create_dalle_variation(image_url, count):
    """
    Asynchronously downloads an image, converts to PNG, and creates DALL-E variations.
    """
    try:
        # 1. Async Download
        headers = {"User-Agent": "Mozilla/5.0 ..."} # Your user agent
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(image_url, headers=headers, follow_redirects=True)
            resp.raise_for_status()
            img_data = await resp.aread() # Read bytes asynchronously

        # 2. Process Image (Sync parts wrapped in thread)
        def process_image_sync(image_bytes):
            img = Image.open(BytesIO(image_bytes))
            png_buffer = BytesIO()
            img.save(png_buffer, format="PNG")
            png_buffer.seek(0)

            # Check size and resize if necessary
            if png_buffer.getbuffer().nbytes > 4 * 1024 * 1024:
                img = img.resize((512, 512)) # DALL-E variation input preferred size
                png_buffer = BytesIO()
                img.save(png_buffer, format="PNG")
                png_buffer.seek(0)
            return png_buffer.getvalue() # Return bytes

        png_bytes = await asyncio.to_thread(process_image_sync, img_data)

        # 3. Call Async DALL-E Client
        response = await async_openai_client.images.create_variation(
            image=png_bytes, # Pass bytes directly
            n=count,
            size="512x512" # Variations often limited in size options
        )
        return response.data # List of image objects (containing URLs)

    except httpx.HTTPStatusError as e:
         st.error(f"Error downloading image for DALL-E variation ({e.response.status_code}): {image_url}")
         return None
    except Exception as e:
        st.error(f"Error generating async DALL-E variation: {e}")
        # logger.exception("Async DALL-E Variation Error")
        return None


# --- Sync functions (to be wrapped or kept if not I/O bound) ---
# Keep sync google search for now, wrap with to_thread if becomes blocking
# @log_function_call
def fetch_google_images(query, num_images=3, max_retries=5):
    # This function uses a synchronous library (google_images_search)
    # It *should* be wrapped in asyncio.to_thread if called from async context
    # to avoid blocking the event loop.
    from google_images_search import GoogleImagesSearch # Import inside or ensure global

    # Ensure API Keys and CX are lists or handle single values
    api_keys = st.secrets.get("GOOGLE_API_KEY", [])
    google_cx = st.secrets.get("GOOGLE_CX")

    if not api_keys or not google_cx:
        st.error("GOOGLE_API_KEY or GOOGLE_CX missing from secrets.")
        return []
    if isinstance(api_keys, str): api_keys = [api_keys]


    for trial in range(max_retries):
        terms_list = query.split('~')
        res_urls = set() # Use a set to avoid duplicates inherently
        try:
            for term in terms_list:
                API_KEY = random.choice(api_keys)
                CX = google_cx

                gis = GoogleImagesSearch(API_KEY, CX) # Instantiate per request (or manage state)

                search_params = {
                    'q': term.strip(),
                    'num': num_images,
                    'safe': 'medium', # Consider adding safety filter
                    # 'imgType': 'photo', # Optional: filter by type
                }

                # This is the synchronous part
                gis.search(search_params=search_params)
                image_urls = [result.url for result in gis.results()]
                res_urls.update(image_urls) # Add found URLs to the set

            return list(res_urls) # Return unique URLs as a list

        except Exception as e:
            st.warning(f"Error fetching Google Images for '{query}' (Try {trial+1}): {e}")
            if trial < max_retries - 1:
                time.sleep(random.uniform(3, 5)) # Sync sleep
            else:
                st.error(f"Max retries reached for Google Images search: {query}")
                return [] # Return empty list on final failure
    return [] # Should be reached only if loop finishes without success


# --- HTML Generation (Sync) ---
# @log_function_call
def save_html(headline, image_url, cta_text, template, tag_line='', output_file="advertisement.html"):
    # (Keep your existing HTML template logic here - it's sync CPU work)
    # ... (your existing template logic) ...
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
                   width: 1000px; /* Fixed width for screenshot */
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
                   /* background-size: contain; */ /* Cover usually better */
                   text-align: center;
                   box-sizing: border-box; /* Include padding in size */
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
               <div class="ad-title">{headline}</div>
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
                body {{
                    font-family: 'Gisha', sans-serif;
                    font-weight: 550;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    width: 1000px; /* Fixed width */
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
                    box-sizing: border-box;
                }}
                .ad-title {{
                    font-size: 3.2em;
                    color: #333;
                    background-color: white;
                    padding: 20px;
                    text-align: center;
                    flex: 0 0 auto; /* Don't flex grow/shrink, height based on content */
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 15%; /* Ensure some minimum height */
                    box-sizing: border-box;
                }}
                .ad-image {{
                    flex: 1 1 auto; /* Allow image to fill remaining space */
                    background: url('{image_url}') no-repeat center center/cover;
                    /* background-size: fill; */ /* cover is usually better */
                    position: relative;
                    min-height: 0; /* Allow shrinking if needed */
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
                 .cta-button:hover {{
                   background-color: #E64A19;
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
                    width: 1000px; /* Fixed width */
                }}
                .container {{
                    position: relative;
                    width: 1000px;
                    height: 1000px;
                    margin: 0;
                    padding: 0;
                    overflow: hidden;
                    box-shadow: 0 0 20px rgba(0,0,0,0.2);
                    box-sizing: border-box;
                }}
                .image {{
                    display: block; /* Remove extra space below image */
                    width: 100%; /* Use percentage */
                    height: 100%; /* Use percentage */
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
                    word-wrap: break-word; /* Allow wrapping */
                    overflow-wrap: break-word; /* Better wrapping */
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
                    box-shadow: 0 5px 15px rgba(255,107,107,0.4); /* Check color contrast */
                    white-space: nowrap; /* Prevent button text wrapping */
                }}
                 /* Style for template 7 (no button) */
                .c1ta-button {{ display: none !important; }}

                .cta-button:hover {{
                    background: #4ECDC4; /* Ensure good contrast */
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
                <img src="{image_url}" class="image" alt="Ad Image">
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
        if template == 7:
             # Hide the button for template 7 specifically
             html_template = html_template.replace('<button class="cta-button">', '<button class="cta-button c1ta-button">', 1)

    elif template == 4:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template 4</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap');
                /* @font-face for Calibre needs a valid src path if used */
                body {{
                    margin: 0; padding: 0;
                    display: flex; justify-content: center; align-items: center;
                    height: 100vh; background-color: #F4F4F4;
                     width: 1000px; /* Fixed width */
                }}
                .container {{
                    position: relative; width: 1000px; height: 1000px;
                    background-image: url('{image_url}');
                    background-size: cover; background-position: center;
                    border-radius: 10px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
                    box-sizing: border-box;
                }}
                .text-overlay {{
                    position: absolute; width: 95%;
                    background-color: rgba(255, 255, 255, 1);
                    padding: 30px; border-radius: 10px;
                    top: 50%; left: 50%; transform: translate(-50%, -50%);
                    text-align: center; box-sizing: border-box;
                }}
                .small-text {{
                    font-size: 36px; font-weight: bold; color: #333;
                    margin-bottom: 10px;
                    font-family: 'Arial', sans-serif; /* Fallback font */
                }}
                .primary-text {{
                    font-size: 60px; font-weight: bold; color: #FF8C00;
                    font-family: 'Montserrat', sans-serif;
                    line-height: 1.2;
                    text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000;
                    word-wrap: break-word; overflow-wrap: break-word; /* Allow wrapping */
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
    elif template == 5:
        # CORRECTED: Directly use the intended HTML structure in the f-string.
        # The headline variable might contain HTML (like the <span>), which is fine here.
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template 5</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@100..900&display=swap');
                @import url('https://fonts.googleapis.com/css2?family=Noto+Color+Emoji&display=swap');
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    width: 1000px; height: 1000px;
                    margin: 0 auto;
                    font-family: 'Outfit', sans-serif;
                    display: flex;
                    justify-content: center; align-items: center;
                }}
                .container {{
                    width: 1000px; height: 1000px;
                    display: flex; flex-direction: column;
                    position: relative;
                    overflow: hidden;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                }}
                .image-container {{
                    width: 100%; height: 60%;
                    background-color: #f0f0f0;
                    display: flex; align-items: center; justify-content: center;
                    overflow: hidden;
                }}
                .image-container img {{
                    width: 100%; height: 100%;
                    object-fit: cover;
                    display: block;
                }}
                .content-container {{
                    width: 100%; height: 40%;
                    background-color: #121421;
                    display: flex; flex-direction: column;
                    align-items: center; justify-content: center;
                    padding: 2rem; gap: 1.5rem;
                    text-align: center;
                    box-sizing: border-box;
                }}
                .main-text {{
                    color: white; font-size: 3.0rem;
                    font-weight: 700;
                    max-width: 90%;
                    word-wrap: break-word; overflow-wrap: break-word;
                }}
                /* Style for highlighted span */
                .highlight {{
                    color: #FFFF00; /* Yellow */
                    font-style: italic;
                    font-weight: 900; /* Bold */
                }}
                .cta-button {{
                    background-color: #ff0000; color: white;
                    padding: 1rem 2rem; font-size: 3.0rem;
                    font-weight: 700; font-family: 'Outfit', sans-serif;
                    border: none; font-style: italic; border-radius: 8px;
                    cursor: pointer; transition: background-color 0.3s ease;
                    white-space: nowrap;
                }}
                .cta-button:hover {{ background-color: #cc0000; }}
                .intersection-rectangle {{
                    position: absolute;
                    max-width: 80%;
                    min-width: 30%;
                    height: 80px; background-color: #121421;
                    left: 50%; transform: translateX(-50%);
                    top: calc(60% - 40px);
                    border-radius: 10px;
                    display: flex; align-items: center; justify-content: center;
                    padding: 0 30px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                    z-index: 10;
                }}
                .rectangle-text {{
                    font-family: 'Noto Color Emoji', 'Outfit', sans-serif;
                    color: #66FF00; font-weight: 700;
                    text-align: center; font-size: 40px;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="image-container">
                    <img src="{image_url}" alt="Advertisement Image">
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
        # REMOVED: The problematic .replace call is no longer needed:
        # html_template = html_template.replace('<h1 class="main-text" dangerouslySetInnerHTML={{ __html: headline }}></h1>', f'<h1 class="main-text">{headline}</h1>')
    elif template == 41: # Text overlay at top
        html_template = f"""
        <!DOCTYPE html><html lang="en"><head>
            <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template 41</title><style>
                @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap');
                body {{ margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #F4F4F4; width: 1000px; }}
                .container {{ position: relative; width: 1000px; height: 1000px; background-image: url('{image_url}'); background-size: cover; background-position: center; border-radius: 10px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); box-sizing: border-box; }}
                .text-overlay {{ position: absolute; width: 95%; background-color: rgba(255, 255, 255, 1); padding: 30px; border-radius: 10px; top: 15%; left: 50%; transform: translate(-50%, -50%); text-align: center; box-sizing: border-box; }}
                .small-text {{ font-size: 36px; font-weight: bold; color: #333; margin-bottom: 10px; font-family: 'Arial', sans-serif; }}
                .primary-text {{ font-size: 60px; font-weight: bold; color: #FF8C00; font-family: 'Montserrat', sans-serif; line-height: 1.2; text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000; word-wrap: break-word; overflow-wrap: break-word; }}
            </style></head><body><div class="container"><div class="text-overlay">
                <div class="small-text">{cta_text}</div>
                <div class="primary-text">{headline}</div>
            </div></div></body></html>
        """
    elif template == 42: # Text overlay at bottom
        html_template = f"""
        <!DOCTYPE html><html lang="en"><head>
            <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template 42</title><style>
                @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap');
                 body {{ margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #F4F4F4; width: 1000px; }}
                .container {{ position: relative; width: 1000px; height: 1000px; background-image: url('{image_url}'); background-size: cover; background-position: center; border-radius: 10px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); box-sizing: border-box; }}
                .text-overlay {{ position: absolute; width: 95%; background-color: rgba(255, 255, 255, 1); padding: 30px; border-radius: 10px; /* Changed top to bottom */ bottom: 5%; left: 50%; transform: translate(-50%, 0); text-align: center; box-sizing: border-box; }}
                .small-text {{ font-size: 36px; font-weight: bold; color: #333; margin-bottom: 10px; font-family: 'Arial', sans-serif; }}
                .primary-text {{ font-size: 60px; font-weight: bold; color: #FF8C00; font-family: 'Montserrat', sans-serif; line-height: 1.2; text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000; word-wrap: break-word; overflow-wrap: break-word; }}
            </style></head><body><div class="container"><div class="text-overlay">
                <div class="small-text">{cta_text}</div>
                <div class="primary-text">{headline}</div>
            </div></div></body></html>
        """
    elif template == 6: # Image only
        html_template = f"""
        <!DOCTYPE html><html lang="en"><head>
            <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template 6</title><style>
                body {{ margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #F4F4F4; width: 1000px; }}
                .container {{ position: relative; width: 1000px; height: 1000px; background-image: url('{image_url}'); background-size: cover; background-position: center; border-radius: 10px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); overflow: hidden; /* Ensure image stays within bounds */ }}
                img {{ display: block; width: 100%; height: 100%; object-fit: cover; }} /* Alternative: Use img tag */
            </style></head><body>

             
            </body></html>
        """
    else:
        st.warning(f'HTML template {template} not found')
        html_template = f"<html><body><p>Template {template} not found. Image: <img src='{image_url}' width='300'></p></body></html>"
    if template == 7:
        html_template = html_template.replace('<button class="cta-button">', '<button class="cta-button c1ta-button">', 1)

    return html_template
    # No need to save to file, just return the string
    # with open(output_file, "w", encoding='utf-8') as file:
    #     file.write(html_template)
    return html_template


# --- Playwright Screenshot (Keep Sync for now) ---
# @log_function_call
def capture_html_screenshot_playwright(html_content):
    if 'playwright_installed' not in st.session_state or not st.session_state.playwright_installed:
        st.error("Playwright browsers not installed or installation check failed.")
        return None

    try:
        with sync_playwright() as p:
            # Use try-except for browser launch robustness
            try:
                browser = p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu', '--single-process'] # Added args
                )
            except Exception as launch_error:
                st.error(f"Playwright Chromium launch failed: {launch_error}")
                # Attempt firefox as fallback?
                # try:
                #     browser = p.firefox.launch(headless=True)
                # except Exception as ff_launch_error:
                #     st.error(f"Playwright Firefox launch also failed: {ff_launch_error}")
                #     return None
                return None

            page = browser.new_page(viewport={'width': 1000, 'height': 1000}) # Match body/container size

            # Use temp file for reliable rendering vs data URI limitations
            with NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
                f.write(html_content)
                temp_html_path = f.name

            try:
                # Go to the local HTML file
                await_nav = page.goto(f'file://{temp_html_path}', wait_until='networkidle', timeout=15000) # Increased timeout, wait for network idle

                # Add explicit waits if necessary (e.g., for fonts or complex JS)
                # page.wait_for_load_state('domcontentloaded')
                # page.wait_for_timeout(1000) # Small delay for rendering stabilization

                screenshot_bytes = page.screenshot(type='png') # Specify PNG

            except Exception as page_error:
                 st.error(f"Playwright page navigation/screenshot error: {page_error}")
                 screenshot_bytes = None
            finally:
                # Ensure browser is closed and temp file is deleted
                browser.close()
                if os.path.exists(temp_html_path):
                    os.unlink(temp_html_path)

            if screenshot_bytes:
                return Image.open(BytesIO(screenshot_bytes))
            else:
                return None

    except Exception as e:
        st.error(f"Overall Playwright screenshot capture error: {str(e)}")
        # logger.exception("Playwright capture error")
        return None


# --- Main Async Orchestration Logic ---
async def generate_single_image_task(task_data):
    """
    Async function to handle generation for one image based on task_data.
    Returns a dictionary like {'url': ..., 'template': ..., 'source': ..., 'dalle_generated': ...} or None on failure.
    """
    topic = task_data["topic"]
    lang = task_data["lang"]
    template_input = task_data["template_input"] # Raw template string from row
    iteration = task_data["iteration"]
    # global_topic = task_data["global_topic"] # Original topic before potential modifications
    enhance_input = task_data["enhance_input"]
    is_google_search = task_data["is_google_search"]
    google_urls = task_data.get("google_urls", []) # URLs if pre-fetched for google


    try:
        if is_google_search:
            if iteration < len(google_urls):
                 return {
                    'url': google_urls[iteration],
                    'selected': False, # Default state
                    'template': template_input, # Keep original template string for later processing
                    'source': 'google',
                    'dalle_generated': False
                }
            else:
                st.warning(f"Requested google image index {iteration} out of bounds for {topic}")
                return None # No more URLs for this iteration

        # --- Non-Google Generation ---
        current_template_str = template_input
        if ',' in template_input: # Random selection if multiple templates
            current_template_str = random.choice([t.strip() for t in template_input.split(",") if t.strip()])
        elif "*" in template_input: # Handle '*' for new prompt logic
             current_template_str = template_input.replace("*", "").strip()
        # Ensure it's not empty
        if not current_template_str:
            st.warning(f"Empty template string derived for topic {topic}. Skipping.")
            return None


        # Enhance topic if requested (do this once per topic ideally, but here per task if needed)
        current_topic = topic
        if enhance_input:
            # Use async version of chatGPT
            enhanced = await async_chatGPT(f"write this as more commercially attractive for ad promoting an article in {int(topic.count(' '))+1} words, 1 best option\n\n {topic}")
            if enhanced:
                current_topic = enhanced.strip().strip('"')


        # --- Gemini Generation ---
        if 'gemini' in current_template_str.lower():
            gemini_prompt = None
            headline_temp = None # For gemini6

            # Use async text generation for prompts
            if current_template_str == 'gemini':
                 gemini_prompt = await async_chatGPT(f"write short prompt for\ngenerate square image promoting '{current_topic}' in language {lang} {random.choice(['use photos',''])}. add a CTA button with 'Learn More Here >>' in appropriate language\nshould be low quality and very enticing and alerting\nstart with 'square image aspect ratio of 1:1 of '\n\n example output:\n\nsquare image of a concerned middle-aged woman looking at her tongue in the mirror under harsh bathroom lighting, with a cluttered counter and slightly blurry focus — big bold red text says “Early Warning Signs?” and a janky yellow button below reads “Learn More Here >>” — the image looks like it was taken on an old phone, with off angle, bad lighting, and a sense of urgency and confusion to provoke clicks.", model="gpt-4o", temperature=1.0)
            elif current_template_str == 'gemini7claude':
                gemini_prompt = await async_claude(f"""write short prompt for\ngenerate square image promoting '{current_topic}' in language {lang} . add a CTA button with 'Learn More Here >>' in appropriate language\ \nshould be low quality and very enticing and alerting, don't make specific promises like x% discount   \n\nstart with 'square image aspect ratio of 1:1 of '\n\n be specific in what is shown . return JUST the best option, no intros\nif you want to add a caption, specifically instruct it to be on the image. and be short""", is_thinking=False)
            elif current_template_str == 'geminicandid':
                 gemini_prompt = await async_claude(f"""write a image prompt of a candid unstaged photo taken of a regular joe showing off his\her {current_topic} . the image is taken with smartphone candidly. in 1-2 sentences. Describe the quality of the image looking smartphone. start with "Square photo 1:1 iphone 12 photo uploaded to reddit:"\n\nthis is for a fb ad that tries to look organic, but also make the image content intecing and somewhat perplexing, so try to be that but also draw clicks with high energy in the photo. dont make up facts like discounts! or specific prices! if you want to add a caption, specifically instruct it to be on the image. and be short in language {lang}""", is_thinking=True)
                 if gemini_prompt: gemini_prompt = gemini_prompt.replace("#","")
            elif current_template_str == 'geministock':
                 gemini_prompt = await async_chatGPT(f"write short image prompt for {current_topic}, no text on image", model="gpt-4o", temperature= 1.0)

            # Add other gemini prompt variations using await async_...

            if gemini_prompt:
                # Generate the image
                st.info(f"Generating Gemini image for: {current_topic} (Prompt: {gemini_prompt[:60]}...)")
                gemini_img_pil = await async_gen_gemini_image(gemini_prompt)

                if gemini_img_pil:
                    # Upload to S3
                    st.info(f"Uploading Gemini image for: {current_topic}")
                    gemini_image_url = await async_upload_pil_image_to_s3(
                        image=gemini_img_pil,
                        bucket_name=S3_BUCKET_NAME,
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                        region_name=AWS_REGION
                    )
                    if gemini_image_url:
                        return {
                            'url': gemini_image_url,
                            'selected': False,
                            'template': current_template_str, # Store the specific template used
                            'source': 'gemini',
                            'dalle_generated': False
                        }
                    else:
                         st.warning(f"Failed to upload Gemini image for {current_topic}")
                else:
                     st.warning(f"Failed to generate Gemini image for {current_topic}")
            else:
                 st.warning(f"Failed to generate Gemini prompt for {current_topic}, template: {current_template_str}")

        # --- Flux Generation ---
        else:
            try:
                template_int = int(current_template_str) # Convert to int for template logic
            except ValueError:
                st.warning(f"Invalid non-gemini template '{current_template_str}' for topic {topic}. Skipping.")
                return None

            image_prompt = None
            st.info(f"Generating Flux prompt for: {current_topic} (Template: {template_int})")
             # Use async text generation for prompts
            if template_int == 5:
                 rand_prompt = f"""Generate a concise visual image description (15 words MAX) for {current_topic}. Be wildly creative, curious, and push the limits of imagination—while staying grounded in real-life scenarios! Depict an everyday, highly relatable yet dramatically eye-catching scene that sparks immediate curiosity within 3 seconds. Ensure the image conveys the value of early detection (e.g., saving money, time, improving health, or education) in a sensational but simple way. The scene must feature one person, clearly illustrating the topic without confusion. Avoid surreal or abstract elements; instead, focus on relatable yet RANDOM high-energy moments from daily life. Do not include any text in the image. Your final output should be 8-13 words, written as if describing a snapshot from a camera. Make sure the offer’s value is unmistakably clear and visually intriguing"""
                 image_prompt = await async_chatGPT(rand_prompt, model='gpt-4', temperature=1.2) # Model choice? gpt-4o might be faster/cheaper
            elif template_int == 7:
                 image_prompt = await async_chatGPT(f"Generate a visual image description 50 words MAX for {current_topic}, candid moment unstaged, taken in the moment by eye witness like with a smartphone, viral reddit style, make it dramatic and visually enticing", model='gpt-4o-mini') # Use cheaper/faster model if possible
            else: # Default prompt generation
                 image_prompt = await async_chatGPT(f"Generate a visual image description 15 words MAX for {current_topic}. Be creative, show the value of the offer (saving money, time, health, etc.) in a sensational yet simplistic scene. Include one person and do not include text in the image. Output is up to 5 words. Think like a camera snapshot!", model='gpt-4o-mini', temperature=1.15)


            if image_prompt:
                st.info(f"Generating Flux image for: {current_topic} (Prompt: {image_prompt[:60]}...)")
                flux_image_url = None
                flux_prompt = f"{random.choice(['cartoony clipart of ', 'cartoony clipart of ', '', ''])}{image_prompt}"
                width, height = 960, 784 # Default Flux size

                if template_int == 5:
                    width, height = 688, 416
                    # flux_image_url = await async_gen_flux_img(flux_prompt, width=width, height=height)
                # Add Flux LoRA call if needed for specific templates (needs async version)
                # elif template_int == 7:
                #     flux_image_url = await async_gen_flux_img_lora(image_prompt) # Assuming async_gen_flux_img_lora exists

                # Default Flux call
                flux_image_url = await async_gen_flux_img(flux_prompt, width=width, height=height)


                if flux_image_url:
                    return {
                        'url': flux_image_url,
                        'selected': False,
                        'template': template_int, # Store the integer template
                        'source': 'flux',
                        'dalle_generated': False
                    }
                else:
                     st.warning(f"Failed to generate Flux image for {current_topic}")
            else:
                st.warning(f"Failed to generate Flux prompt for {current_topic}, template: {template_int}")

        # If generation failed for any reason
        return None

    except Exception as e:
        st.error(f"Error in generation task for {topic} (Iter {iteration}): {e}")
        # logger.exception(f"Error in generation task for {topic} (Iter {iteration})")
        return None


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Creative Gen", page_icon="🎨")

# Install playwright if needed (Keep this sync at the start)
if 'playwright_installed' not in st.session_state:
    with st.spinner("Checking and installing browser dependencies (one-time setup)..."):
        st.session_state.playwright_installed = install_playwright_browsers()
    if st.session_state.playwright_installed:
        st.success("Browser dependencies ready.")
    else:
        st.error("Failed to setup browser dependencies. Screenshot generation might fail.")


st.title("Creative Maker ")

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = [] # Will store list of dicts: {"topic":.. "lang":.. "images": [...]}
if 'image_cols' not in st.session_state: # For shift_left_and_pad
     st.session_state.image_cols = []
if 'cta_texts' not in st.session_state: # Cache for translations
     st.session_state.cta_texts = {}


# --- Examples Expander (Sync) ---
with st.expander(f"Click to see examples for templates ", expanded=False):
     image_list = [
         {"image": "https://image-script.s3.us-east-1.amazonaws.com/image_1744112392_1276.png", "caption": "2"},
         # ... (rest of your examples) ...
        {"image": "https://image-script.s3.us-east-1.amazonaws.com/image_1744212220_8873.png", "caption": "geministock"}
     ]
     num_columns_ex = 6
     num_images_ex = len(image_list)
     num_rows_ex = (num_images_ex + num_columns_ex - 1) // num_columns_ex
     for i in range(num_rows_ex):
        cols = st.columns(num_columns_ex)
        row_images = image_list[i * num_columns_ex : (i + 1) * num_columns_ex]
        for j, item in enumerate(row_images):
            if item:
                with cols[j]:
                    st.image(item["image"], use_container_width=True)
                    st.caption(item["caption"])


st.subheader("Enter Topics for Image Generation")
# Define default structure for data editor
default_df_data = {"topic": ["example_topic | optional context"], "count": [2], "lang": ["english"], "template": ["gemini7claude, 5, 7"]}
df_input = st.data_editor(
    pd.DataFrame(default_df_data),
    num_rows="dynamic",
    key="table_input",
    column_config={ # Optional: Improve editor experience
        "topic": st.column_config.TextColumn("Topic (use '|' for context)", help="Main topic before '|', context/keywords after '|' (e.g., for Google Search)", width="large"),
        "count": st.column_config.NumberColumn("Number of Images", min_value=1, max_value=50, step=1, default=1),
        "lang": st.column_config.TextColumn("Language", default="english"),
        "template": st.column_config.TextColumn("Template(s)", help="e.g., 2,3,5,gemini,gemini7claude. Comma-separated for random choice.", default="5, gemini7claude"),
    }
)

# --- Global Flags (affect text generation) ---
# Define these *before* they are potentially used in async functions
is_pd_policy = st.checkbox("Add PD policy text to prompts?", value=False, key="pd_policy_check")
predict_policy = """  Approved CTAs: Use calls-to-action like "Learn More" or "See Options" that clearly indicate leading to an article. Avoid CTAs like "Apply Now" or "Shop Now" that promise value not offered on the landing page.   \nProhibited Language: Do not use urgency terms ("Click now"), geographic suggestions ("Near you"), or superlatives ("Best").   \nEmployment/Education Claims: Do not guarantee employment benefits (like high pay or remote work) or education outcomes (like degrees or job placements).   \nFinancial Ad Rules: Do not guarantee loans, credit approval, specific investment returns, or debt relief. Do not offer banking, insurance, or licensed financial services.   \n"Free" Promotions: Generally avoid promoting services/products as "free". Exceptions require clarity: directing to an info article about a real free service, promoting a genuinely free course, or advertising free trials with clear terms. USE text on image, the most persuasive as you can you can add visual elements to the text to make up for the policy .the above is NOT relevent to the visual aspect of the image!  """ if is_pd_policy else ""


# --- Generation Trigger ---
auto_mode = st.checkbox("Auto mode? (Auto-selects 1 image per topic)", value=False, key="auto_mode_check")
enhance_input = st.checkbox("Enhance input topic using AI?", value=False, key="enhance_input_check")

# Create placeholder for progress bar
progress_placeholder = st.empty()

if st.button("Generate Images Async", key="generate_button"):
    st.session_state.generated_images = []  # Clear previous results
    tasks = []
    task_details = [] # To map results back: {"topic": ..., "lang": ...}
    total_images_to_generate = 0
    google_search_cache = {} # Cache google results per query

    st.info("Preparing image generation tasks...")

    # --- Phase 1: Create all tasks ---
    # ...(code to prepare tasks remains the same)...
    # Example:
    for index, row in df_input.iterrows():
        # ... (logic to determine topic, count, lang, template, is_google_search) ...

        # # Pre-fetch Google Images (Sync - Keep this before creating async tasks)
        # if is_google_search and topic_for_google not in google_search_cache:
        #      st.text(f"Fetching Google images for: {topic_for_google}")
        #      google_urls = fetch_google_images(topic_for_google, num_images=count * 2)
        #      google_search_cache[topic_for_google] = google_urls
        #      st.text(f"Found {len(google_urls)} Google images.")

        # total_images_to_generate += count

        # Create 'count' number of tasks for this row
        for i in range(count):
            task_data = {
                "topic": topic_for_gen,
                "lang": lang,
                "template_input": template_input,
                "iteration": i,
                "enhance_input": enhance_input,
                "is_google_search": is_google_search,
                "google_urls": google_search_cache.get(topic_for_google, []) if is_google_search else []
            }
            # Don't create task here, just prepare the awaitable
            # tasks.append(generate_single_image_task(task_data)) # <-- CHANGE THIS
            # Instead, prepare the coroutine object
            tasks.append(generate_single_image_task(task_data)) # We gather coroutines
            task_details.append({"topic": topic_for_gen, "lang": lang})

    st.info(f"Created {len(tasks)} generation tasks. Starting concurrent execution...")

    # --- Phase 2: Run tasks concurrently ---
    results = []
    start_time = time.time()
    progress_count = 0
    with progress_placeholder.container(): # Show progress bar here
         my_bar = st.progress(0, text=f"Generating images... 0/{total_images_to_generate}")

         try:
             # ----------------------------
             # CORRECTED LINE: Use asyncio.run() to execute the awaitable gather
             # ----------------------------
             if tasks: # Only run if there are tasks to execute
                 results = asyncio.run(asyncio.gather(*tasks, return_exceptions=True))

                 # Update progress bar after completion
                 progress_count = len([r for r in results if r is not None and not isinstance(r, Exception)])
                 my_bar.progress(1.0, text=f"Generation complete! {progress_count}/{total_images_to_generate} images successful.")
             else:
                 my_bar.progress(1.0, text="No tasks to generate.")
                 results = [] # Ensure results is an empty list if no tasks

         except Exception as e:
             st.error(f"An error occurred during asyncio.run/gather: {e}")
             my_bar.progress(1.0, text="Generation failed.") # Update bar on error
             results = [] # Ensure results is empty on error


    end_time = time.time()
    st.success(f"Image generation phase finished in {end_time - start_time:.2f} seconds.")

    # --- Phase 3: Process results ---
    # ...(processing logic remains the same)...

    # Clean up progress bar
    time.sleep(2) # Sync sleep is okay here in the main thread
    progress_placeholder.empty()

    if st.session_state.generated_images:
         play_sound("audio/bonus-points-190035.mp3")
         st.success("Image generation process complete. View and select images below.")
         st.experimental_rerun()
    else:
         st.warning("No images were successfully generated or tasks were created.")


# --- Step 2: Display generated images for selection (UI remains mostly sync) ---
if st.session_state.generated_images:
    st.subheader("Select Images to Process")
    zoom = st.slider("Zoom Level (%)", min_value=20, max_value=100, value=50, step=10, key="zoom_slider")
    display_width = int(1000 * (zoom / 100)) # Calculate width based on percentage of 1000px base

    # Add Select All / Deselect All buttons
    col1, col2, col3 = st.columns([1,1,3])
    if col1.button("Select All (1 Count)", key="select_all_1"):
        for entry in st.session_state.generated_images:
            for img in entry["images"]:
                img['selected_count'] = 1
        st.experimental_rerun()

    if col2.button("Deselect All", key="deselect_all_0"):
        for entry in st.session_state.generated_images:
             for img in entry["images"]:
                img['selected_count'] = 0
        st.experimental_rerun()


    for entry_idx, entry in enumerate(st.session_state.generated_images):
        topic = entry["topic"]
        lang = entry["lang"]
        images = entry["images"]

        if not images: continue # Skip topics with no images

        st.markdown(f"--- \n ### {topic} ({lang})")

        num_columns_display = 4 # Adjust number of columns for display
        rows_display = (len(images) + num_columns_display - 1) // num_columns_display

        for row in range(rows_display):
            cols = st.columns(num_columns_display)
            row_start_idx = row * num_columns_display
            for col_idx, img_idx in enumerate(range(row_start_idx, min(row_start_idx + num_columns_display, len(images)))):
                img = images[img_idx]
                with cols[col_idx]:
                    st.image(img['url'], width=display_width, caption=f"Src: {img['source']}, Tpl: {img['template']}")
                    # Ensure unique key using entry index, image index
                    unique_key_select = f"num_select_{entry_idx}_{img_idx}"
                    # Get current value (handle if key doesn't exist yet)
                    current_value = img.get('selected_count', 0 if not auto_mode else 1)

                    img['selected_count'] = st.number_input(
                        f"Count", # Shorter label
                        min_value=0, max_value=10, value=current_value, step=1,
                        key=unique_key_select, label_visibility="collapsed"
                    )

                    # DALL-E Variation button (Needs async handling on click)
                    if img.get("source") == "google" and not img.get("dalle_generated", False):
                         unique_key_dalle = f"dalle_button_{entry_idx}_{img_idx}"
                         if st.button("DALL-E Var", key=unique_key_dalle, help="Generate DALL-E variations (uses selected count)"):
                             variation_count = img.get('selected_count', 1)
                             if variation_count > 0:
                                 with st.spinner(f"Generating {variation_count} DALL-E variation(s)..."):
                                     # Run the async DALL-E function
                                     dalle_results = asyncio.run(async_create_dalle_variation(img['url'], variation_count))

                                     if dalle_results:
                                         st.success(f"{len(dalle_results)} DALL-E variation(s) generated!")
                                         img["dalle_generated"] = True # Mark original as processed
                                         # Append new DALL-E images to the list for this topic
                                         for dalle_img_data in dalle_results:
                                             entry["images"].append({
                                                 "url": dalle_img_data.url,
                                                 "selected": False, # Default new images to not selected
                                                 "template": img["template"], # Inherit template? Or specific DALL-E template?
                                                 "source": "dalle",
                                                 "dalle_generated": True,
                                                 "selected_count": 0 # Default count for new
                                             })
                                         st.experimental_rerun() # Rerun to show new images
                                     else:
                                         st.error("Failed to generate DALL-E variations.")
                             else:
                                 st.warning("Set count > 0 to generate DALL-E variations.")


# --- Step 3: Process selected images (Mostly Sync, but uses async text gen) ---
async def process_selected_images_async():
    """Contains the logic for processing, including async calls for text."""
    final_results = []
    st.session_state.cta_texts = {} # Reset CTA cache for this run

    # Determine max number of image columns needed for the output DataFrame
    max_images_per_topic = 0
    for entry in st.session_state.generated_images:
         selected_count_sum = sum(img.get('selected_count', 0) for img in entry['images'] if img.get('selected_count', 0) > 0)
         max_images_per_topic = max(max_images_per_topic, selected_count_sum)

    st.session_state['image_cols'] = [f'Image_{i+1}' for i in range(max_images_per_topic)]


    total_to_process = sum(img.get('selected_count', 0) for entry in st.session_state.generated_images for img in entry['images'] if img.get('selected_count', 0) > 0)
    processed_count = 0
    process_bar = st.progress(0, text=f"Processing selected images... 0/{total_to_process}")

    for entry in st.session_state.generated_images:
        topic = entry["topic"]
        lang = entry["lang"]
        images = entry["images"]

        res = {'Topic': topic, 'Language': lang}
        output_image_index = 1 # Counter for Image_N column names for this topic

        selected_images_with_counts = [(img, img.get('selected_count', 0)) for img in images if img.get('selected_count', 0) > 0]

        # --- Get CTA text for this language (once) ---
        if lang not in st.session_state.cta_texts:
             try:
                 # Use async version
                 cta_trans = await async_chatGPT(f"Return EXACTLY the text 'Learn More' in {lang} (no quotes).")
                 st.session_state.cta_texts[lang] = cta_trans.strip().strip('"') if cta_trans else "Learn More" # Fallback
             except Exception as e:
                 st.warning(f"Failed to get CTA translation for {lang}: {e}. Using default.")
                 st.session_state.cta_texts[lang] = "Learn More"
        cta_text_default = st.session_state.cta_texts[lang]


        for img, count in selected_images_with_counts:
            for i in range(count): # Process each instance requested by the count
                template = img['template'] # Template used/selected for this image
                image_url = img['url']
                source = img['source']

                # Skip processing for raw Gemini/DALL-E images if template isn't applicable
                # Add raw URL directly to results
                if source in ['gemini', 'dalle'] and (isinstance(template, str) and 'gemini' in template.lower()): # Or template == 'dalle' etc.
                    if output_image_index <= max_images_per_topic:
                         res[f'Image_{output_image_index}'] = image_url
                         output_image_index += 1
                    processed_count += 1
                    process_bar.progress(processed_count / total_to_process, text=f"Processing... {processed_count}/{total_to_process} (Raw: {topic[:20]}...)")
                    await asyncio.sleep(0.01) # Yield control briefly
                    continue # Move to next instance


                # --- Process images needing HTML templates ---
                try:
                    template_int = int(template) # Convert template to int for logic
                except (ValueError, TypeError):
                    st.warning(f"Skipping image with invalid template '{template}' for HTML generation.")
                    processed_count += 1
                    process_bar.progress(processed_count / total_to_process, text=f"Processing... {processed_count}/{total_to_process} (Skipped: {topic[:20]}...)")
                    await asyncio.sleep(0.01)
                    continue

                headline_text = ""
                cta_text = cta_text_default
                tag_line = ""

                try:
                     # --- Generate Headline (Async) ---
                     headline_prompt = ""
                     if template_int in [1, 2]:
                         headline_prompt = f"write a short text (up to 20 words) to promote an article about {topic} in {lang}. Goal: be concise yet compelling to click."
                     elif template_int in [3]:
                          headline_prompt = f"write 1 statement, same length, no quotes, for {re.sub(r'\\|.*','',topic)} in {lang}. Examples:\n'Surprising Travel Perks You Might Be Missing'\n'Little-Known Tax Tricks to Save Big'\nDont mention 'Hidden' or 'Unlock'."
                     elif template_int == 5:
                          headline_prompt = f"write 1 statement, same length, no quotes, for {re.sub(r'\\|.*','',topic)} in {lang}. ALL IN CAPS. wrap the 1-2 most urgent words in <span class='highlight'>...</span>. Make it under 60 chars total, to drive curiosity."
                     elif template_int == 7:
                          headline_prompt = f"write short punchy 1 sentence text for this article: \n casual and sharp and concise\nuse ill-tempered language\n don't address the reader (don't use 'you' etc)\n Avoid dark themes like drugs, death etc..\n MAX 70 CHARS, no !, Title Case, in lang {lang} for: {re.sub(r'\\|.*','',topic)}"
                     elif template_int in [4, 41, 42]:
                          headline_text = topic # Use topic directly as headline
                          cta_prompt = f"Return EXACTLY 'Read more about' in {lang} (no quotes)."
                          cta_text_specific = await async_chatGPT(cta_prompt)
                          cta_text = cta_text_specific.strip().strip('"') if cta_text_specific else "Read more about"
                     elif template_int == 6:
                          headline_text = '' # No headline for template 6
                     else: # Default headline
                          headline_prompt = f"Write a concise headline for {topic} in {lang}, no quotes."

                     # Generate headline if prompt exists and text not already set
                     if headline_prompt and not headline_text:
                         # Use faster/cheaper model for standard headlines maybe?
                         headline_text_raw = await async_chatGPT(prompt=headline_prompt, model='gpt-4o-mini')
                         headline_text = headline_text_raw.strip().strip('"').strip("'") if headline_text_raw else f"Explore {topic}" # Fallback


                     # --- Generate Tagline (Async) ---
                     if template_int == 5:
                          tag_line_prompt = f"Write a short tagline for {re.sub(r'\\|.*','',topic)} in {lang}, to drive action, max 25 chars, ALL CAPS, possibly with emoji. Do NOT mention the topic explicitly."
                          tag_line_raw = await async_chatGPT(tag_line_prompt, model='gpt-4o-mini')
                          tag_line = tag_line_raw.strip().strip('"').strip("'").strip("!") if tag_line_raw else "CHECK THIS OUT" # Fallback
                          # Minor formatting fix for span spacing in headline
                          headline_text = headline_text.replace(r"</span>", r"</span> ")


                     # --- Generate HTML (Sync) ---
                     html_content = save_html(
                         headline=headline_text, image_url=image_url,
                         cta_text=cta_text, template=template_int, tag_line=tag_line
                     )

                     # --- Capture Screenshot (Sync - consider wrapping if slow) ---
                     # screenshot_image = await asyncio.to_thread(capture_html_screenshot_playwright, html_content)
                     screenshot_image = capture_html_screenshot_playwright(html_content) # Keep sync for now


                     if screenshot_image:
                         # --- Upload Screenshot (Async) ---
                         s3_url = await async_upload_pil_image_to_s3(
                             image=screenshot_image, bucket_name=S3_BUCKET_NAME,
                             aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                             region_name=AWS_REGION
                         )
                         if s3_url:
                             if output_image_index <= max_images_per_topic:
                                 res[f'Image_{output_image_index}'] = s3_url
                                 output_image_index += 1
                             # Optionally display the generated ad image
                             # st.image(screenshot_image, caption=f"Processed: {topic} (Tpl: {template_int})", width=300)
                         else:
                              st.warning(f"Failed to upload screenshot for {topic} (Tpl: {template_int})")
                     else:
                          st.warning(f"Failed to capture screenshot for {topic} (Tpl: {template_int})")

                except Exception as process_error:
                     st.error(f"Error processing image for {topic} (Tpl: {template}, URL: {image_url}): {process_error}")
                     # logger.exception(f"Error processing image {topic} {template}")

                processed_count += 1
                process_bar.progress(processed_count / total_to_process, text=f"Processing... {processed_count}/{total_to_process} ({topic[:20]}...)")
                await asyncio.sleep(0.01) # Yield control


        # Fill remaining image columns with empty strings for this topic row
        for i in range(output_image_index, max_images_per_topic + 1):
             res[f'Image_{i}'] = ''
        final_results.append(res)


    process_bar.empty() # Remove progress bar

    if final_results:
        output_df = pd.DataFrame(final_results)

        # Apply shift_left_and_pad (Ensure image_cols is set correctly)
        img_cols_output = st.session_state.get('image_cols', [])
        if img_cols_output and all(col in output_df.columns for col in img_cols_output):
             output_df[img_cols_output] = output_df[img_cols_output].apply(shift_left_and_pad, axis=1)
        else:
             st.warning("Could not apply padding - image columns mismatch.")


        st.subheader("Final Results")
        st.dataframe(output_df)

        # Download CSV
        try:
             csv = output_df.to_csv(index=False).encode('utf-8')
             st.download_button(
                 label="Download Results as CSV",
                 data=csv,
                 file_name=f'final_results_{int(time.time())}.csv',
                 mime='text/csv',
             )
        except Exception as e:
             st.error(f"Failed to generate CSV: {e}")
    else:
        st.warning("No images were selected or processed successfully.")


# Button to trigger the processing (which now calls the async function)
if st.button("Process Selected Images Async", key="process_button"):
     if not st.session_state.get("generated_images"):
         st.warning("Please generate images first.")
     else:
          # Check if any images are selected
          any_selected = any(img.get('selected_count', 0) > 0 for entry in st.session_state.generated_images for img in entry['images'])
          if not any_selected:
               st.warning("No images selected for processing. Set count > 0 for images you want.")
          else:
                st.info("Starting asynchronous processing of selected images...")
                # Run the async processing function
                asyncio.run(process_selected_images_async())
                st.success("Processing finished.")
                play_sound("audio/completion.mp3") # Add a completion sound file
