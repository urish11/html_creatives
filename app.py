import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
import boto3
from botocore.exceptions import NoCredentialsError, ReadTimeoutError
import random
import string
import requests
from google import genai
import anthropic
import json
import base64
import os
import time
from playwright.sync_api import sync_playwright, Error as PlaywrightError
from tempfile import NamedTemporaryFile
import re
import math
from google_images_search import GoogleImagesSearch
import openai
import logging
from openai import OpenAI
from urllib.parse import urlencode
import zlib # For potentially compressing data in URL

# --- Basic Logging Setup (Optional but Recommended) ---
# Configure logging (adjust level as needed)
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    level=logging.INFO # Change to DEBUG for more verbose logs
)
logger = logging.getLogger(__name__)

# --- Retry Decorator (using tenacity) ---
# Install with: pip install tenacity
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, wait_exponential

# Define common transient errors for retries
RETRYABLE_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    ReadTimeoutError,
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
    # Add specific Google GenAI errors if available/needed
    PlaywrightError, # Retry playwright errors
    NoCredentialsError, # Could be transient in some environments
    # Add google_images_search specific errors if identifiable
)

# General purpose retry decorator
retry_decorator = retry(
    stop=stop_after_attempt(4), # Retry up to 3 times (4 attempts total)
    wait=wait_exponential(multiplier=1, min=2, max=10), # Exponential backoff: 2s, 4s, 8s
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    reraise=True # Reraise the exception if all retries fail
)
# --- Constants ---
MAX_IMAGES_PER_RUN = 80 # Threshold for splitting runs

# --------------------------------------------
# Load Secrets (Important: Ensure these are set in Streamlit secrets)
# --------------------------------------------
try:
    AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
    S3_BUCKET_NAME = st.secrets["S3_BUCKET_NAME"]
    AWS_REGION = st.secrets.get("AWS_REGION", "us-east-1")
    GPT_API_KEY = st.secrets["GPT_API_KEY"]
    FLUX_API_KEY = st.secrets["FLUX_API_KEY"] # Expecting a list/tuple of keys
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] # Expecting a list/tuple of keys
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") # For DALL-E
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] # Expecting a list/tuple for GIS
    GOOGLE_CX = st.secrets["GOOGLE_CX"]

    # --- Initialize API Clients ---
    openai_client = OpenAI(api_key=GPT_API_KEY ) # Client for DALL-E variations
    gpt_client = OpenAI(api_key=GPT_API_KEY) # Client for GPT text (if using openai lib) - Note: Your chatGPT uses requests
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    # genai client initialized within function using random key
    # s3 client initialized within function
    # gis client initialized within function using random key

except KeyError as e:
    st.error(f"Missing secret: {e}. Please configure Streamlit secrets.")
    st.stop()
except Exception as e:
    st.error(f"Error initializing API keys/clients: {e}")
    st.stop()

# --- Parameter Handling for Chunked Runs (Place early) ---
query_params = st.query_params
chunk_data_encoded = query_params.get("chunk_data")
autorun = query_params.get("autorun") == "true"

# Default DataFrame structure
default_df_structure = {"topic": ["example_topic"], "count": [1], "lang": ["english"], "template": ["2,3,4,41,42,5,6,7,gemini,gemini2,gemini7,gemini7claude,geminicandid,geministock,gemini7claude_simple | use , for random template"]}
initial_df_data = pd.DataFrame(default_df_structure)

st.session_state.setdefault('is_chunk_run', False) # Initialize if not present

if chunk_data_encoded and not st.session_state.is_chunk_run: # Process only once if not already marked
    logger.info("Received chunk data via URL. Preparing for auto-run...")
    try:
        # Decode the Base64 data (add decompression if you implement it)
        json_str = base64.urlsafe_b64decode(chunk_data_encoded).decode('utf-8')
        # Load the DataFrame chunk from JSON
        chunk_df = pd.read_json(json_str, orient='split')
        initial_df_data = chunk_df # Use this chunk as the initial data
        st.session_state.is_chunk_run = True # Mark this session as a chunk run
        st.warning("This tab is processing a specific chunk. Input is read-only.")
        logger.info(f"Loaded chunk with {len(chunk_df)} rows.")
    except Exception as e:
        st.error(f"Error decoding or loading chunk data from URL: {e}")
        logger.error(f"Failed to load chunk data: {e}", exc_info=True)
        st.error("Falling back to default input.")
        initial_df_data = pd.DataFrame(default_df_structure)
        autorun = False # Don't autorun if data is bad

# Set the auto_start_processing flag if autorun is true and it's a valid chunk run
# Use session_state to prevent re-triggering on simple reruns
if autorun and st.session_state.is_chunk_run and 'auto_start_triggered' not in st.session_state:
    st.session_state.auto_start_processing = True
    st.session_state.auto_start_triggered = True # Mark that we've set the flag based on the URL
    logger.info("Auto-start flag set for this chunk.")

# --------------------------------------------
# Helper Functions (with @retry_decorator where appropriate)
# --------------------------------------------

# --- DataFrame Utility ---
def shift_left_and_pad(row, target_cols):
    """Left-shifts non-null values and pads with empty strings."""
    valid_values = [x for x in row if pd.notna(x)]
    padded_values = valid_values + [''] * (len(target_cols) - len(valid_values))
    return pd.Series(padded_values[:len(target_cols)], index=target_cols)

# --- Audio Player ---
def play_sound(audio_file_path):
    """Plays an audio file in the Streamlit app."""
    try:
        with open(audio_file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay style="display:none;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
            """
            st.markdown(md, unsafe_allow_html=True)
        logger.info(f"Played sound: {audio_file_path}")
    except FileNotFoundError:
        logger.warning(f"Audio file not found: {audio_file_path}")
    except Exception as e:
        logger.error(f"Error playing sound: {e}")


# --- Playwright Setup ---
#@retry_decorator # Retry installation? Maybe not ideal, could hide persistent issues.
def install_playwright_browsers():
    """Installs Playwright browsers (Chromium) if not installed yet."""
    logger.info("Attempting to install Playwright browsers...")
    try:
        # Use subprocess for better error checking than os.system
        import subprocess
        subprocess.run(['playwright', 'install-deps'], check=True, capture_output=True, text=True)
        subprocess.run(['playwright', 'install', 'chromium'], check=True, capture_output=True, text=True)
        logger.info("Playwright browsers installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to install Playwright browsers: {e}\nOutput:\n{e.stdout}\n{e.stderr}")
        logger.error(f"Playwright install failed: {e}\nOutput:\n{e.stdout}\n{e.stderr}")
        return False
    except FileNotFoundError:
        st.error("Playwright command not found. Is Playwright installed in the environment?")
        logger.error("Playwright command not found.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during Playwright installation: {str(e)}")
        logger.error(f"Unexpected Playwright install error: {e}", exc_info=True)
        return False

# --- S3 Upload ---
@retry_decorator
def upload_pil_image_to_s3(
    image,
    bucket_name,
    aws_access_key_id,
    aws_secret_access_key,
    object_name='',
    region_name='us-east-1',
    image_format='PNG'
):
    """Upload a PIL image to S3."""
    logger.debug(f"Attempting S3 upload to {bucket_name}/{object_name or 'auto'}")
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
        logger.info(f"S3 Upload successful: {url}")
        return url

    except NoCredentialsError:
        st.error("AWS credentials not found or invalid.")
        logger.error("S3 Upload failed: AWS credentials error.")
        return None
    except Exception as e:
        st.error(f"Error in S3 upload: {str(e)}")
        logger.error(f"S3 Upload failed for {object_name}: {e}", exc_info=True)
        # Reraise if retry decorator is used, otherwise return None
        raise # Reraise for tenacity to handle retries

# --- API Call Functions (Add @retry_decorator) ---

@retry_decorator
def fetch_google_images(query, num_images=3):
    """Fetch images from Google Images using google_images_search."""
    logger.debug(f"Fetching Google Images for query: {query} (num: {num_images})")
    terms_list = query.split('~')
    res_urls = []
    for term in terms_list:
        try:
            api_key = random.choice(GOOGLE_API_KEY)
            gis = GoogleImagesSearch(api_key, GOOGLE_CX)
            search_params = {'q': term.strip(), 'num': num_images}
            gis.search(search_params)
            image_urls = [result.url for result in gis.results()]
            res_urls.extend(image_urls)
            logger.debug(f"Found {len(image_urls)} images for term: {term}")
        except Exception as e:
            # Let tenacity handle retries for transient errors
            # Log specific errors if needed
            logger.error(f"Error fetching Google Images for term '{term}': {e}", exc_info=True)
            st.warning(f"Warning fetching Google Images for '{term}': {e}")
            # If retries fail, tenacity will reraise, otherwise loop continues
            # Consider adding a placeholder or empty list for this term if needed after retries fail
            # For now, we rely on tenacity to handle it.

    unique_urls = list(set(res_urls))
    logger.info(f"Fetched {len(unique_urls)} unique Google Image URLs for query: {query}")
    return unique_urls

@retry_decorator
def gemini_text(prompt: str, api_key: str = None, model_id: str = "gemini-1.5-flash", api_endpoint: str = "generateContent", pd_policy: bool = False, predict_policy_text: str = "") -> str | None:
    """Calls the Gemini API via REST."""
    logger.debug(f"Calling Gemini Text API (REST) model: {model_id}")
    final_prompt = prompt + predict_policy_text if pd_policy else prompt

    if api_key is None:
        api_key = random.choice(GEMINI_API_KEY) # Use random key if not provided

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:{api_endpoint}?key={api_key}"
    headers = {"Content-Type": "application/json"}
    request_data = {
        "contents": [{"role": "user", "parts": [{"text": final_prompt}]}],
        "generationConfig": {"responseMimeType": "text/plain"},
    }

    try:
        response = requests.post(api_url, headers=headers, json=request_data, timeout=90) # Increased timeout
        response.raise_for_status() # Check for HTTP errors
        result_json = response.json()
        logger.debug(f"Gemini Text API response: {result_json}")
        # Safer extraction of text
        text_result = result_json.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').replace('```','')
        logger.info(f"Gemini Text API call successful for model {model_id}")
        return text_result
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Gemini API (REST): {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}, Response text: {e.response.text}")
            st.error(f"Gemini API Error: Status {e.response.status_code} - {e.response.text[:200]}") # Show limited error in UI
        else:
            st.error(f"Gemini API network error: {e}")
        raise # Reraise for tenacity

@retry_decorator
def gemini_text_lib(prompt, model='gemini-1.5-flash', pd_policy: bool = False, predict_policy_text: str = ""):
    """Calls Gemini API using the Python library."""
    logger.debug(f"Calling Gemini Text API (Library) model: {model}")
    final_prompt = prompt + predict_policy_text if pd_policy else prompt

    try:
        client = genai.Client(api_key=random.choice(GEMINI_API_KEY))
        # Use generate_content for newer models/API versions
        response = client.generate_content(model=f"models/{model}", contents=final_prompt)
        logger.info(f"Gemini Text API (Library) call successful for model {model}")
        return response.text
    except Exception as e:
        logger.error(f"Error in gemini_text_lib for model {model}: {e}", exc_info=True)
        st.error(f"Gemini Library Error: {str(e)[:200]}")
        raise # Reraise for tenacity

@retry_decorator
def chatGPT(prompt, model="gpt-4o", temperature=1.0, reasoning_effort='', pd_policy: bool = False, predict_policy_text: str = ""):
    """Calls OpenAI API (compatible) via REST for text generation."""
    logger.debug(f"Calling OpenAI Text API model: {model}, temp: {temperature}")
    final_prompt = prompt + predict_policy_text if pd_policy else prompt

    headers = {
        'Authorization': f'Bearer {GPT_API_KEY}',
        'Content-Type': 'application/json'
    }
    # Adjust data structure based on actual API endpoint and model requirements
    # This looks like a custom endpoint 'responses', standard is 'chat/completions'
    # Assuming '/v1/responses' is correct based on original code:
    data = {
        'model': model,
        'input': final_prompt, # Assuming 'input' is the correct field
        # Add other parameters like temperature only if non-default
    }
    if temperature != 1.0: # Add only if not default
         data['temperature'] = temperature
    # 'reasoning' seems specific, include if required by the '/responses' endpoint
    if reasoning_effort:
        data['reasoning'] = {"effort": reasoning_effort}

    api_url = '[https://api.openai.com/v1/responses](https://api.openai.com/v1/responses)' # Using the URL from original code

    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=90) # Increased timeout
        response.raise_for_status()
        content_data = response.json()
        logger.debug(f"OpenAI Text API response: {content_data}")

        # Extract text based on observed structure (adjust if needed)
        output_list = content_data.get('output', [])
        if 'o1' in model or 'o3' in model: # Specific model handling from original code
             if len(output_list) > 1 and 'content' in output_list[1] and len(output_list[1]['content']) > 0:
                 text_result = output_list[1]['content'][0].get('text', '')
             else:
                  text_result = ''
                  logger.warning("Unexpected response structure for o1/o3 model.")
        else:
             if len(output_list) > 0 and 'content' in output_list[0] and len(output_list[0]['content']) > 0:
                 text_result = output_list[0]['content'][0].get('text', '')
             else:
                  text_result = ''
                  logger.warning("Unexpected response structure for standard model.")

        logger.info(f"OpenAI Text API call successful for model {model}")
        return text_result

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling OpenAI Text API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}, Response text: {e.response.text}")
            st.error(f"OpenAI API Error: Status {e.response.status_code} - {e.response.text[:200]}")
        else:
            st.error(f"OpenAI API network error: {e}")
        raise # Reraise for tenacity

@retry_decorator
def claude(prompt, model="claude-3-sonnet-20240229", temperature=1, is_thinking=False, pd_policy: bool = False, predict_policy_text: str = ""):
    """Calls Anthropic Claude API."""
    logger.debug(f"Calling Claude API model: {model}, temp: {temperature}, thinking: {is_thinking}")
    final_prompt = prompt + predict_policy_text if pd_policy else prompt

    try:
        # Client is initialized globally now: anthropic_client
        message_params = {
            "model": model,
            "max_tokens": 4096, # Increased token limit slightly
            "temperature": temperature,
            "messages": [{"role": "user", "content": [{"type": "text", "text": final_prompt}]}]
        }
        # Add top_p if needed, was in original code but commented out later
        # if temperature < 1.0: # Often good practice to use top_p with lower temp
        #    message_params["top_p"] = 0.8

        if is_thinking:
             # This 'thinking' parameter seems non-standard for the official API as of early 2024.
             # Verify if this is a specific beta feature or internal endpoint.
             # Assuming it's valid based on original code:
             message_params["thinking"] = {"type": "enabled", "budget_tokens": 16000} # Budget seems high
             logger.warning("Using non-standard 'thinking' parameter for Claude.")


        message = anthropic_client.messages.create(**message_params)

        logger.debug(f"Claude API response: {message}")

        # Extract text (handle potential variations in response structure)
        if message.content:
            # Handle 'thinking' mode response structure if different
            if is_thinking and len(message.content) > 1 and hasattr(message.content[1], 'text'):
                 text_result = message.content[1].text
            elif hasattr(message.content[0], 'text'):
                 text_result = message.content[0].text
            else:
                 text_result = ""
                 logger.warning("Could not extract text from Claude response.")
        else:
            text_result = ""
            logger.warning("Empty content list in Claude response.")

        logger.info(f"Claude API call successful for model {model}")
        return text_result

    except (anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.AuthenticationError) as e:
        logger.error(f"Error calling Claude API: {type(e).__name__} - {e}")
        st.error(f"Claude API Error: {str(e)[:200]}")
        raise # Reraise for tenacity
    except Exception as e:
        logger.error(f"Unexpected error in Claude call: {e}", exc_info=True)
        st.error(f"Unexpected Claude Error: {str(e)[:200]}")
        raise # Reraise for tenacity


@retry_decorator
def gen_flux_img(prompt, height=784, width=960):
    """Generate images using FLUX model via Together.xyz API."""
    logger.debug(f"Calling Flux API (Schnell) for prompt: '{prompt[:50]}...'")
    url = "[https://api.together.xyz/v1/images/generations](https://api.together.xyz/v1/images/generations)"
    payload = {
        "prompt": prompt,
        "model": "black-forest-labs/FLUX.1-schnell", # Updated model name if needed
        "steps": 4, # Schnell works best with few steps
        "n": 1,
        "height": height,
        "width": width,
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {random.choice(FLUX_API_KEY)}" # Assumes list of keys
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120) # Longer timeout for image gen
        response.raise_for_status()
        result_json = response.json()
        logger.debug(f"Flux API response: {result_json}")
        image_url = result_json.get("data", [{}])[0].get("url")
        if image_url:
            logger.info("Flux API (Schnell) call successful.")
            return image_url
        else:
            logger.warning("Flux API call succeeded but no image URL found in response.")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Flux API (Schnell): {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}, Response text: {e.response.text}")
            # Check for specific errors like NSFW
            if "NSFW" in e.response.text:
                 logger.warning(f"Flux API blocked prompt as NSFW: {prompt[:50]}...")
                 st.warning(f"Prompt blocked by safety filter: '{prompt[:50]}...'")
                 return None # Return None specifically for NSFW
            st.error(f"Flux API Error: Status {e.response.status_code} - {e.response.text[:200]}")
        else:
            st.error(f"Flux API network error: {e}")
        raise # Reraise for tenacity

@retry_decorator
def gen_gemini_image(prompt):
    """Generates an image using the Gemini Image Generation API."""
    logger.debug(f"Calling Gemini Image API for prompt: '{prompt[:50]}...'")
    api_key = random.choice(GEMINI_API_KEY)
    # Verify the correct model endpoint for image generation (this might change)
    url = f"[https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent?key=](https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent?key=){api_key}" # Placeholder, might be wrong
    # Trying a documented image generation model (as of late 2023/early 2024):
    # url = f"[https://generativelanguage.googleapis.com/v1beta/models/imagegeneration:generateImage?key=](https://generativelanguage.googleapis.com/v1beta/models/imagegeneration:generateImage?key=){api_key}" # Needs verification
    # Using the model from the original code, assuming it's correct:
    url = f"[https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=](https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=){api_key}" # Updated model


    headers = {"Content-Type": "application/json"}
    # Structure might differ based on the actual endpoint. Assuming generateContent:
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json", # Request JSON response
             # Parameters might need adjustment based on model
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
           # "response_modalities": ["image"] # Maybe needed?
        },
        # Add safety settings if needed
        # "safetySettings": [...]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        res_json = response.json()
        logger.debug(f"Gemini Image API response: {res_json}")

        # --- Extract Image Data (Highly dependent on actual API response structure) ---
        # This part needs verification based on the correct endpoint and response format.
        # Assuming a structure similar to text generation but with image data:
        try:
            # Look for base64 data directly if available (common pattern)
            # This structure is a GUESS based on common patterns and the original code's attempt
            image_b64 = res_json['candidates'][0]["content"]["parts"][0]["inlineData"]['data']
            image_data = base64.b64decode(image_b64.encode('utf-8'))
            pil_image = Image.open(BytesIO(image_data))
            logger.info("Gemini Image API call successful.")
            return pil_image
        except (KeyError, IndexError, TypeError, base64.binascii.Error) as extract_err:
            logger.error(f"Failed to extract image data from Gemini response: {extract_err}. Response was: {res_json}")
            st.error("Failed to process image from Gemini response structure.")
            return None
        # --- End Image Data Extraction ---

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Gemini Image API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}, Response text: {e.response.text}")
            st.error(f"Gemini Image API Error: Status {e.response.status_code} - {e.response.text[:200]}")
        else:
            st.error(f"Gemini Image API network error: {e}")
        raise # Reraise for tenacity

@retry_decorator
def gen_flux_img_lora(prompt, height=784, width=960, lora_path="[https://huggingface.co/ddh0/FLUX-Amateur-Photography-LoRA/resolve/main/FLUX-Amateur-Photography-LoRA-v2.safetensors?download=true](https://huggingface.co/ddh0/FLUX-Amateur-Photography-LoRA/resolve/main/FLUX-Amateur-Photography-LoRA-v2.safetensors?download=true)"):
    """Generate images using FLUX dev model with LoRA via Together.xyz API."""
    logger.debug(f"Calling Flux API (LoRA) for prompt: '{prompt[:50]}...'")
    url = "[https://api.together.xyz/v1/images/generations](https://api.together.xyz/v1/images/generations)"
    headers = {
        "Authorization": f"Bearer {random.choice(FLUX_API_KEY)}",
        "Content-Type": "application/json"
    }
    # Add trigger words if needed for the LoRA
    lora_prompt = f"candid unstaged taken with iphone 8 : {prompt}" # From original code
    data = {
        "model": "black-forest-labs/FLUX.1-dev-lora",
        "prompt": lora_prompt,
        "width": width,
        "height": height,
        "steps": 20, # LoRAs might need more steps
        "n": 1,
        "response_format": "url",
        "image_loras": [{"path": lora_path, "scale": 0.99}]
        # "update_at" seems unnecessary unless API requires it
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        response_data = response.json()
        logger.debug(f"Flux LoRA API response: {response_data}")
        image_url = response_data.get('data', [{}])[0].get('url')
        if image_url:
            logger.info(f"Flux LoRA API call successful. Image URL: {image_url}")
            return image_url
        else:
            logger.warning("Flux LoRA API call succeeded but no image URL found.")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Flux LoRA API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}, Response text: {e.response.text}")
            st.error(f"Flux LoRA API Error: Status {e.response.status_code} - {e.response.text[:200]}")
        else:
            st.error(f"Flux LoRA API network error: {e}")
        raise # Reraise for tenacity


@retry_decorator
def capture_html_screenshot_playwright(html_content, width=1000, height=1000):
    """Use Playwright to capture a screenshot of the given HTML snippet."""
    logger.debug(f"Capturing Playwright screenshot ({width}x{height})")
    if not st.session_state.get('playwright_installed', False):
        st.error("Playwright browsers not installed properly")
        logger.error("Playwright screenshot skipped: browsers not installed.")
        return None

    temp_html_path = None # Define outside try
    browser = None # Define outside try
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu'] # Added disable-gpu
            )
            page = browser.new_page(viewport={'width': width, 'height': height})

            # Use a temporary file to reliably pass HTML to the browser
            with NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
                f.write(html_content)
                temp_html_path = f.name

            page.goto(f'file://{temp_html_path}')
            # Wait for network idle might be more reliable than fixed timeout
            page.wait_for_load_state('networkidle', timeout=15000) # Increased timeout
            # page.wait_for_timeout(1000) # Fallback if networkidle causes issues

            screenshot_bytes = page.screenshot()
            logger.info("Playwright screenshot captured successfully.")
            pil_image = Image.open(BytesIO(screenshot_bytes))

            page.close() # Close page first
            browser.close() # Then close browser

            return pil_image

    except PlaywrightError as e:
        logger.error(f"Playwright screenshot capture error: {e}", exc_info=True)
        st.error(f"Screenshot capture error: {str(e)}")
        raise # Reraise for tenacity
    except Exception as e:
         logger.error(f"Unexpected error during screenshot capture: {e}", exc_info=True)
         st.error(f"Unexpected screenshot error: {str(e)}")
         return None # Don't retry on unexpected errors
    finally:
        # Ensure cleanup happens even if errors occur
        if browser and browser.is_connected():
             try:
                 browser.close()
             except Exception as close_err:
                 logger.warning(f"Error closing playwright browser: {close_err}")
        if temp_html_path and os.path.exists(temp_html_path):
            try:
                os.unlink(temp_html_path)
            except Exception as unlink_err:
                logger.warning(f"Error deleting temp HTML file {temp_html_path}: {unlink_err}")


def save_html(headline, image_url, cta_text, template, tag_line=''):
    """Returns an HTML string based on the chosen template ID."""
    logger.debug(f"Generating HTML for template: {template}")
    # --- Paste ALL your HTML template definitions here (Template 1 to Template 8) ---
    # Ensure all template numbers used in the UI have corresponding HTML here
    # Example for Template 1 (replace with your full definitions)
    if template == 1:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template 1</title>
            <style>
                /* Your CSS for template 1 */
                body {{ font-family: sans-serif; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #f0f0f0; }}
                .ad-container {{ width: 1000px; height: 1000px; border: 1px solid #ccc; background: url('{image_url}') no-repeat center center; background-size: cover; display: flex; flex-direction: column; justify-content: space-between; align-items: center; padding: 20px; box-sizing: border-box; text-align: center; }}
                .ad-title {{ background-color: rgba(255, 255, 255, 0.8); padding: 15px; border-radius: 5px; font-size: 2.5em; margin-top: 20px; }}
                .cta-button {{ padding: 15px 30px; font-size: 1.8em; color: white; background-color: #FF5722; border: none; border-radius: 5px; text-decoration: none; cursor: pointer; margin-bottom: 20px; }}
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
         # --- Your HTML for Template 2 ---
         html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template 2</title>
            <style>
                 /* Your CSS for template 2 */
                body {{ font-family: sans-serif; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #f0f0f0; }}
                .ad-container {{ width: 1000px; height: 1000px; border: 1px solid #ccc; display: flex; flex-direction: column; overflow: hidden; position: relative; }}
                .ad-title {{ background-color: white; padding: 15px; font-size: 2.5em; text-align: center; flex: 0 0 auto; border-bottom: 1px solid #ccc;}}
                .ad-image {{ flex-grow: 1; background: url('{image_url}') no-repeat center center; background-size: cover; position: relative;}}
                .cta-button {{ position: absolute; bottom: 10%; left: 50%; transform: translateX(-50%); padding: 15px 30px; font-size: 1.8em; color: white; background-color: #FF5722; border: none; border-radius: 5px; text-decoration: none; cursor: pointer; z-index: 10; }}
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
    # --- Add elif blocks for templates 3, 4, 41, 42, 5, 6, 7, 8 ---
    # --- Make sure to include the exact HTML/CSS from your original code ---
  # Template 3 / 7 (Corrected code)
    elif template == 3 or template == 7: # Assuming 3 and 7 use the same base
        # --- FIX: Calculate the button class beforehand ---
        button_class = 'c1ta-button' if template == 7 else 'cta-button'
        # --- End FIX ---

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                /* CSS from original code for template 3/7 */
                body {{ margin: 0; padding: 0; font-family: 'Boogaloo', 'Segoe UI Emoji', 'Apple Color Emoji', sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; background: #f0f0f0; }}
                .container {{ position: relative; width: 1000px; height: 1000px; margin: 0; padding: 0; overflow: hidden; box-shadow: 0 0 20px rgba(0,0,0,0.2); }}
                .image {{ width: 1000px; height: 1000px; object-fit: cover; filter: saturate(130%) contrast(110%); transition: transform 0.3s ease; }}
                .image:hover {{ transform: scale(1.05); }}
                .overlay {{ position: absolute; top: 0; left: 0; width: 100%; min-height: 14%; background: red; display: flex; justify-content: center; align-items: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); padding: 20px; box-sizing: border-box; }}
                .overlay-text {{ color: #FFFFFF; font-size: 4em; text-align: center; text-shadow: 2.5px 2.5px 2px #000000; letter-spacing: 2px; font-family: 'Boogaloo', 'Segoe UI Emoji', 'Apple Color Emoji', sans-serif; margin: 0; word-wrap: break-word; }}
                .cta-button {{ position: absolute; bottom: 10%; left: 50%; transform: translateX(-50%); padding: 20px 40px; background: blue; color: white; border: none; border-radius: 50px; font-size: 3.5em; cursor: pointer; transition: all 0.3s ease; font-family: 'Boogaloo', 'Segoe UI Emoji', 'Apple Color Emoji', sans-serif; text-transform: uppercase; letter-spacing: 2px; box-shadow: 0 5px 15px rgba(0,0,255,0.4); }} /* Adjusted shadow */
                .cta-button:hover {{ background: #4ECDC4; transform: translateX(-50%) translateY(-5px); box-shadow: 0 8px 20px rgba(78,205,196,0.6); }}
                .c1ta-button {{ position: absolute; bottom: 10%; left: 50%; transform: translateX(-50%); padding: 20px 40px; background: green; color: white; border: none; border-radius: 50px; font-size: 3.5em; cursor: pointer; transition: all 0.3s ease; font-family: 'Boogaloo', 'Segoe UI Emoji', 'Apple Color Emoji', sans-serif; text-transform: uppercase; letter-spacing: 2px; box-shadow: 0 5px 15px rgba(0,255,0,0.4); }} /* Example style for c1ta */
                .c1ta-button:hover {{ background: #FFA500; }} /* Example hover for c1ta */
                /* Make sure both .cta-button and .c1ta-button styles are defined correctly */
            </style>
            <link href="https://fonts.googleapis.com/css2?family=Boogaloo&display=swap" rel="stylesheet">
        </head>
        <body>
            <div class="container">
                <img src="{image_url}" class="image" alt="Ad Image">
                <div class="overlay">
                    <h1 class="overlay-text">{headline}</h1>
                </div>
                <button class="{button_class}">{cta_text}</button>
            </div>
        </body>
        </html>
        """
    elif template == 4 or template == 41 or template == 42:
        # --- HTML for Template 4 / 41 / 42 (adjust overlay position) ---
        top_position = "50%" if template == 4 else ("15%" if template == 41 else "90%")
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template 4/41/42</title>
            <style>
                /* CSS from original code for template 4/41/42 */
                @import url('[https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap](https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap)');
                /* @font-face for Calibre might need adjustment if font isn't hosted/available */
                body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #F4F4F4; }}
                .container {{ position: relative; width: 1000px; height: 1000px; background-image: url('{image_url}'); background-size: cover; background-position: center; border-radius: 10px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); }}
                .text-overlay {{ position: absolute; width: 95%; background-color: rgba(255, 255, 255, 1); padding: 30px; border-radius: 10px; top: {top_position}; left: 50%; transform: translate(-50%, -50%); text-align: center; }}
                .small-text {{ font-size: 36px; font-weight: bold; color: #333; margin-bottom: 10px; font-family: 'Calibre', Arial, sans-serif; }} /* Ensure Calibre font works */
                .primary-text {{ font-size: 60px; font-weight: bold; color: #FF8C00; font-family: 'Montserrat', sans-serif; line-height: 1.2; text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000; }}
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
         # --- HTML for Template 5 ---
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
             <meta charset="UTF-8">
             <meta name="viewport" content="width=device-width, initial-scale=1.0">
             <title>Ad Template 5</title>
             <style>
                /* CSS from original code for template 5 */
                @import url('[https://fonts.googleapis.com/css2?family=Outfit:wght@100..900&display=swap](https://fonts.googleapis.com/css2?family=Outfit:wght@100..900&display=swap)');
                @import url('[https://fonts.googleapis.com/css2?family=Noto+Color+Emoji&display=swap](https://fonts.googleapis.com/css2?family=Noto+Color+Emoji&display=swap)');
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ width: 1000px; height: 1000px; margin: 0 auto; font-family: 'Outfit', sans-serif; }}
                .container {{ width: 100%; height: 100%; display: flex; flex-direction: column; position: relative; }}
                .image-container {{ width: 100%; height: 60%; background-color: #f0f0f0; display: flex; align-items: center; justify-content: center; }}
                .image-container img {{ width: 100%; height: 100%; object-fit: cover; }}
                .content-container {{ width: 100%; height: 40%; background-color: #121421; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2rem; gap: 2rem; }}
                .main-text {{ color: white; font-size: 3.5rem; font-weight: 700; text-align: center; }}
                .cta-button {{ background-color: #ff0000; color: white; padding: 1rem 2rem; font-size: 3.5rem; font-weight: 700; font-family: 'Outfit', sans-serif; border: none; font-style: italic; border-radius: 8px; cursor: pointer; transition: background-color 0.3s ease; }}
                .cta-button:hover {{ background-color: #cc0000; }}
                .intersection-rectangle {{ position: absolute; max-width: 70%; min-width: max-content; height: 80px; background-color: #121421; left: 50%; transform: translateX(-50%); top: calc(60% - 40px); border-radius: 10px; display: flex; align-items: center; justify-content: center; padding: 0 40px; }}
                .rectangle-text {{ font-family: 'Noto Color Emoji', sans-serif; color: #66FF00; font-weight: 700; text-align: center; font-size: 45px; white-space: nowrap; }}
                .highlight {{ color: #FFFF00; /* Style for highlighted span */ }}
             </style>
        </head>
        <body>
             <div class="container">
                <div class="image-container">
                     <img src="{image_url}" alt="Ad image">
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
    elif template == 6 or template == 8: # Assuming 6 and 8 use the same base (adjust dimensions for 8 later)
        # --- HTML for Template 6 / 8 ---
        width_px = "999px" if template == 8 else "1000px"
        height_px = "666px" if template == 8 else "1000px"
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
             <meta charset="UTF-8">
             <meta name="viewport" content="width=device-width, initial-scale=1.0">
             <title>Ad Template 6/8</title>
             <style>
                /* CSS from original code for template 6/8 */
                body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #F4F4F4; }}
                .container {{ position: relative; width: {width_px}; height: {height_px}; background-image: url('{image_url}'); background-size: cover; background-position: center; border-radius: 10px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); }}
                /* Template 6/8 might not have text overlay elements defined in original code? */
             </style>
        </head>
        <body>
             <div class="container">
                {f'<div class="text-overlay">{headline} {cta_text}</div>' if headline or cta_text else ''} 
             </div>
        </body>
        </html>
        """
    else:
        logger.warning(f"HTML template not found for ID: {template}")
        html_template = f"<html><body><p>Template {template} not found. Image: <img src='{image_url}' width='200'></p><p>Headline: {headline}</p><p>CTA: {cta_text}</p></body></html>"

    return html_template


@retry_decorator
def create_dalle_variation(image_url, count):
    """Creates DALL-E variations from a source image URL."""
    logger.debug(f"Creating {count} DALL-E variations for: {image_url}")
    if not OPENAI_API_KEY:
        st.error("OpenAI API Key for DALL-E is not configured.")
        logger.error("DALL-E variation skipped: OpenAI key missing.")
        return None

    try:
        # Download the source image
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        resp = requests.get(image_url, headers=headers, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))

        # Convert to PNG and check size
        png_buffer = BytesIO()
        img.save(png_buffer, format="PNG")
        png_size_mb = len(png_buffer.getvalue()) / (1024 * 1024)

        if png_size_mb > 4.0:
            logger.warning(f"Source image too large ({png_size_mb:.2f} MB), resizing to 512x512.")
            st.warning(f"Source image for variation was large, resizing...")
            # Resize (consider maintaining aspect ratio or different sizes like 1024x1024 if API supports)
            img = img.resize((512, 512))
            png_buffer = BytesIO()
            img.save(png_buffer, format="PNG")

        png_buffer.seek(0) # Reset buffer position after save

        # Call DALL-E API (ensure client is initialized)
        response = openai_client.images.create_variation(
            image=png_buffer.getvalue(), # Pass bytes directly
            n=count,
            size="512x512" # Or "1024x1024" if supported and desired
        )
        logger.info(f"DALL-E variation generated {len(response.data)} images.")
        return response.data # Returns a list of Image objects (containing URLs)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image for DALL-E variation: {e}")
        st.error(f"Failed to download source image: {e}")
        raise # Reraise for tenacity
    except (openai.APIError, openai.InvalidRequestError) as e:
         logger.error(f"Error generating DALL-E variation: {e}")
         st.error(f"DALL-E Error: {str(e)[:200]}")
         # Don't retry on InvalidRequestError, but do retry on APIError via decorator
         if isinstance(e, openai.InvalidRequestError):
              return None # Bad request, likely won't succeed on retry
         raise # Reraise APIError for tenacity
    except Exception as e:
         logger.error(f"Unexpected error during DALL-E variation: {e}", exc_info=True)
         st.error(f"Unexpected DALL-E error: {str(e)[:200]}")
         return None # Don't retry unknown errors


# --- Policy Text ---
predict_policy = """  Approved CTAs: Use calls-to-action like "Learn More" or "See Options" that clearly indicate leading to an article. Avoid CTAs like "Apply Now" or "Shop Now" that promise value not offered on the landing page.   \nProhibited Language: Do not use urgency terms ("Click now"), geographic suggestions ("Near you"), or superlatives ("Best").   \nEmployment/Education Claims: Do not guarantee employment benefits (like high pay or remote work) or education outcomes (like degrees or job placements).   \nFinancial Ad Rules: Do not guarantee loans, credit approval, specific investment returns, or debt relief. Do not offer banking, insurance, or licensed financial services.   \n"Free" Promotions: Generally avoid promoting services/products as "free". Exceptions require clarity: directing to an info article about a real free service, promoting a genuinely free course, or advertising free trials with clear terms. USE text on image, the most persuasive as you can you can add visual elements to the text to make up for the policy .the above is NOT relevent to the visual aspect of the image!  """

# --------------------------------------------
# Main Generation Logic Function
# --------------------------------------------
def run_generation(df_to_process):
    """Processes the DataFrame to generate images based on topics and counts."""
    st.session_state.generated_images = []  # Clear previous images for this specific run/chunk
    processed_combinations = set()
    if 'accumulated_generated_images' in st.session_state:
       del st.session_state.accumulated_generated_images # Ensure clean start

    # --- Start Progress Bar ---
    try:
        total_images_in_chunk = int(sum(df_to_process['count']))
    except Exception:
        st.error("Invalid 'count' value in input table.")
        return

    if total_images_in_chunk == 0:
        st.warning("No images to generate based on current input.")
        return

    progress_text = f"Generating {total_images_in_chunk} images..."
    if st.session_state.get("is_chunk_run"):
        progress_text += " (Chunk Run)"
    my_bar = st.progress(0, text=progress_text)
    images_processed_count = 0
    # ---

    logger.info(f"Starting generation run for {total_images_in_chunk} images.")
    st.spinner(f"Generating {total_images_in_chunk} images...")

    with st.spinner(f"Running generation for {total_images_in_chunk} images..."):
        for index, row in df_to_process.iterrows():
            topic = row['topic']
            try:
                count = int(row['count'])
                if count <= 0: continue # Skip rows with zero or negative count
            except (ValueError, TypeError):
                st.warning(f"Skipping row {index+1} due to invalid count: {row.get('count')}")
                continue
            lang = row['lang']
            template_str = str(row["template"]) # Ensure it's a string
            original_topic = topic # Keep original topic
            current_topic_images = [] # Store images for this row

            # --- Apply Topic Enhancement (only if enabled and not a chunk run) ---
            if ennhance_input and not st.session_state.get('is_chunk_run'):
                try:
                    enhanced_topic = chatGPT(
                        f"write this as more commercially attractive for ad promoting a article in {int(topic.count(' ') + 1)} words, 1 best option\n\n {topic}",
                        pd_policy=is_pd_policy, predict_policy_text=predict_policy
                    )
                    if enhanced_topic:
                        topic = enhanced_topic
                        logger.info(f"Enhanced topic: {topic}")
                    else:
                        logger.warning(f"Topic enhancement failed for '{original_topic}', using original.")
                except Exception as enhance_err:
                    logger.warning(f"Could not enhance topic '{original_topic}': {enhance_err}", exc_info=True)
                    st.warning(f"Failed to enhance topic '{original_topic}'. Using original.")
                    # topic remains original_topic

            # --- Handle 'google' keyword ---
            if "google" in topic.lower():
                logger.info(f"Processing Google image request for: {topic}")
                google_topic_query = topic.replace('google', ' ').strip()
                if '|' in google_topic_query:
                    google_topic_query = re.sub("^.*\|", "", google_topic_query).strip()

                try:
                    google_image_urls = fetch_google_images(google_topic_query, num_images=count)
                    for img_url in google_image_urls:
                        current_topic_images.append({
                            'url': img_url, 'selected': False, 'template': template_str,
                            'source': 'google', 'dalle_generated': False
                        })
                    images_processed_this_row = len(google_image_urls) # Count actual found URLs
                    images_processed_count += images_processed_this_row
                    percent_complete = min(1.0, images_processed_count / total_images_in_chunk) if total_images_in_chunk > 0 else 0
                    my_bar.progress(percent_complete, text=f"{progress_text} ({images_processed_count}/{total_images_in_chunk})")
                    logger.info(f"Processed {images_processed_this_row} Google images for '{google_topic_query}'.")

                except Exception as gis_err:
                     logger.error(f"Failed fetching Google images for '{google_topic_query}': {gis_err}", exc_info=True)
                     st.warning(f"Could not fetch Google images for '{google_topic_query}'.")
                     # Update progress even on failure to avoid getting stuck
                     images_processed_count += count # Assume failure counts towards requested
                     percent_complete = min(1.0, images_processed_count / total_images_in_chunk) if total_images_in_chunk > 0 else 0
                     my_bar.progress(percent_complete, text=f"{progress_text} ({images_processed_count}/{total_images_in_chunk})")


            # --- Handle AI Image Generation ---
            else:
                logger.info(f"Processing AI image request for: {topic}")
                completed_images_this_row = 0
                generation_attempts = 0
                max_attempts_per_image = count * 3 # Allow some retries overall

                while completed_images_this_row < count and generation_attempts < max_attempts_per_image:
                    generation_attempts += 1
                    selected_template = None
                    image_url = None
                    pil_image = None
                    source_api = "unknown"

                    try:
                        # --- Choose template ---
                        current_template_str = template_str
                        if ',' in current_template_str:
                            current_template_str = random.choice([t.strip() for t in current_template_str.split(',') if t.strip()])
                        logger.debug(f"Selected template string for generation: {current_template_str}")

                        # --- Gemini Image Generation ---
                        if 'gemini' in current_template_str.lower():
                            source_api = "gemini"
                            logger.debug(f"Generating Gemini image with template type: {current_template_str}")
                            gemini_prompt = None
                            # --- Logic to create gemini_prompt based on template_str variation ---
                            # (Using try-except around each API call for prompt generation)
                            prompt_generation_model = 'gpt-4o' # Default, adjust if needed
                            try:
                                if current_template_str == 'gemini2':
                                    gemini_prompt = chatGPT(f"write short prompt for\ngenerate square image promoting '{topic}' in language {lang} {random.choice(['use photos',''])}. add a CTA button with 'Learn More Here >>' in appropriate language\ns\nstart with 'square image aspect ratio of 1:1 of '\n\n", model=prompt_generation_model, pd_policy=is_pd_policy, predict_policy_text=predict_policy)
                                elif current_template_str == 'geminicandid':
                                     gemini_prompt = claude(f"""write a image prompt of a candid unstaged photo taken of a regular joe showing off his\her {topic} . the image is taken with smartphone candidly. in 1-2 sentences. Describe the quality of the image looking smartphone. start with "Square photo 1:1 iphone 12 photo uploaded to reddit:" this is for a fb ad that tries to look organic, but also make the image content intecing and somewhat perplexing, so try to be that but also draw clicks with high energy in the photo. dont make up facts like discounts! or specific prices! if you want to add a caption, specifically instruct it to be on the image. and be short in language {lang}""", is_thinking=True, pd_policy=is_pd_policy, predict_policy_text=predict_policy).replace("#","")
                                elif current_template_str == 'gemini7':
                                     gemini_prompt = gemini_text_lib(f"write short prompt for\ngenerate square image promoting '{topic}' in language {lang} . add a CTA button with 'Learn More Here >>' in appropriate language\\nand 'act fast' or 'limited available' \n \nshould be low quality and very enticing and alerting \n\nstart with 'square image aspect ratio of 1:1 of '\n\n be specific in what is shown . return JUST the best option, no intros", pd_policy=is_pd_policy, predict_policy_text=predict_policy)
                                elif current_template_str == 'gemini7claude':
                                     gemini_prompt = claude(f"write short prompt for\ngenerate square image promoting '{topic}' in language {lang} . add a CTA button with 'Learn More Here >>' in appropriate language\ \nshould be low quality and very enticing and alerting, don't make specific promises like x% discount and 'act fast' or 'limited available'  \n\nstart with 'square image aspect ratio of 1:1 of '\n\n be specific in what is shown . return JUST the best option, no intros\nif you want to add a caption, specifically instruct it to be on the image. and be short", is_thinking=False, pd_policy=is_pd_policy, predict_policy_text=predict_policy)
                                elif current_template_str == 'geministock':
                                     gemini_prompt = chatGPT(f"write short image prompt for {topic},no text on image,A high-quality image in a realistic setting, well-lit and visually appealing, suitable for use in marketing or editorial content.", model=prompt_generation_model, temperature= 1.0, pd_policy=is_pd_policy, predict_policy_text=predict_policy)
                                else: # Default gemini or other variations
                                     gemini_prompt = chatGPT(f"write short prompt for\ngenerate square image promoting '{topic}' in language {lang} {random.choice(['use photos',''])}. add a CTA button with 'Learn More Here >>' in appropriate language\nshould be low quality and very enticing and alerting\nstart with 'square image aspect ratio of 1:1 of '\n\n example output:\n\nsquare image of a concerned middle-aged woman looking at her tongue in the mirror under harsh bathroom lighting, with a cluttered counter and slightly blurry focus — big bold red text says “Early Warning Signs?” and a janky yellow button below reads “Learn More Here >>” — the image looks like it was taken on an old phone, with off angle, bad lighting, and a sense of urgency and confusion to provoke clicks.",model=prompt_generation_model, temperature= 1.0, pd_policy=is_pd_policy, predict_policy_text=predict_policy)

                            except Exception as prompt_err:
                                logger.error(f"Failed to generate Gemini prompt for {current_template_str}: {prompt_err}", exc_info=True)
                                st.warning(f"Skipping image: Failed to generate prompt for {current_template_str}")
                                continue # Try next attempt or next row

                            if gemini_prompt:
                                logger.info(f"Generated Gemini Img Prompt: {gemini_prompt[:100]}...")
                                pil_image = gen_gemini_image(gemini_prompt) # Call Gemini image gen API
                            else:
                                logger.warning(f"Gemini prompt generation returned empty for {current_template_str}.")


                        # --- FLUX Image Generation (or other non-Gemini templates) ---
                        else:
                            source_api = "flux" # Assume flux for others unless specified
                            selected_template = None
                            try:
                                selected_template = int(current_template_str) # Expecting integer template IDs now
                            except ValueError:
                                logger.warning(f"Invalid non-Gemini template ID: {current_template_str}. Skipping.")
                                continue # Skip this attempt

                            logger.debug(f"Generating Flux image with template ID: {selected_template}")
                            flux_prompt = None
                            prompt_generation_model = 'gpt-4' # Default for flux prompts
                            # --- Generate Flux prompt based on template ID ---
                            try:
                                if selected_template == 5:
                                    flux_prompt = chatGPT(f"Generate a concise visual image description (15 words MAX) for {topic}. Be wildly creative, curious... (rest of prompt 5)", model=prompt_generation_model, temperature=1.2, pd_policy=is_pd_policy, predict_policy_text=predict_policy)
                                elif selected_template == 7 :
                                    flux_prompt = chatGPT(f"Generate a visual image description 50 words MAX for {topic} , candid moment unstaged... (rest of prompt 7)", model='gpt-4o-mini', pd_policy=is_pd_policy, predict_policy_text=predict_policy) # Use faster model?
                                elif selected_template == 8:
                                     flux_prompt = chatGPT(f"A clean, high-resolution stock photo prompt of {topic}, no people, well-lit... (rest of prompt 8)", pd_policy=is_pd_policy, predict_policy_text=predict_policy)
                                else: # Default prompt for other templates (1, 2, 3, 4, 6, etc.)
                                     flux_prompt = chatGPT(f"Generate a visual image description 15 words MAX for {topic}. Use a visually enticing style with high CTR, avoid obvious descriptions.", model=prompt_generation_model, pd_policy=is_pd_policy, predict_policy_text=predict_policy)
                            except Exception as prompt_err:
                                logger.error(f"Failed to generate Flux prompt for template {selected_template}: {prompt_err}", exc_info=True)
                                st.warning(f"Skipping image: Failed to generate prompt for template {selected_template}")
                                continue

                            if flux_prompt:
                                logger.info(f"Generated Flux Img Prompt: {flux_prompt[:100]}...")
                                # --- Call appropriate Flux function based on template ---
                                if selected_template == 7: # LoRA template
                                    image_url = gen_flux_img_lora(flux_prompt)
                                elif selected_template == 8: # Specific dimensions
                                    image_url = gen_flux_img(flux_prompt, width=720, height=480)
                                elif selected_template == 5: # Specific dimensions
                                     image_url = gen_flux_img(flux_prompt, width=688, height=416)
                                else: # Default Flux call
                                    image_url = gen_flux_img(f"{random.choice(['cartoony clipart of ', ''])} {flux_prompt}")
                            else:
                                 logger.warning(f"Flux prompt generation returned empty for template {selected_template}.")


                        # --- Process generated image (Upload PIL or store URL) ---
                        final_image_url = None
                        if pil_image: # If Gemini returned an image
                            try:
                                final_image_url = upload_pil_image_to_s3(
                                    image=pil_image, bucket_name=S3_BUCKET_NAME,
                                    aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                    region_name=AWS_REGION
                                )
                                if not final_image_url: logger.error("S3 Upload failed after Gemini generation.")
                            except Exception as upload_err:
                                logger.error(f"S3 upload failed for Gemini image: {upload_err}", exc_info=True)
                                # Don't increment completed count if upload fails
                        elif image_url: # If Flux returned a URL
                            final_image_url = image_url
                        else:
                            logger.warning(f"No image generated or returned for attempt {generation_attempts} of topic '{topic}'.")
                            # Optionally sleep before retrying
                            time.sleep(2)


                        # --- If image successful, store and update progress ---
                        if final_image_url:
                            current_topic_images.append({
                                'url': final_image_url, 'selected': False,
                                'template': current_template_str, # Store the specific template used
                                'source': source_api, 'dalle_generated': False
                            })
                            completed_images_this_row += 1
                            images_processed_count += 1
                            percent_complete = min(1.0, images_processed_count / total_images_in_chunk) if total_images_in_chunk > 0 else 0
                            my_bar.progress(percent_complete, text=f"{progress_text} ({images_processed_count}/{total_images_in_chunk})")
                            logger.info(f"Successfully generated image {completed_images_this_row}/{count} for topic '{topic}' using {source_api}/{current_template_str}")
                        # else: # Optional: Handle failure explicitly if needed, maybe longer sleep

                    except Exception as gen_err:
                        logger.error(f"Error during image generation attempt {generation_attempts} for topic '{topic}': {gen_err}", exc_info=True)
                        st.warning(f"Generation failed for '{topic}' (attempt {generation_attempts}). Retrying if possible...")
                        time.sleep(3) # Sleep on error before potentially retrying loop

                if completed_images_this_row < count:
                     logger.warning(f"Failed to generate all requested images for topic '{topic}'. Got {completed_images_this_row}/{count} after {generation_attempts} attempts.")
                     st.warning(f"Could not generate all {count} images for '{topic}'.")
                     # Update progress bar for the ones requested but not generated to avoid stall
                     images_processed_count += (count - completed_images_this_row)
                     percent_complete = min(1.0, images_processed_count / total_images_in_chunk) if total_images_in_chunk > 0 else 0
                     my_bar.progress(percent_complete, text=f"{progress_text} ({images_processed_count}/{total_images_in_chunk})")


            # --- Append results for this row to accumulation list ---
            if 'accumulated_generated_images' not in st.session_state:
                st.session_state.accumulated_generated_images = []

            st.session_state.accumulated_generated_images.append({
                "topic": topic, # Use potentially enhanced topic
                "original_topic": original_topic,
                "lang": lang,
                "images": current_topic_images # Add images generated for this row
            })
            logger.info(f"Finished processing row {index+1}: '{original_topic}'")

    # --- Finalize Generation Run ---
    # Assign the accumulated results back to the state variable the rest of the UI uses
    st.session_state.generated_images = st.session_state.pop('accumulated_generated_images', [])
    if total_images_in_chunk > 0:
        my_bar.progress(1.0, text=f"Image generation complete! ({images_processed_count}/{total_images_in_chunk})")
    else:
        my_bar.empty()
    st.success(f"Finished generation run.")
    logger.info(f"Finished generation run. Processed {images_processed_count}/{total_images_in_chunk} images.")
    if images_processed_count > 0: # Play sound only if something was generated
        play_sound("audio/bonus-points-190035.mp3") # Play sound on completion

# --------------------------------------------
# Streamlit UI Setup
# --------------------------------------------
st.set_page_config(layout="wide", page_title="Creative Gen", page_icon="🎨")

# --- Initialize Playwright ---
# Guard installation with session state
# if 'playwright_installed' not in st.session_state:
#     with st.spinner("Initializing browser automation (one-time setup)..."):
#         st.session_state.playwright_installed = install_playwright_browsers()
# elif not st.session_state.playwright_installed:
#      st.error("Browser automation setup failed previously. Screenshots may not work.")

st.title("Creative Maker ")

# --- Examples Expander ---
with st.expander(f"Click to see examples for templates ", expanded=False):
    # ... (Your image list and grid display logic remains the same) ...
     image_list =  [
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744112392_1276.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744112392_1276.png)", "caption": "2"},
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744114470_4147.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744114470_4147.png)", "caption": "3"},
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744114474_6128.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744114474_6128.png)", "caption": "3"},
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744112152_3198.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744112152_3198.png)", "caption": "4"},
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744112606_9864.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744112606_9864.png)", "caption": "5"},
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744112288_6237.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744112288_6237.png)", "caption": "6 (image as is)"},
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744114000_7129.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744114000_7129.png)", "caption": "'google' in topic, google image search, use w/ templates 1-6"},
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744111656_7460.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744111656_7460.png)", "caption": "gemini"},
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744111645_6029.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744111645_6029.png)", "caption": "gemini7"},
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1743927991_4259.png](https://image-script.s3.us-east-1.amazonaws.com/image_1743927991_4259.png)", "caption": "gemini7"},
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1743411423_1020.png](https://image-script.s3.us-east-1.amazonaws.com/image_1743411423_1020.png)", "caption": "gemini7"},
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744098746_6749.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744098746_6749.png)", "caption": "gemini7claude"},
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744107557_6569.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744107557_6569.png)", "caption": "geminicandid"},
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744107067_6826.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744107067_6826.png)", "caption": "geminicandid"},
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744107348_8664.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744107348_8664.png)", "caption": "geminicandid"},
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744212220_8873.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744212220_8873.png)", "caption": "geministock"},
        {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744716627_8304.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744716627_8304.png)", "caption": "gemini7claude_simple"}
    ]
     num_columns = 6
     num_images = len(image_list)
     num_rows = (num_images + num_columns - 1) // num_columns
     # for i in range(num_rows):
     #    cols = st.columns(num_columns) # Create columns for the current row
     #    # Get the slice of images for the current row
     #    row_images = image_list[i * num_columns : (i + 1) * num_columns]

     #    # Populate columns with images and captions
     #    for j, item in enumerate(row_images):
     #        if item: # Check if there's an item
     #            # Use the j-th column *within the expander*
     #            with cols[j]:
     #                st.image(
     #                    item["image"],
     #                    use_container_width=True
     #                    )
     #                st.caption(item["caption"])

st.subheader("Enter Topics for Image Generation")

# --- Data Editor ---
# Initialize with potentially chunk-specific data
df = st.data_editor(
    initial_df_data,
    num_rows="dynamic",
    key="table_input", # Consistent key
    disabled=st.session_state.is_chunk_run, # Disable if it's a chunk run
    use_container_width=True
)

# --- Options ---
col_opts1, col_opts2, col_opts3 = st.columns(3)
with col_opts1:
    auto_mode = st.checkbox("Auto mode (Select all after generation)?", value=st.session_state.is_chunk_run) # Default True for chunks
with col_opts2:
    is_pd_policy = st.checkbox("PD policy (Adds policy text to prompts)?")
with col_opts3:
    ennhance_input = st.checkbox("Enhance input topic (via AI)?", disabled=st.session_state.is_chunk_run)


# --- Calculate Total Images and Handle Splitting ---
try:
    # Ensure 'count' column exists and is numeric
    if 'count' in df.columns:
         # Attempt conversion, fill errors with 0
        df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)
        total_images = df['count'].sum()
    else:
        st.error("Input table must contain a 'count' column.")
        total_images = 0
except Exception as e:
    st.error(f"Error processing 'count' column: {e}")
    logger.error(f"Error calculating total images: {e}", exc_info=True)
    total_images = 0

st.info(f"Total images requested: {total_images}")

# --- Conditional Split Button ---
if total_images > MAX_IMAGES_PER_RUN and not st.session_state.is_chunk_run:
    st.warning(f"Total image count ({total_images}) exceeds the limit ({MAX_IMAGES_PER_RUN}). Consider splitting.")
    if st.button("🚀 Split Run into Multiple Tabs", key="split_run"):
        logger.info(f"Splitting run for {total_images} images.")
        st.spinner("Calculating and preparing chunks...")
        chunks_df = []
        current_chunk_rows_indices = []
        current_chunk_image_count = 0

        # Split DataFrame rows based on cumulative image count
        for index, row in df.iterrows():
            # Ensure count is integer for summation
            row_count = int(row.get('count', 0))
            current_chunk_rows_indices.append(index)
            current_chunk_image_count += row_count

            # Create chunk if count exceeds limit (and chunk is not empty)
            if current_chunk_image_count >= MAX_IMAGES_PER_RUN and current_chunk_rows_indices:
                chunks_df.append(df.loc[current_chunk_rows_indices].copy())
                current_chunk_rows_indices = []
                current_chunk_image_count = 0

        # Add any remaining rows as the last chunk
        if current_chunk_rows_indices:
            chunks_df.append(df.loc[current_chunk_rows_indices].copy())

        if not chunks_df:
             st.error("No chunks could be created. Check input data.")
        else:
            st.write(f"Splitting into {len(chunks_df)} chunks.")
            urls_to_open = []
            max_url_len = 0

            for i, chunk_df_item in enumerate(chunks_df):
                try:
                    # Serialize chunk DataFrame to JSON
                    json_data = chunk_df_item.to_json(orient='split')
                    # Base64 encode
                    encoded_data = base64.urlsafe_b64encode(json_data.encode('utf-8')).decode('utf-8')
                    # Create query parameters
                    params = urlencode({'chunk_data': encoded_data, 'autorun': 'true'})
                    url_fragment = f"/?{params}"
                    urls_to_open.append(url_fragment)
                    max_url_len = max(max_url_len, len(url_fragment))
                    logger.debug(f"Chunk {i+1} URL fragment created (length: {len(url_fragment)} chars)")

                except Exception as e:
                     logger.error(f"Error processing chunk {i+1}: {e}", exc_info=True)
                     st.error(f"Failed to prepare chunk {i+1}.")

            logger.info(f"Max URL fragment length: {max_url_len}")
            if max_url_len > 2000: # Check typical safe URL length
                 st.warning(f"Warning: Generated URLs are very long (up to {max_url_len} chars). This might exceed browser limits.")

            # Inject JavaScript to open tabs if URLs were created
            if urls_to_open:
                urls_json = json.dumps(urls_to_open) # Python list -> JSON string

                # --- Define the JavaScript string WITH the placeholder ---
                javascript = """
                    <script>
                        const urls = {urls_json_placeholder}; // <<< ENSURE THIS LINE IS PRESENT AND CORRECT
                        let openedCount = 0;
                        if (urls && urls.length > 0) {{
                            alert(`Attempting to open ${urls.length} new tabs for processing. Please allow pop-ups if prompted.`);
                            urls.forEach((url, index) => {{
                                console.log(`Opening chunk ${index + 1}`);
                                const tabWindow = window.open(url, `_blank_chunk_${index}`);
                                if (tabWindow) {{
                                    openedCount++;
                                }} else {{
                                    console.error(`Failed to open tab for chunk ${index + 1}. Pop-up blocked?`);
                                }}
                            }});
                            if (openedCount < urls.length) {{
                                alert(`Could only open ${openedCount}/${urls.length} tabs. Please check your pop-up blocker settings.`);
                            }}
                        }} else {{
                             alert("No chunks were prepared to be opened.");
                        }}
                    </script>
                """
                # --- End of the multi-line string definition ---

                # --- This format call requires the placeholder above ---
                try:
                    # Call format using the exact placeholder name
                    # formatted_javascript = javascript.format(urls_json_placeholder=urls_json)
                    formatted_javascript  = javascript.replace("{urls_json_placeholder}", urls_json)
                    st.text(formatted_javascript)
                    st.markdown(formatted_javascript, unsafe_allow_html=True)
                    st.success(f"Attempted to open {len(urls_to_open)} new tabs. Please check your browser.")
                except KeyError:
                     # This error block is being hit currently
                     st.error("Error formatting JavaScript for split tabs: Placeholder missing in the script template. Please check the code.")
                     logger.error("KeyError: urls_json_placeholder missing in javascript string template.")
                except Exception as format_err:
                     st.error(f"An unexpected error occurred while preparing split tab script: {format_err}")
                     logger.error(f"Error during JS formatting: {format_err}", exc_info=True)

            else:
                 st.error("No valid URLs could be generated for splitting.")
        # Stop execution in the current tab after attempting to split
        st.stop()

# --- Manual Generate Button ---
# Show if not splitting OR if it's a chunk run (allow manual trigger for chunks too)
allow_manual_generate = not (total_images > MAX_IMAGES_PER_RUN and not st.session_state.is_chunk_run) or st.session_state.is_chunk_run
generate_button_disabled = (total_images > MAX_IMAGES_PER_RUN and not st.session_state.is_chunk_run)

if st.button("Generate Images", key="manual_generate", disabled=generate_button_disabled, type="primary"):
    # Clear any auto-start flag if manually triggered
    st.session_state.pop("auto_start_processing", None)
    # Clear previous results before starting new manual run
    if 'generated_images' in st.session_state:
        del st.session_state.generated_images
    if 'accumulated_generated_images' in st.session_state:
        del st.session_state.accumulated_generated_images
    logger.info("Manual 'Generate Images' clicked.")
    run_generation(df.copy()) # Process the current state of the DataFrame

# --- Auto-start Logic ---
# Check the flag *after* button definitions
if st.session_state.get("auto_start_processing"):
    # Consume the flag immediately
    st.session_state.pop("auto_start_processing")
    logger.info("Auto-starting generation for URL chunk...")
    st.info("Auto-starting generation for this chunk...")
    # Clear previous results before starting auto run
    if 'generated_images' in st.session_state:
        del st.session_state.generated_images
    if 'accumulated_generated_images' in st.session_state:
        del st.session_state.accumulated_generated_images
    # df should hold the chunk data because it was initialized using initial_df_data
    run_generation(df.copy())


# --- Display Generated Images & Selection ---
st.divider()
if 'generated_images' in st.session_state and st.session_state.generated_images:
    st.subheader("Select Images to Process")
    zoom = st.slider("Zoom Level for Selection", min_value=50, max_value=400, value=200, step=25)

    # --- Auto-Select All Logic ---
    if auto_mode:
        st.write("Auto-mode enabled: Setting count to 1 for all generated images.")
        for entry in st.session_state.generated_images:
            for img in entry["images"]:
                img['selected_count'] = 1 # Default to 1 in auto mode

    # --- Grid Display ---
    for entry_idx, entry in enumerate(st.session_state.generated_images):
        topic = entry.get("topic", "Unknown Topic")
        original_topic = entry.get("original_topic", topic)
        lang = entry.get("lang", "N/A")
        images = entry.get("images", [])

        if not images: continue # Skip if no images were generated for this entry

        st.write(f"#### Topic: {topic} ({lang})")
        if topic != original_topic: st.caption(f"(Original: {original_topic})")

        num_columns_display = 5 # Adjust as needed
        rows_display = math.ceil(len(images) / num_columns_display)

        for row_idx in range(rows_display):
            cols = st.columns(num_columns_display)
            row_images_data = images[row_idx * num_columns_display : (row_idx + 1) * num_columns_display]

            for col_idx, img_data in enumerate(row_images_data):
                if col_idx < len(cols): # Ensure we don't exceed columns
                    with cols[col_idx]:
                        img_url = img_data.get('url')
                        img_source = img_data.get('source', 'unknown')
                        img_template = img_data.get('template', 'N/A')
                        is_dalle_generated = img_data.get('dalle_generated', False)

                        st.image(img_url, width=zoom, caption=f"Source: {img_source} | Template: {img_template}")

                        # Use a more unique key for the number input
                        unique_key = f"num_select_{entry_idx}_{img_url[-10:]}" # Use part of URL for uniqueness

                        # Initialize selected_count if not present
                        img_data.setdefault('selected_count', 1 if auto_mode else 0)

                        # Number input for selection count
                        selected_count = st.number_input(
                            f"Select Count",
                            min_value=0, max_value=10,
                            value=img_data['selected_count'],
                            key=unique_key,
                            label_visibility="collapsed", # Keep UI cleaner
                            help=f"How many times to process this image ({img_url[-10:]})"
                        )
                        # Update session state immediately
                        img_data['selected_count'] = selected_count

                        # DALL-E Variation button for Google images
                        if img_source == "google" and not is_dalle_generated:
                            dalle_button_key = f"dalle_button_{entry_idx}_{img_url[-10:]}"
                            if st.button("DALL-E Variation", key=dalle_button_key, help="Generate AI variations (uses DALL-E)"):
                                if selected_count > 0:
                                    with st.spinner("Generating DALL-E variations..."):
                                        dalle_results = create_dalle_variation(img_url, selected_count)
                                        if dalle_results:
                                            st.success(f"{len(dalle_results)} DALL-E variations generated!")
                                            img_data["dalle_generated"] = True # Mark original as processed for variation
                                            # Append new DALL-E images to the list for this entry
                                            for dalle_img in dalle_results:
                                                entry["images"].append({
                                                    "url": dalle_img.url,
                                                    "selected": False, # Start unselected
                                                    "selected_count": 1 if auto_mode else 0, # Default selection
                                                    "template": img_data["template"], # Inherit template?
                                                    "source": "dalle",
                                                    "dalle_generated": True
                                                })
                                            st.rerun() # Rerun to display the newly added images
                                        else:
                                            st.error("Failed to generate DALL-E variations.")
                                else:
                                    st.warning("Set count > 0 to generate variations.")


elif st.session_state.get("is_chunk_run") and 'generated_images' not in st.session_state:
     st.info("Waiting for auto-run to complete image generation for this chunk...")
elif not st.session_state.get("is_chunk_run"):
     st.info("Enter topics above and click 'Generate Images' or 'Split Run'.")


# --- Process Selected Images Button ---
st.divider()
st.subheader("Process & Finalize Selected Images")
if st.button("Process Selected Images", type="primary"):
    logger.info("'Process Selected Images' clicked.")
    # Ensure generated_images exists and is populated
    if 'generated_images' not in st.session_state or not st.session_state.generated_images:
        st.warning("No images have been generated or selected yet.")
        st.stop()

    final_results = []
    selected_image_count_total = 0
    # Calculate total selected images for progress
    for entry in st.session_state.generated_images:
        for img in entry.get("images", []):
            selected_image_count_total += img.get('selected_count', 0)

    if selected_image_count_total == 0:
        st.warning("No images selected for processing. Set count > 0 for desired images.")
        st.stop()

    logger.info(f"Processing {selected_image_count_total} selected image instances.")
    process_bar = st.progress(0, text=f"Processing {selected_image_count_total} selected images...")
    processed_count = 0
    cta_texts_cache = {} # Cache translations

    with st.spinner("Processing selected images (generating text, HTML, screenshots)..."):
        for entry in st.session_state.generated_images:
            topic = entry.get("topic", "Unknown Topic")
            original_topic = entry.get("original_topic", topic)
            lang = entry.get("lang", "en")
            images = entry.get("images", [])

            res_row = {'Topic': topic, 'Original Topic': original_topic, 'Language': lang}
            image_col_counter = 1 # Counter for image columns in the output row

            selected_images_in_entry = [img for img in images if img.get('selected_count', 0) > 0]

            if not selected_images_in_entry: continue

            # --- Get CTA text (cached per language) ---
            if lang not in cta_texts_cache:
                try:
                    cta_trans = chatGPT(f"Return EXACTLY the text 'Learn More' in {lang} (no quotes).", model='gpt-4o-mini').strip('"').strip("'")
                    if not cta_trans: cta_trans = "Learn More" # Fallback
                    cta_texts_cache[lang] = cta_trans
                    logger.info(f"Translated 'Learn More' for {lang}: {cta_trans}")
                except Exception as cta_err:
                     logger.error(f"Failed to translate CTA for {lang}: {cta_err}", exc_info=True)
                     cta_texts_cache[lang] = "Learn More" # Fallback
            cta_text_base = cta_texts_cache[lang]
            # ---

            for img_data in selected_images_in_entry:
                count_to_process = img_data['selected_count']
                template_str_or_int = img_data['template'] # Could be 'geminiX' or int

                for i in range(count_to_process):
                    processed_count += 1
                    logger.info(f"Processing instance {i+1}/{count_to_process} for image {img_data['url'][-10:]}, Topic: {topic}")
                    final_s3_url = img_data['url'] # Default to original URL

                    # --- Handle Gemini images (no HTML template processing needed) ---
                    if isinstance(template_str_or_int, str) and "gemini" in template_str_or_int.lower():
                        logger.debug("Skipping HTML/Screenshot for Gemini generated image.")
                        # Just add the existing S3 URL to the results
                        pass # final_s3_url is already set

                    # --- Handle Templated Images ---
                    else:
                        try:
                             template_id = int(template_str_or_int) # Convert to int for template logic
                        except (ValueError, TypeError):
                             logger.warning(f"Invalid template ID '{template_str_or_int}' for processing image {img_data['url']}. Skipping HTML generation.")
                             # Add original URL but skip processing
                             res_row[f'Image_{image_col_counter}'] = final_s3_url
                             image_col_counter += 1
                             process_bar.progress(min(1.0, processed_count / selected_image_count_total), text=f"Processing {processed_count}/{selected_image_count_total}...")
                             continue # Skip to next instance/image

                        headline_text = ""
                        cta_text_final = cta_text_base
                        tag_line = ""

                        try:
                            # --- Generate Headline & Tagline based on template ---
                            topic_for_prompt = re.sub('\\|.*','', topic) # Clean topic for prompts

                            if template_id in [1, 2]:
                                headline_prompt = f"write a short text (up to 20 words) to promote an article about {topic_for_prompt} in {lang}. Goal: be concise yet compelling to click."
                                headline_text = chatGPT(prompt=headline_prompt, model='gpt-4o-mini', pd_policy=is_pd_policy, predict_policy_text=predict_policy).strip('"').strip("'")
                            elif template_id in [3, 7]: # Use template 3 prompt logic
                                headline_prompt = f"write 1 statement, same length, no quotes, for {topic_for_prompt} in {lang}. Examples:\n'Surprising Travel Perks You Might Be Missing'\n'Little-Known Tax Tricks to Save Big'\nDont mention 'Hidden' or 'Unlock'."
                                headline_text = chatGPT(prompt=headline_prompt, model='gpt-4o-mini', pd_policy=is_pd_policy, predict_policy_text=predict_policy).strip('"').strip("'")
                                if template_id == 7: cta_text_final = chatGPT(f"Return EXACTLY the text 'Check This Out' in {lang} (no quotes).", model='gpt-4o-mini').strip('"') or "Check This Out" # Custom CTA for T7?
                            elif template_id == 5:
                                headline_prompt = f"write 1 statement, same length, no quotes, for {topic_for_prompt} in {lang}. ALL IN CAPS. wrap the 1-2 most urgent words in <span class='highlight'>...</span>. Make it under 60 chars total, to drive curiosity."
                                headline_text = chatGPT(prompt=headline_prompt, model='gpt-4o-mini', pd_policy=is_pd_policy, predict_policy_text=predict_policy).strip('"').strip("'")
                                tagline_prompt = f"Write a short tagline for {topic_for_prompt} in {lang}, to drive action, max 25 chars, ALL CAPS, possibly with emoji. Do NOT mention the topic explicitly."
                                tag_line = chatGPT(prompt=tagline_prompt, model='gpt-4o-mini').strip('"').strip("'").strip("!")
                                headline_text = headline_text.replace(r"</span>", r"</span> ") # Ensure space after span
                            elif template_id in [4, 41, 42]:
                                headline_text = topic # Use topic as headline
                                cta_text_final = chatGPT(f"Return EXACTLY 'Read more about' in {lang} (no quotes).", model='gpt-4o-mini').strip('"') or "Read more about"
                            elif template_id in [6, 8]:
                                headline_text = '' # No headline for these templates
                                cta_text_final = '' # No CTA for these templates
                            else: # Default fallback headline
                                headline_prompt = f"Write a concise headline for {topic_for_prompt} in {lang}, no quotes."
                                headline_text = chatGPT(prompt=headline_prompt, model='gpt-4o-mini', pd_policy=is_pd_policy, predict_policy_text=predict_policy).strip('"').strip("'")

                            # --- Generate HTML ---
                            html_content = save_html(
                                headline=headline_text or "", # Ensure not None
                                image_url=img_data['url'],
                                cta_text=cta_text_final or "", # Ensure not None
                                template=template_id,
                                tag_line=tag_line or "" # Ensure not None
                            )

                            # --- Capture Screenshot ---
                            screenshot_width = 720 if template_id == 8 else 1000
                            screenshot_height = 480 if template_id == 8 else 1000
                            screenshot_image = capture_html_screenshot_playwright(
                                html_content, width=screenshot_width, height=screenshot_height
                            )

                            # --- Upload Screenshot to S3 ---
                            if screenshot_image:
                                final_s3_url = upload_pil_image_to_s3(
                                    image=screenshot_image,
                                    bucket_name=S3_BUCKET_NAME,
                                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                    region_name=AWS_REGION
                                )
                                if not final_s3_url:
                                    logger.error(f"S3 upload failed for screenshot of {img_data['url']}. Using original URL.")
                                    final_s3_url = img_data['url'] # Fallback to original if upload fails
                                else:
                                    logger.info(f"Processed and uploaded screenshot: {final_s3_url}")
                            else:
                                logger.warning(f"Screenshot capture failed for {img_data['url']}. Using original URL.")
                                final_s3_url = img_data['url'] # Fallback

                        except Exception as process_err:
                            logger.error(f"Error processing image instance {img_data['url']} with template {template_str_or_int}: {process_err}", exc_info=True)
                            st.warning(f"Failed to process image {img_data['url'][-10:]} instance {i+1}. Using original URL.")
                            final_s3_url = img_data['url'] # Fallback to original URL on any processing error


                    # --- Add final URL to results row ---
                    res_row[f'Image_{image_col_counter}'] = final_s3_url
                    image_col_counter += 1

                    # Update progress bar
                    process_bar.progress(min(1.0, processed_count / selected_image_count_total), text=f"Processing {processed_count}/{selected_image_count_total}...")

            # Add the completed row for this topic to the final results
            final_results.append(res_row)

    # --- Display Final Results ---
    if final_results:
        logger.info("Processing complete. Preparing final results DataFrame.")
        output_df = pd.DataFrame(final_results)

        # Reorganize and flatten image links
        image_cols = sorted([col for col in output_df.columns if col.startswith("Image_")], key=lambda x: int(x.split('_')[1])) # Sort by Image_N
        if image_cols:
             # Apply shift_left_and_pad row-wise
             output_df[image_cols] = output_df.apply(
                  lambda row: shift_left_and_pad(row[image_cols], image_cols),
                  axis=1
             )
             # Rename columns sequentially
             output_df.rename(columns={old_col: f"Image_{i+1}" for i, old_col in enumerate(image_cols)}, inplace=True)
             # Reorder columns (optional)
             cols_order = ['Topic', 'Original Topic', 'Language'] + [f"Image_{i+1}" for i in range(len(image_cols))]
             output_df = output_df[cols_order]

        st.subheader("Final Results")
        st.dataframe(output_df)

        # Download CSV
        try:
            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='final_results.csv',
                mime='text/csv',
            )
            logger.info("Final results displayed and download button provided.")
        except Exception as e:
            logger.error(f"Error generating CSV for download: {e}", exc_info=True)
            st.error("Could not prepare CSV for download.")
    else:
        logger.warning("Processing finished, but no final results were generated.")
        st.warning("Processing finished, but no results were generated (check logs for errors).")
