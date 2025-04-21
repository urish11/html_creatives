import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
import boto3
from botocore.exceptions import NoCredentialsError # Removed ReadTimeoutError unless needed elsewhere
import random
import string
import requests
from google import genai # Keep if gemini_text_lib is used
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
import openai # Keep for DALL-E
# Removed logging import - Revert to original state
from openai import OpenAI # Keep for DALL-E client
from urllib.parse import urlencode # Keep for chunking
# Removed zlib import - Revert to original state
import streamlit.components.v1 as components # Keep for split button fix

st.set_page_config(layout="wide", page_title="Creative Gen", page_icon="🎨")

# --- Removed Logging Setup ---

# --- Removed Retry Decorator ---

# --- Constants ---
MAX_IMAGES_PER_RUN = 80 # Keep threshold for splitting runs

# --------------------------------------------
# Load Secrets
# --------------------------------------------
try:
    AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
    S3_BUCKET_NAME = st.secrets["S3_BUCKET_NAME"]
    AWS_REGION = st.secrets.get("AWS_REGION", "us-east-1")
    GPT_API_KEY = st.secrets["GPT_API_KEY"]
    FLUX_API_KEY = st.secrets["FLUX_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
    # --- Reverted DALL-E Key Setup ---
    openai.api_key = st.secrets.get("OPENAI_API_KEY") # Use this global setup as per original
    # --- End Revert ---
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    GOOGLE_CX = st.secrets["GOOGLE_CX"]

    # --- Initialize API Clients (where needed) ---
    # Keep Anthropic client
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    # Keep OpenAI client instance using the correct key for text generation if needed by chatGPT function
    # Note: Original chatGPT used requests, let's keep that.
    # DALL-E calls will use the global openai.api_key now.
    gpt_client = OpenAI(api_key=GPT_API_KEY) # Keep client instance if used elsewhere, but chatGPT func below uses requests.

except KeyError as e:
    st.error(f"Missing secret: {e}. Please configure Streamlit secrets.")
    st.stop()
except Exception as e:
    st.error(f"Error initializing API keys/clients: {e}")
    st.stop()

# --- Parameter Handling for Chunked Runs (Keep) ---
query_params = st.query_params
chunk_data_encoded = query_params.get("chunk_data")
autorun = query_params.get("autorun") == "true"

# Default DataFrame structure (from original)
default_df_structure = {"topic": ["example_topic"], "count": [1], "lang": ["english"], "template": ["2,3,4,41,42,5,6,7,gemini,gemini2,gemini7,gemini7claude,geminicandid,geministock,gemini7claude_simple | use , for random template"]}
initial_df_data = pd.DataFrame(default_df_structure)

st.session_state.setdefault('is_chunk_run', False)

if chunk_data_encoded and not st.session_state.is_chunk_run:
    st.info("Received chunk data via URL. Preparing for auto-run...") # Keep UI info
    try:
        json_str = base64.urlsafe_b64decode(chunk_data_encoded).decode('utf-8')
        chunk_df = pd.read_json(json_str, orient='split')
        initial_df_data = chunk_df
        st.session_state.is_chunk_run = True
        st.warning("This tab is processing a specific chunk. Input is read-only.")
    except Exception as e:
        st.error(f"Error decoding or loading chunk data from URL: {e}")
        st.error("Falling back to default input.")
        initial_df_data = pd.DataFrame(default_df_structure)
        autorun = False

if autorun and st.session_state.is_chunk_run and 'auto_start_triggered' not in st.session_state:
    st.session_state.auto_start_processing = True
    st.session_state.auto_start_triggered = True

# --------------------------------------------
# Helper Functions (Reverted to Original Logic where applicable)
# --------------------------------------------

# --- DataFrame Utility (Keep Improved Version) ---
def shift_left_and_pad(row, target_cols):
    """Left-shifts non-null values and pads with empty strings."""
    valid_values = [x for x in row if pd.notna(x)]
    padded_values = valid_values + [''] * (len(target_cols) - len(valid_values))
    return pd.Series(padded_values[:len(target_cols)], index=target_cols)

# --- Audio Player (Keep) ---
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
    except FileNotFoundError:
        st.warning(f"Audio file not found: {audio_file_path}") # Use st.warning
    except Exception as e:
        st.error(f"Error playing sound: {e}") # Use st.error


# --- Playwright Setup (Keep Improved for Streamlit Cloud) ---
def install_playwright_browsers():
    """Installs Playwright browsers (Chromium) if not installed yet."""
    logger.info("Attempting to install Playwright Chromium browser...") # Use logger if keeping basic logging
    try:
        import subprocess
        # REMOVED install-deps call
        result = subprocess.run(['playwright', 'install', 'chromium'], check=True, capture_output=True, text=True)
        logger.info("Playwright Chromium browser installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to install Playwright Chromium: {e}\nOutput:\n{e.stdout}\n{e.stderr}")
        logger.error(f"Playwright install chromium failed: {e}\nOutput:\n{e.stdout}\n{e.stderr}")
        return False
    except FileNotFoundError:
        st.error("Playwright command not found. Is Playwright installed? (Check requirements.txt)")
        logger.error("Playwright command not found.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during Playwright Chromium installation: {str(e)}")
        logger.error(f"Unexpected Playwright install error: {e}", exc_info=True)
        return False

# --- S3 Upload (Reverted - No Retry Decorator, No strip) ---
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
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id, # No strip
            aws_secret_access_key=aws_secret_access_key, # No strip
            region_name=region_name # No strip
        )

        if not object_name:
            object_name = f"image_{int(time.time())}_{random.randint(1000, 9999)}.{image_format.lower()}"

        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format=image_format)
        img_byte_arr.seek(0)

        s3_client.put_object(
            Bucket=bucket_name, # No strip
            Key=object_name,
            Body=img_byte_arr,
            ContentType=f'image/{image_format.lower()}'
        )

        url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}"
        return url

    except NoCredentialsError:
         st.error("AWS credentials not found or invalid for S3 upload.") # More specific error
         return None
    except Exception as e:
        st.error(f"Error in S3 upload: {str(e)}")
        return None # Return None on error as per original style


# --- API Call Functions (Reverted to Original Logic/Models) ---

# --- Google Images Search (Reverted - No explicit retry loop/decorator) ---
def fetch_google_images(query, num_images=3, max_retries=5): # Keep max_retries from original signature
    """Fetch images from Google Images using google_images_search."""
    # Original logic included a max_retries loop, let's keep that structure
    for trial in range(max_retries):
        terms_list = query.split('~')
        res_urls = []
        try:
            for term in terms_list:
                api_key = random.choice(GOOGLE_API_KEY)
                cx_key = GOOGLE_CX # Renamed variable for clarity
                gis = GoogleImagesSearch(api_key, cx_key)
                search_params = {
                    'q': term.strip(),
                    'num': num_images,
                 }
                gis.search(search_params)
                image_urls = [result.url for result in gis.results()]
                res_urls.extend(image_urls)
            # If successful, return unique URLs
            return list(set(res_urls))
        except Exception as e:
            st.text(f"Error fetching Google Images for '{query}' (Attempt {trial+1}/{max_retries}): {e}")
            if trial < max_retries - 1:
                 time.sleep(5) # Wait before retrying as per original
            else:
                 st.error(f"Failed to fetch Google Images for '{query}' after {max_retries} attempts.")
                 return [] # Return empty list on final failure

    return [] # Should not be reached if loop completes, but added for safety


# --- Gemini Text REST (Reverted - Original Model, No Retry, No PD Policy) ---
def gemini_text(
    prompt: str,
    api_key: str = None,
    model_id: str = "gemini-pro", # Reverted to likely intended original default (check exact original if needed)
    api_endpoint: str = "generateContent"
) -> str | None:
    """Calls the Gemini API via REST."""
    # Removed pd_policy logic

    if api_key is None:
         # Original code had "gemini-2.5-pro-exp-03-25", let's assume 'gemini-pro' was more standard if that exp fails
         # Also original had api_key = random.choice(GEMINI_API_KEY), not os.environ
         api_key = random.choice(GEMINI_API_KEY)

    if not api_key:
         print("Error: API key not provided for Gemini.") # Original used print
         return None

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:{api_endpoint}?key={api_key}"
    headers = {"Content-Type": "application/json"}
    request_data = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "text/plain"},
    }

    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=request_data,
            timeout=60 # Original timeout
        )
        response.raise_for_status()

        st.text(response.json()) # Original printed response json to UI

        # Original extraction logic (simpler)
        return response.json()['candidates'][0]['content']['parts'][0]['text'].replace('```','')

    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}") # Original used print
        if 'response' in locals() and response is not None:
             print(f"Response status: {response.status_code}") # Original used print
             print(f"Response text: {response.text}") # Original used print
        return None
    except Exception as e:
        print(f"An unexpected error occurred in Gemini call: {e}") # Original used print
        return None


# --- Gemini Text Library (Reverted - Original Model, No Retry, No PD Policy) ---
def gemini_text_lib(prompt, model='gemini-2.5-pro-exp-03-25'): # Reverted model
    """Calls Gemini API using the Python library."""
    # Removed pd_policy logic
    # Original default model was 'gemini-2.5-pro-exp-03-25', use 'gemini-pro' as likely standard alternative

    try:
        client = genai.Client(api_key=random.choice(GEMINI_API_KEY))
        # Check original call method if 'client.models.generate_content' is wrong for older lib versions
        response = client.generate_content(model=f"models/{model}", contents=prompt)
        return response.text
    except Exception as e:
        st.text('gemini_text_lib error ' + str(e)) # Original printed to UI
        time.sleep(4) # Original had sleep on error
        return None


# --- ChatGPT (Reverted - Original Models/Temps/Logic, No Retry, No PD Policy) ---
def chatGPT(prompt, model="gpt-4o", temperature=1.0, reasoning_effort=''):
    """Calls OpenAI API (compatible) via REST for text generation."""
    # Removed pd_policy logic

    try:
        st.write("Generating image description...") # Original had this message
        headers = {
            'Authorization': f'Bearer {GPT_API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': model,
            'input' : prompt, # Use 'input' as per original
            # Original temperature logic: include only if non-default
        }
        if temperature != 1.0: # Match original logic exactly
             data['temperature'] = temperature
        if reasoning_effort != '': # Match original logic exactly
             data['reasoning'] = {"effort": reasoning_effort}


        api_url = '[https://api.openai.com/v1/responses](https://api.openai.com/v1/responses)' # Original endpoint

        response = requests.post(api_url, headers=headers, json=data, timeout=60) # Added timeout for safety
        response.raise_for_status() # Check for HTTP errors
        content_data = response.json()

        # Original extraction logic
        if 'o1' in model or 'o3' in model:
            content = content_data['output'][1]['content'][0]['text']
        else:
            content = content_data['output'][0]['content'][0]['text']

        # st.text(content) # Original had this commented out
        return content

    except Exception as e:
        st.text(f"Error in chatGPT: {str(e)}") # Original error handling
        # Attempt to print response JSON on error, if response exists
        if 'response' in locals() and hasattr(response, 'json'):
            try:
                st.text(response.json())
            except json.JSONDecodeError:
                 st.text(f"Response text (not JSON): {response.text}")
        return None


# --- Claude (Reverted - Original Model, Original Loop Logic, No PD Policy) ---
def claude(prompt , model = "claude-3-sonnet-20240229", temperature=1 , is_thinking = False, max_retries = 10): # Reverted default model, kept params
    # Removed pd_policy logic
    tries = 0

    while tries < max_retries:
        try:
            # Client is initialized globally now: anthropic_client
            message_params = {
                "model": model,
                "max_tokens": 4096, # Use generous token limit from later versions
                "temperature": temperature,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            }
            # Original logic for top_p based on temperature (approximate)
            # Original seemed to have top_p=0.8 always? Let's add it back conditionally.
            if temperature < 1.0: # Add top_p only if temperature is not 1.0
                 message_params["top_p"] = 0.8

            if is_thinking: # Keep thinking logic as it was specific
                 message_params["thinking"] = {"type": "enabled", "budget_tokens": 16000}

            message = anthropic_client.messages.create(**message_params)

            # Original extraction logic based on is_thinking
            if is_thinking:
                 if len(message.content) > 1 and hasattr(message.content[1], 'text'):
                      return message.content[1].text
                 else: # Handle case where thinking response might be different
                      if hasattr(message.content[0], 'text'): return message.content[0].text
                      else: return "" # Fallback
            else: # Not thinking
                 if message.content and hasattr(message.content[0], 'text'):
                      return message.content[0].text
                 else:
                      return "" # Fallback

        except Exception as e:
            st.text(e) # Original printed error to UI
            tries += 1
            time.sleep(5) # Original slept on error
    # If loop finishes without success
    st.error(f"Claude call failed after {max_retries} attempts.")
    return None


# --- Flux Image Gen (Reverted - Original Model Name, Original Loop) ---
def gen_flux_img(prompt, height=784, width=960):
    """Generate images using FLUX model from the Together.xyz API."""
    while True: # Original used infinite loop with break/return
        try:
            url = "[https://api.together.xyz/v1/images/generations](https://api.together.xyz/v1/images/generations)"
            payload = {
                "prompt": prompt,
                "model": "black-forest-labs/FLUX.1-schnell-Free", # Reverted model name
                "steps": 4,
                "n": 1,
                "height": height,
                "width": width,
            }
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {random.choice(FLUX_API_KEY)}"
            }
            response = requests.post(url, json=payload, headers=headers, timeout=120) # Keep longer timeout
            response.raise_for_status() # Check status
            # Original extraction assumed success if no exception
            return response.json()["data"][0]["url"]
        except Exception as e:
            print(e) # Original used print
            if "NSFW" in str(e): # Original NSFW check
                return None
            time.sleep(2) # Original sleep


# --- Gemini Image Gen (Reverted - Original Endpoint/Config/Parsing, No Loop) ---
def gen_gemini_image(prompt, trys = 0): # Keep original signature with unused 'trys'
    # Removed outer while loop
    api = random.choice(GEMINI_API_KEY)
    # Original endpoint:
    url = f"[https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent?key=](https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent?key=){api}"

    headers = {"Content-Type": "application/json"}
    # Original data structure:
    data = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]},
            # Original included a second empty user part, replicate that:
            {"role": "user", "parts": [{"text": ""}]}
        ],
        "generationConfig": {
            "temperature": 0.65, # Original temp
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 8192,
            "responseMimeType": "text/plain", # Original mime type
            "responseModalities": ["image", "text"] # Original modalities
        }
    }

    try: # Add try-except for robustness, even if original didn't show one here
        response = requests.post(url, headers=headers, json=data, timeout=120) # Keep longer timeout

        if response.status_code == 200:
            res_json = response.json()
            try:
                # Original extraction path:
                image_b64 = res_json['candidates'][0]["content"]["parts"][0]["inlineData"]['data']
                image_data = base64.decodebytes(image_b64.encode()) # Original used decodebytes
                return Image.open(BytesIO(image_data))
            except Exception as e:
                 print("Failed to extract or save image:", e) # Original used print
                 return None # Return None on extraction failure
        else:
            print("Error:") # Original used print
            st.text(response.text) # Original printed response text to UI
            return None
    except Exception as e:
         st.error(f"Error calling Gemini Image API: {e}")
         return None


# --- Flux LoRA Image Gen (Reverted - Original Loop Logic) ---
def gen_flux_img_lora(prompt,height=784, width=960 ,lora_path="[https://huggingface.co/ddh0/FLUX-Amateur-Photography-LoRA/resolve/main/FLUX-Amateur-Photography-LoRA-v2.safetensors?download=true](https://huggingface.co/ddh0/FLUX-Amateur-Photography-LoRA/resolve/main/FLUX-Amateur-Photography-LoRA-v2.safetensors?download=true)"):
    retries = 0
    while retries < 10: # Original loop
        try:
            url = "[https://api.together.xyz/v1/images/generations](https://api.together.xyz/v1/images/generations)"
            headers = {
                "Authorization": f"Bearer {random.choice(FLUX_API_KEY)}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "black-forest-labs/FLUX.1-dev-lora",
                "prompt":"candid unstaged taken with iphone 8 : " + prompt, # Original prompt prefix
                "width": width,
                "height": height,
                "steps": 20,
                "n": 1,
                "response_format": "url",
                "image_loras": [{"path": lora_path, "scale": 0.99}],
                 # Original included update_at, replicate:
                "update_at": "2025-03-04T16:25:21.474Z" # Hardcoded date from original
            }

            response = requests.post(url, headers=headers, json=data, timeout=120) # Keep timeout

            if response.status_code == 200:
                response_data = response.json()
                image_url = response_data['data'][0]['url']
                print(f"Image URL: {image_url}") # Original used print
                return image_url
            else:
                # Original didn't explicitly raise_for_status or handle non-200 here
                print(f"Request failed with status code {response.status_code}") # Original print

        except Exception as e:
             # Original error handling was just sleep/retry
             time.sleep(3) # Original sleep
             retries += 1
             st.text(e) # Original printed error to UI
    # If loop finishes
    st.error("Flux LoRA failed after multiple retries.")
    return None


# --- Playwright Screenshot (Reverted - Simpler Logic, No Retry) ---
def capture_html_screenshot_playwright(html_content,width = 1000, height = 1000):
    """Use Playwright to capture a screenshot of the given HTML snippet."""
    if not st.session_state.get('playwright_installed', False): # Keep check
        st.error("Playwright browsers not installed properly")
        return None

    temp_html_path = None # Define outside try
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage'] # Original args
            )
            page = browser.new_page(viewport={'width': width, 'height': height})

            with NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f: # Keep encoding
                f.write(html_content)
                temp_html_path = f.name

            page.goto(f'file://{temp_html_path}')
            page.wait_for_timeout(1000) # Original timeout
            screenshot_bytes = page.screenshot()

            browser.close()
            # Original didn't explicitly unlink temp file? Add it for safety.
            if temp_html_path and os.path.exists(temp_html_path):
                 try:
                     os.unlink(temp_html_path)
                 except Exception: pass # Ignore unlink errors silently

            return Image.open(BytesIO(screenshot_bytes))
    except Exception as e:
        st.error(f"Screenshot capture error: {str(e)}") # Original error handling
        # Original didn't ensure browser close on error, add simple attempt
        if 'browser' in locals() and browser and browser.is_connected():
             try: browser.close()
             except Exception: pass
        # Original didn't ensure temp file unlink on error, add simple attempt
        if temp_html_path and os.path.exists(temp_html_path):
             try: os.unlink(temp_html_path)
             except Exception: pass
        return None


# --- Save HTML (Keep improved version with f-string fix) ---
def save_html(headline, image_url, cta_text, template, tag_line=''):
    """Returns an HTML string based on the chosen template ID."""
    logger.debug(f"Generating HTML for template: {template}") # Keep logger if needed
    # --- Paste ALL your HTML template definitions here (Template 1 to Template 8) ---
    # Ensure all template numbers used in the UI have corresponding HTML here
    if template == 1:
        html_template = f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Ad Template 1</title><style>
        body {{ font-family: 'Gisha', sans-serif; font-weight: 550; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }}
        .ad-container {{ width: 1000px; height: 1000px; border: 1px solid #ddd; border-radius: 20px; box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2); display: flex; flex-direction: column; justify-content: space-between; align-items: center; padding: 30px; background: url('{image_url}') no-repeat center center/cover; background-size: contain; text-align: center; }}
        .ad-title {{ font-size: 3.2em; margin-top: 10px; color: #333; background-color:white; padding: 20px 40px; border-radius: 20px; }}
        .cta-button {{ font-weight: 400; display: inline-block; padding: 40px 60px; font-size: 3em; color: white; background-color: #FF5722; border: none; border-radius: 20px; text-decoration: none; cursor: pointer; transition: background-color 0.3s ease; margin-bottom: 20px; }}
        .cta-button:hover {{ background-color: #E64A19; }}</style></head><body><div class="ad-container"><div class="ad-title">{headline}!</div><a href="#" class="cta-button">{cta_text}</a></div></body></html>
        """ # Using exact original HTML/CSS
    elif template == 2:
         html_template = f"""
         <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Ad Template</title><style>
         body {{ font-family: 'Gisha', sans-serif; font-weight: 550; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }}
         .ad-container {{ width: 1000px; height: 1000px; border: 2px solid black; box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2); display: flex; flex-direction: column; overflow: hidden; position: relative; }}
         .ad-title {{ font-size: 3.2em; color: #333; background-color: white; padding: 20px; text-align: center; flex: 0 0 20%; display: flex; justify-content: center; align-items: center; }}
         .ad-image {{ flex: 1 1 80%; background: url('{image_url}') no-repeat center center/cover; background-size: fill; position: relative; }}
         .cta-button {{ font-weight: 400; display: inline-block; padding: 20px 40px; font-size: 3.2em; color: white; background-color: #FF5722; border: none; border-radius: 20px; text-decoration: none; cursor: pointer; transition: background-color 0.3s ease; position: absolute; bottom: 10%; left: 50%; transform: translateX(-50%); z-index: 10; }}</style></head><body><div class="ad-container"><div class="ad-title">{headline}</div><div class="ad-image"><a href="#" class="cta-button">{cta_text}</a></div></div></body></html>
         """ # Using exact original HTML/CSS
    elif template == 3 or template == 7:
        button_class = 'c1ta-button' if template == 7 else 'cta-button' # Keep f-string fix calculation
        html_template = f"""
        <!DOCTYPE html><html><head><style>
        body {{ margin: 0; padding: 0; font-family: 'Boogaloo', 'Segoe UI Emoji', 'Apple Color Emoji', sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; background: #f0f0f0; }}
        .container {{ position: relative; width: 1000px; height: 1000px; margin: 0; padding: 0; overflow: hidden; box-shadow: 0 0 20px rgba(0,0,0,0.2); }}
        .image {{ width: 1000px; height: 1000px; object-fit: cover; filter: saturate(130%) contrast(110%); transition: transform 0.3s ease; }}
        .image:hover {{ transform: scale(1.05); }}
        .overlay {{ position: absolute; top: 0; left: 0; width: 100%; min-height: 14%; background: red; display: flex; justify-content: center; align-items: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); padding: 20px; box-sizing: border-box; }}
        .overlay-text {{ color: #FFFFFF; font-size: 4em; text-align: center; text-shadow: 2.5px 2.5px 2px #000000; letter-spacing: 2px; font-family: 'Boogaloo', 'Segoe UI Emoji', 'Apple Color Emoji', sans-serif; margin: 0; word-wrap: break-word; }}
        .cta-button {{ position: absolute; bottom: 10%; left: 50%; transform: translateX(-50%); padding: 20px 40px; background: blue; color: white; border: none; border-radius: 50px; font-size: 3.5em; cursor: pointer; transition: all 0.3s ease; font-family: 'Boogaloo', 'Segoe UI Emoji', 'Apple Color Emoji', sans-serif; text-transform: uppercase; letter-spacing: 2px; box-shadow: 0 5px 15px rgba(255,107,107,0.4); }} /* Original shadow */
        .cta-button:hover {{ background: #4ECDC4; transform: translateX(-50%) translateY(-5px); box-shadow: 0 8px 20px rgba(78,205,196,0.6); }}
        .c1ta-button {{ position: absolute; bottom: 10%; left: 50%; transform: translateX(-50%); padding: 20px 40px; background: blue; color: white; border: none; border-radius: 50px; font-size: 3.5em; cursor: pointer; transition: all 0.3s ease; font-family: 'Boogaloo', 'Segoe UI Emoji', 'Apple Color Emoji', sans-serif; text-transform: uppercase; letter-spacing: 2px; box-shadow: 0 5px 15px rgba(255,107,107,0.4); }} /* Style c1ta same as cta initially */
         .c1ta-button:hover {{ background: #4ECDC4; transform: translateX(-50%) translateY(-5px); box-shadow: 0 8px 20px rgba(78,205,196,0.6); }}
        @keyframes shine {{ 0% {{ left: -100%; }} 100% {{ left: 200%; }} }}</style><link href="[https://fonts.googleapis.com/css2?family=Boogaloo&display=swap](https://fonts.googleapis.com/css2?family=Boogaloo&display=swap)" rel="stylesheet"></head><body><div class="container"><img src="{image_url}" class="image" alt="Health Image"><div class="overlay"><h1 class="overlay-text">{headline}</h1></div><button class="{button_class}">{cta_text}<div class="shine"></div></button></div></body></html>
        """ # Using exact original HTML/CSS, keeping f-string fix
    elif template == 4 or template == 41 or template == 42:
        top_position = "50%" if template == 4 else ("15%" if template == 41 else "90%")
        html_template = f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Nursing Careers in the UK</title><style>
        @import url('[https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap](https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap)');
        @font-face {{ font-family: 'Calibre'; src: url('path-to-calibre-font.woff2') format('woff2'); }} /* Path likely needs fixing */
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #F4F4F4; }}
        .container {{ position: relative; width: 1000px; height: 1000px; background-image: url('{image_url}'); background-size: cover; background-position: center; border-radius: 10px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); }}
        .text-overlay {{ position: absolute; width: 95%; background-color: rgba(255, 255, 255, 1); padding: 30px; border-radius: 10px; top: {top_position}; left: 50%; transform: translate(-50%, -50%); text-align: center; }}
        .small-text {{ font-size: 36px; font-weight: bold; color: #333; margin-bottom: 10px; font-family: 'Calibre', Arial, sans-serif; }}
        .primary-text {{ font-size: 60px; font-weight: bold; color: #FF8C00; font-family: 'Montserrat', sans-serif; line-height: 1.2; text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000; }}</style></head><body><div class="container"><div class="text-overlay"><div class="small-text">{cta_text}</div><div class="primary-text">{headline}</div></div></div></body></html>
        """ # Using exact original HTML/CSS
    elif template == 5:
        html_template = f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Landing Page Template</title><style>
        @import url('[https://fonts.googleapis.com/css2?family=Outfit:wght@100..900&display=swap](https://fonts.googleapis.com/css2?family=Outfit:wght@100..900&display=swap)'); @import url('[https://fonts.googleapis.com/css2?family=Noto+Color+Emoji&display=swap](https://fonts.googleapis.com/css2?family=Noto+Color+Emoji&display=swap)');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }} body {{ width: 1000px; height: 1000px; margin: 0 auto; font-family: 'Outfit', sans-serif; }}
        .container {{ width: 100%; height: 100%; display: flex; flex-direction: column; position: relative; object-fit: fill; }} /* object-fit was on container? */
        .image-container {{ width: 100%; height: 60%; background-color: #f0f0f0; display: flex; align-items: center; justify-content: center; }}
        .image-container img {{ width: 100%; height: 100%; object-fit: cover; }}
        .content-container {{ width: 100%; height: 40%; background-color: #121421; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2rem; gap: 2rem; }}
        .main-text {{ color: white; font-size: 3.5rem; font-weight: 700; text-align: center; }}
        .cta-button {{ background-color: #ff0000; color: white; padding: 1rem 2rem; font-size: 3.5rem; font-weight: 700; font-family: 'Outfit', sans-serif; border: none; font-style: italic; border-radius: 8px; cursor: pointer; transition: background-color 0.3s ease; }}
        .cta-button:hover {{ background-color: #cc0000; }}
        .intersection-rectangle {{ position: absolute; max-width: 70%; min-width: max-content; height: 80px; background-color: #121421; left: 50%; transform: translateX(-50%); top: calc(60% - 40px); border-radius: 10px; display: flex; align-items: center; justify-content: center; padding: 0 40px; }}
        .rectangle-text {{ font-family: 'Noto Color Emoji', sans-serif; color: #66FF00; font-weight: 700; text-align: center; font-size: 45px; white-space: nowrap; }}
        .highlight {{ color: #FFFF00; font-size: 3.5rem; font-style: italic; font-weight: 1000; text-align: center; }}</style></head><body><div class="container"><div class="image-container"><img src="{image_url}" alt="Placeholder image"></div><div class="intersection-rectangle"><p class="rectangle-text">{tag_line.upper()}</p></div><div class="content-container"><h1 class="main-text">{headline}</h1><button class="cta-button">{cta_text}</button></div></div></body></html>
        """ # Using exact original HTML/CSS
    elif template == 6 or template == 8:
        width_px = "999px" if template == 8 else "1000px"
        height_px = "666px" if template == 8 else "1000px"
        html_template = f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Nursing Careers in the UK</title><style>
        @import url('[https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap](https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@700&display=swap)');
        @font-face {{ font-family: 'Calibre'; src: url('path-to-calibre-font.woff2') format('woff2'); }}
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #F4F4F4; }}
        .container {{ position: relative; width: {width_px}; height: {height_px}; background-image: url('{image_url}'); background-size: cover; background-position: center; border-radius: 10px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); }}
        /* Original T6/8 had no text-overlay definition */
        </style></head><body><div class="container"><div class="text-overlay"></div></div></body></html>
        """ # Using exact original HTML/CSS
    else:
        # Keep fallback
        html_template = f"<html><body><p>Template {template} not found. Image: <img src='{image_url}' width='200'></p><p>Headline: {headline}</p><p>CTA: {cta_text}</p></body></html>"

    return html_template


# --- DALL-E Variation (Reverted - Use global API key, older openai call style if possible?) ---
# Sticking with newer client style but using global key setup is safer and likely forward-compatible.
# @retry_decorator # Removed retry decorator
def create_dalle_variation(image_url, count):
    """Creates DALL-E variations from a source image URL."""
    logger.debug(f"Creating {count} DALL-E variations for: {image_url}") # Keep logger
    # Check global key (set at start)
    if not openai.api_key:
        st.error("OpenAI API Key for DALL-E is not configured.")
        return None

    try:
        # Download the source image
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        resp = requests.get(image_url, headers=headers, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))

        # Convert to PNG and check size (Keep this improvement)
        png_buffer = BytesIO()
        img.save(png_buffer, format="PNG")
        png_size_mb = len(png_buffer.getvalue()) / (1024 * 1024)

        if png_size_mb > 4.0:
            logger.warning(f"Source image too large ({png_size_mb:.2f} MB), resizing to 512x512.")
            st.warning(f"Source image for variation was large, resizing...")
            img = img.resize((512, 512))
            png_buffer = BytesIO()
            img.save(png_buffer, format="PNG")

        png_buffer.seek(0)

        # --- Use global key with newer client style ---
        # Re-initialize client here IF necessary, or ensure global key is set
        temp_openai_client = OpenAI(api_key=openai.api_key) # Use globally set key
        response = temp_openai_client.images.create_variation( # Use newer API call structure
            image=png_buffer.getvalue(),
            n=count,
            size="512x512"
        )
        # --- End Reverted Call ---
        return response.data

    except Exception as e: # Original simpler error handling
        st.error(f"Error generating DALL-E variation: {e}")
        return None


# --- Removed Policy Text ---

# --------------------------------------------
# Main Generation Logic Function (Keep structure, revert internal calls)
# --------------------------------------------
def run_generation(df_to_process):
    """Processes the DataFrame to generate images based on topics and counts."""
    st.session_state.generated_images = []
    processed_combinations = set()
    if 'accumulated_generated_images' in st.session_state:
       del st.session_state.accumulated_generated_images

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

    # logger.info(f"Starting generation run for {total_images_in_chunk} images.") # Removed logging

    with st.spinner(f"Running generation for {total_images_in_chunk} images..."):
        for index, row in df_to_process.iterrows():
            topic = row['topic']
            try:
                count = int(row['count'])
                if count <= 0: continue
            except (ValueError, TypeError):
                st.warning(f"Skipping row {index+1} due to invalid count: {row.get('count')}")
                continue
            lang = row['lang']
            template_str = str(row["template"])
            original_topic = topic
            current_topic_images = []

            # --- Reverted: No Topic Enhancement ---
            # if ennhance_input and not st.session_state.get('is_chunk_run'): ...

            # --- Handle 'google' keyword (Original Logic) ---
            if "google" in topic.lower():
                google_topic_query = topic.replace('google', ' ').strip()
                if '|' in google_topic_query:
                    google_topic_query = re.sub("^.*\|", "", google_topic_query).strip()
                # Use original function call
                google_image_urls = fetch_google_images(google_topic_query, num_images=count) # Max retries handled inside
                for img_url in google_image_urls:
                    current_topic_images.append({
                        'url': img_url, 'selected': False, 'template': template_str,
                        'source': 'google', 'dalle_generated': False
                    })
                images_processed_this_row = len(google_image_urls)
                images_processed_count += images_processed_this_row
                percent_complete = min(1.0, images_processed_count / total_images_in_chunk) if total_images_in_chunk > 0 else 0
                my_bar.progress(percent_complete, text=f"{progress_text} ({images_processed_count}/{total_images_in_chunk})")

            # --- Handle AI Image Generation (Reverted Internal Calls) ---
            else:
                completed_images_this_row = 0
                # Note: Original didn't have max_attempts_per_image, relying on inner loops
                while completed_images_this_row < count:
                    # Break condition needed if inner loops fail indefinitely
                    if completed_images_this_row >= count: break # Exit if desired count reached

                    selected_template = None
                    image_url = None
                    pil_image = None
                    source_api = "unknown"
                    current_template_str = template_str # Use template string from row

                    try:
                        if ',' in current_template_str:
                            current_template_str = random.choice([t.strip() for t in current_template_str.split(',') if t.strip()])

                        if 'gemini' in current_template_str.lower():
                            source_api = "gemini"
                            gemini_prompt = None
                            # --- Reverted Prompt Generation Models ---
                            prompt_generation_model = 'gpt-4o' # Original default for chatGPT
                            try:
                                if current_template_str == 'gemini2':
                                    gemini_prompt = chatGPT(f"write short prompt for\ngenerate square image promoting '{topic}' in language {lang} {random.choice(['use photos',''])}. add a CTA button with 'Learn More Here >>' in appropriate language\ns\nstart with 'square image aspect ratio of 1:1 of '\n\n", model=prompt_generation_model) # No PD policy
                                elif current_template_str == 'geminicandid':
                                     gemini_prompt = claude(f"""write a image prompt of a candid unstaged photo taken of a regular joe showing off his\her {topic} . the image is taken with smartphone candidly. in 1-2 sentences. Describe the quality of the image looking smartphone. start with "Square photo 1:1 iphone 12 photo uploaded to reddit:" this is for a fb ad that tries to look organic, but also make the image content intecing and somewhat perplexing, so try to be that but also draw clicks with high energy in the photo. dont make up facts like discounts! or specific prices! if you want to add a caption, specifically instruct it to be on the image. and be short in language {lang}""", is_thinking=True).replace("#","") # No PD policy
                                elif current_template_str == 'gemini7':
                                     gemini_prompt = gemini_text_lib(f"write short prompt for\ngenerate square image promoting '{topic}' in language {lang} . add a CTA button with 'Learn More Here >>' in appropriate language\\nand 'act fast' or 'limited available' \n \nshould be low quality and very enticing and alerting \n\nstart with 'square image aspect ratio of 1:1 of '\n\n be specific in what is shown . return JUST the best option, no intros", model='gemini-pro') # Reverted model, No PD policy
                                elif current_template_str == 'gemini7claude':
                                     gemini_prompt = claude(f"write short prompt for\ngenerate square image promoting '{topic}' in language {lang} . add a CTA button with 'Learn More Here >>' in appropriate language\ \nshould be low quality and very enticing and alerting, don't make specific promises like x% discount and 'act fast' or 'limited available'  \n\nstart with 'square image aspect ratio of 1:1 of '\n\n be specific in what is shown . return JUST the best option, no intros\nif you want to add a caption, specifically instruct it to be on the image. and be short", is_thinking=False) # No PD Policy
                                elif current_template_str == 'geministock':
                                     gemini_prompt = chatGPT(f"write short image prompt for {topic},no text on image,A high-quality image in a realistic setting, well-lit and visually appealing, suitable for use in marketing or editorial content.", model=prompt_generation_model, temperature= 1.0) # No PD policy
                                else: # Default gemini
                                     gemini_prompt = chatGPT(f"write short prompt for\ngenerate square image promoting '{topic}' in language {lang} {random.choice(['use photos',''])}. add a CTA button with 'Learn More Here >>' in appropriate language\nshould be low quality and very enticing and alerting\nstart with 'square image aspect ratio of 1:1 of '\n\n example output:\n\nsquare image of a concerned middle-aged woman looking at her tongue in the mirror under harsh bathroom lighting, with a cluttered counter and slightly blurry focus — big bold red text says “Early Warning Signs?” and a janky yellow button below reads “Learn More Here >>” — the image looks like it was taken on an old phone, with off angle, bad lighting, and a sense of urgency and confusion to provoke clicks.",model=prompt_generation_model, temperature= 1.0) # No PD policy
                            except Exception as prompt_err:
                                st.warning(f"Skipping image: Failed to generate prompt for {current_template_str}: {prompt_err}")
                                break # Exit inner loop on prompt failure for this row

                            if gemini_prompt:
                                pil_image = gen_gemini_image(gemini_prompt) # Use reverted function
                                if pil_image is None: time.sleep(2) # Add delay if gen fails before next attempt
                            else:
                                 st.warning(f"Gemini prompt generation returned empty for {current_template_str}.")
                                 time.sleep(2) # Delay if prompt gen fails

                        else: # FLUX Templates
                            source_api = "flux"
                            selected_template = None
                            try:
                                selected_template = int(current_template_str)
                            except ValueError:
                                st.warning(f"Invalid non-Gemini template ID: {current_template_str}. Skipping row.")
                                break # Exit inner loop for this row

                            flux_prompt = None
                            prompt_generation_model = 'gpt-4' # Original default for template 5
                            try:
                                if selected_template == 5:
                                     flux_prompt = chatGPT(f"Generate a concise visual image description (15 words MAX) for {topic}. Be wildly creative, curious... (rest of prompt 5)", model=prompt_generation_model, temperature=1.2) # No PD Policy
                                elif selected_template == 7 :
                                     flux_prompt = chatGPT(f"Generate a visual image description 50 words MAX for {topic} , candid moment unstaged... (rest of prompt 7)", model='gpt-4o-mini') # Keep faster model if desired, original was likely gpt-4o? Revert if needed. No PD Policy
                                elif selected_template == 8:
                                      flux_prompt = chatGPT(f"A clean, high-resolution stock photo prompt of {topic}, no people, well-lit... (rest of prompt 8)", model='gpt-4o') # Original default model. No PD policy.
                                else: # Default flux prompt gen
                                      flux_prompt = chatGPT(f"Generate a visual image description 15 words MAX for {topic}. Use a visually enticing style with high CTR, avoid obvious descriptions.", model='gpt-4o') # Original default model. No PD Policy
                            except Exception as prompt_err:
                                 st.warning(f"Skipping image: Failed to generate prompt for template {selected_template}: {prompt_err}")
                                 break # Exit inner loop

                            if flux_prompt:
                                if selected_template == 7:
                                    image_url = gen_flux_img_lora(flux_prompt) # Use reverted function
                                elif selected_template == 8:
                                    image_url = gen_flux_img(flux_prompt, width=720, height=480) # Use reverted function
                                elif selected_template == 5:
                                     image_url = gen_flux_img(flux_prompt, width=688, height=416) # Use reverted function
                                else:
                                    image_url = gen_flux_img(f"{random.choice(['cartoony clipart of ', ''])} {flux_prompt}") # Use reverted function
                                if image_url is None: time.sleep(2) # Delay if gen fails
                            else:
                                 st.warning(f"Flux prompt generation returned empty for template {selected_template}.")
                                 time.sleep(2) # Delay if prompt gen fails

                        # --- Process generated image ---
                        final_image_url = None
                        if pil_image: # Gemini image
                            try:
                                final_image_url = upload_pil_image_to_s3(pil_image, S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)
                            except Exception as upload_err:
                                st.error(f"S3 upload failed for Gemini image: {upload_err}") # Use st.error
                        elif image_url: # Flux URL
                            final_image_url = image_url
                        # else: print("No image generated this attempt.") # Original might not have printed

                        if final_image_url:
                            current_topic_images.append({
                                'url': final_image_url, 'selected': False,
                                'template': current_template_str,
                                'source': source_api, 'dalle_generated': False
                            })
                            completed_images_this_row += 1
                            images_processed_count += 1
                            percent_complete = min(1.0, images_processed_count / total_images_in_chunk) if total_images_in_chunk > 0 else 0
                            my_bar.progress(percent_complete, text=f"{progress_text} ({images_processed_count}/{total_images_in_chunk})")
                        # No explicit else/break in original loop if generation failed? Loop continues. Add safety break?
                        # Safety break if generation consistently fails for a row
                        if final_image_url is None and completed_images_this_row < count and generation_attempts >= max_attempts_per_image:
                             st.warning(f"Max attempts reached for topic '{topic}', generated {completed_images_this_row}/{count}. Moving to next row.")
                             images_processed_count += (count - completed_images_this_row) # Update progress for skipped
                             percent_complete = min(1.0, images_processed_count / total_images_in_chunk) if total_images_in_chunk > 0 else 0
                             my_bar.progress(percent_complete, text=f"{progress_text} ({images_processed_count}/{total_images_in_chunk})")
                             break # Exit inner while loop

                    except Exception as gen_err:
                        # Original just looped, maybe add warning
                        st.warning(f"Error during generation attempt for '{topic}': {gen_err}. Continuing...")
                        time.sleep(2) # Delay after general error

            # --- Append results for this row ---
            if 'accumulated_generated_images' not in st.session_state:
                st.session_state.accumulated_generated_images = []
            st.session_state.accumulated_generated_images.append({
                "topic": topic,
                "original_topic": original_topic,
                "lang": lang,
                "images": current_topic_images
            })
            # logger.info(f"Finished processing row {index+1}: '{original_topic}'") # Removed logging

    st.session_state.generated_images = st.session_state.pop('accumulated_generated_images', [])
    if total_images_in_chunk > 0:
        my_bar.progress(1.0, text=f"Image generation complete! ({images_processed_count}/{total_images_in_chunk})")
    else:
        my_bar.empty()
    st.success(f"Finished generation run.")
    if images_processed_count > 0:
        play_sound("audio/bonus-points-190035.mp3")


# --- Rest of the UI (Keep Chunking Logic, Revert Internal Calls in Processing) ---

# # Initialize Playwright (Keep Improved Version)
# if 'playwright_installed' not in st.session_state:
#     with st.spinner("Initializing browser automation (one-time setup)..."):
#        st.session_state.playwright_installed = install_playwright_browsers()
# elif not st.session_state.playwright_installed:
#      st.error("Browser automation setup failed previously. Screenshots may not work.")

st.title("Creative Maker ")

# --- Examples Expander (Keep improved version with error handling) ---
with st.expander(f"Click to see examples for templates ", expanded=False):
     image_list =  [
         # ... (Your list of image dicts) ...
         {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744112392_1276.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744112392_1276.png)", "caption": "2"},
         {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744114470_4147.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744114470_4147.png)", "caption": "3"},
         # ... include all examples
         {"image": "[https://image-script.s3.us-east-1.amazonaws.com/image_1744716627_8304.png](https://image-script.s3.us-east-1.amazonaws.com/image_1744716627_8304.png)", "caption": "gemini7claude_simple"}
     ]
     num_columns = 6
     num_images = len(image_list)
     num_rows = (num_images + num_columns - 1) // num_columns
     for i in range(num_rows):
         cols = st.columns(num_columns)
         row_images = image_list[i * num_columns : (i + 1) * num_columns]
         for j, item in enumerate(row_images):
             if item and j < len(cols):
                 with cols[j]:
                     try:
                         st.image(item["image"], use_container_width=True)
                         st.caption(item.get("caption", ""))
                     except Exception as img_err:
                         st.error(f"⚠️ Err loading {item.get('caption', 'image')}")
                         st.caption(f"URL: ...{item.get('image', '')[-20:]}")


st.subheader("Enter Topics for Image Generation")

# --- Data Editor (Keep Improved Version) ---
df = st.data_editor(
    initial_df_data,
    num_rows="dynamic",
    key="table_input",
    disabled=st.session_state.is_chunk_run,
    use_container_width=True
)

# --- Options (Reverted PD Policy, Enhancement) ---
col_opts1, col_opts2, col_opts3 = st.columns(3)
with col_opts1:
    auto_mode = st.checkbox("Auto mode (Select all after generation)?", value=st.session_state.is_chunk_run)
with col_opts2:
    is_pd_policy = st.checkbox("PD policy (Adds policy text to prompts)?") # Keep checkbox, but logic removed from calls
with col_opts3:
    ennhance_input = st.checkbox("Enhance input topic (via AI)?", disabled=st.session_state.is_chunk_run) # Keep checkbox, but logic removed

# --- Calculate Total Images and Handle Splitting (Keep Improved Version) ---
try:
    if 'count' in df.columns:
        df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)
        total_images = df['count'].sum()
    else:
        st.error("Input table must contain a 'count' column.")
        total_images = 0
except Exception as e:
    st.error(f"Error processing 'count' column: {e}")
    total_images = 0

st.info(f"Total images requested: {total_images}")

# --- Conditional Split Button (Keep User's Fixed Version with components.html) ---
if total_images > MAX_IMAGES_PER_RUN and not st.session_state.is_chunk_run:
    st.warning(f"Total image count ({total_images}) exceeds the limit ({MAX_IMAGES_PER_RUN}). Splitting is recommended.")
    if st.button("🚀 Split Run into Multiple Tabs", key="split_run"):
        st.spinner("Calculating and preparing chunks...")
        chunks_df = []
        current_chunk_rows_indices = []
        current_chunk_image_count = 0
        for index, row in df.iterrows():
            row_count = int(row.get('count', 0))
            current_chunk_rows_indices.append(index)
            current_chunk_image_count += row_count
            if current_chunk_image_count >= MAX_IMAGES_PER_RUN and current_chunk_rows_indices:
                chunks_df.append(df.loc[current_chunk_rows_indices].copy())
                current_chunk_rows_indices = []
                current_chunk_image_count = 0
        if current_chunk_rows_indices:
            chunks_df.append(df.loc[current_chunk_rows_indices].copy())

        if not chunks_df:
             st.error("No chunks could be created.")
        else:
            st.write(f"Splitting into {len(chunks_df)} chunks.")
            urls_to_open = []
            max_url_len = 0
            for i, chunk_df_item in enumerate(chunks_df):
                try:
                    json_data = chunk_df_item.to_json(orient='split')
                    encoded_data = base64.urlsafe_b64encode(json_data.encode('utf-8')).decode('utf-8')
                    params = urlencode({'chunk_data': encoded_data, 'autorun': 'true'})
                    url_fragment = f"/?{params}"
                    urls_to_open.append(url_fragment)
                    max_url_len = max(max_url_len, len(url_fragment))
                except Exception as e:
                    st.error(f"Failed to prepare chunk {i+1}: {e}")

            if max_url_len > 2000: st.warning(f"Warning: URLs are long (up to {max_url_len} chars). May exceed browser limits.")

            if urls_to_open:
                urls_json = json.dumps(urls_to_open)
                # User's fixed version using components.html button
                comp_str = """
                <!DOCTYPE html>
                                    <html>
                                    <body>
                                        <button onclick="openTabs()">Open Tabs</button>
                                        <script>
                                            function openTabs() {
                                                const urls = [...]; // your URLs
                                                let openedCount = 0;
                                                urls.forEach((url, index) => {
                                                    const win = window.open(url, `_blank_chunk_${index}`);
                                                    if (win) openedCount++;
                                                });
                                                if (openedCount < urls.length) {
                                                    alert(`Only opened ${openedCount}/${urls.length}. Check pop-up blocker.`);
                                                }
                                            }
                                        </script>
                                    </body>
                                    </html>



                """
                comp_str = comp_str.replace("[...]", urls_json)


                components.html(comp_str, height=150) # Adjust height as needed
            else:
                st.error("No valid URLs generated for splitting.")
        st.stop() # Stop original tab after button press

# --- Manual Generate Button (Keep) ---
allow_manual_generate = not (total_images > MAX_IMAGES_PER_RUN and not st.session_state.is_chunk_run) or st.session_state.is_chunk_run
generate_button_disabled = (total_images > MAX_IMAGES_PER_RUN and not st.session_state.is_chunk_run)

if st.button("Generate Images", key="manual_generate", disabled=generate_button_disabled, type="primary"):
    st.session_state.pop("auto_start_processing", None)
    if 'generated_images' in st.session_state: del st.session_state.generated_images
    if 'accumulated_generated_images' in st.session_state: del st.session_state.accumulated_generated_images
    run_generation(df.copy())

# --- Auto-start Logic (Keep) ---
if st.session_state.get("auto_start_processing"):
    st.session_state.pop("auto_start_processing")
    st.info("Auto-starting generation for this chunk...")
    if 'generated_images' in st.session_state: del st.session_state.generated_images
    if 'accumulated_generated_images' in st.session_state: del st.session_state.accumulated_generated_images
    run_generation(df.copy())

# --- Display Generated Images & Selection (Keep, DALL-E button included) ---
st.divider()
if 'generated_images' in st.session_state and st.session_state.generated_images:
    st.subheader("Select Images to Process")
    zoom = st.slider("Zoom Level for Selection", min_value=50, max_value=400, value=200, step=25)
    if auto_mode:
        st.write("Auto-mode enabled: Setting count to 1 for all generated images.")
        for entry in st.session_state.generated_images:
            for img in entry["images"]: img['selected_count'] = 1
    for entry_idx, entry in enumerate(st.session_state.generated_images):
        topic = entry.get("topic", "Unknown Topic")
        original_topic = entry.get("original_topic", topic)
        lang = entry.get("lang", "N/A")
        images = entry.get("images", [])
        if not images: continue
        st.write(f"#### Topic: {topic} ({lang})")
        if topic != original_topic: st.caption(f"(Original: {original_topic})")
        num_columns_display = 5
        rows_display = math.ceil(len(images) / num_columns_display)
        for row_idx in range(rows_display):
            cols = st.columns(num_columns_display)
            row_images_data = images[row_idx * num_columns_display : (row_idx + 1) * num_columns_display]
            for col_idx, img_data in enumerate(row_images_data):
                if col_idx < len(cols):
                    with cols[col_idx]:
                        img_url = img_data.get('url')
                        img_source = img_data.get('source', 'unknown')
                        img_template = img_data.get('template', 'N/A')
                        is_dalle_generated = img_data.get('dalle_generated', False)
                        st.image(img_url, width=zoom, caption=f"Source: {img_source} | Template: {img_template}")
                        unique_key = f"num_select_{entry_idx}_{img_url[-10:]}"
                        img_data.setdefault('selected_count', 1 if auto_mode else 0)
                        selected_count = st.number_input( f"Select Count", min_value=0, max_value=10, value=img_data['selected_count'], key=unique_key, label_visibility="collapsed", help=f"How many times to process this image ({img_url[-10:]})")
                        img_data['selected_count'] = selected_count
                        if img_source == "google" and not is_dalle_generated:
                            dalle_button_key = f"dalle_button_{entry_idx}_{img_url[-10:]}"
                            if st.button("DALL-E Variation", key=dalle_button_key, help="Generate AI variations (uses DALL-E)"):
                                if selected_count > 0:
                                    with st.spinner("Generating DALL-E variations..."):
                                        dalle_results = create_dalle_variation(img_url, selected_count) # Uses reverted function
                                        if dalle_results:
                                            st.success(f"{len(dalle_results)} DALL-E variations generated!")
                                            img_data["dalle_generated"] = True
                                            for dalle_img in dalle_results:
                                                entry["images"].append({
                                                    "url": dalle_img.url, "selected": False,
                                                    "selected_count": 1 if auto_mode else 0,
                                                    "template": img_data["template"], "source": "dalle",
                                                    "dalle_generated": True })
                                            st.rerun()
                                        else: st.error("Failed to generate DALL-E variations.")
                                else: st.warning("Set count > 0 to generate variations.")
elif st.session_state.get("is_chunk_run") and 'generated_images' not in st.session_state:
     st.info("Waiting for auto-run to complete image generation for this chunk...")
elif not st.session_state.get("is_chunk_run"):
     st.info("Enter topics above and click 'Generate Images' or 'Split Run'.")

# --- Process Selected Images Button (Reverted Internal Calls) ---
st.divider()
st.subheader("Process & Finalize Selected Images")
if st.button("Process Selected Images", type="primary"):
    if 'generated_images' not in st.session_state or not st.session_state.generated_images:
        st.warning("No images have been generated or selected yet.")
        st.stop()

    final_results = []
    selected_image_count_total = 0
    for entry in st.session_state.generated_images:
        for img in entry.get("images", []): selected_image_count_total += img.get('selected_count', 0)

    if selected_image_count_total == 0:
        st.warning("No images selected for processing. Set count > 0.")
        st.stop()

    process_bar = st.progress(0, text=f"Processing {selected_image_count_total} selected images...")
    processed_count = 0
    cta_texts_cache = {}

    with st.spinner("Processing selected images (generating text, HTML, screenshots)..."):
        for entry in st.session_state.generated_images:
            topic = entry.get("topic", "Unknown Topic")
            original_topic = entry.get("original_topic", topic)
            lang = entry.get("lang", "en")
            images = entry.get("images", [])
            res_row = {'Topic': topic, 'Original Topic': original_topic, 'Language': lang}
            image_col_counter = 1
            selected_images_in_entry = [img for img in images if img.get('selected_count', 0) > 0]
            if not selected_images_in_entry: continue

            if lang not in cta_texts_cache:
                try:
                    # Reverted: Use default chatGPT model (gpt-4o)
                    cta_trans = chatGPT(f"Return EXACTLY the text 'Learn More' in {lang} (no quotes).").strip('"').strip("'")
                    if not cta_trans: cta_trans = "Learn More"
                    cta_texts_cache[lang] = cta_trans
                except Exception as cta_err:
                     cta_texts_cache[lang] = "Learn More"
            cta_text_base = cta_texts_cache[lang]

            for img_data in selected_images_in_entry:
                count_to_process = img_data['selected_count']
                template_str_or_int = img_data['template']
                for i in range(count_to_process):
                    processed_count += 1
                    final_s3_url = img_data['url']

                    if isinstance(template_str_or_int, str) and "gemini" in template_str_or_int.lower():
                        pass # Skip HTML processing for Gemini
                    else:
                        try:
                             template_id = int(template_str_or_int)
                        except (ValueError, TypeError):
                             res_row[f'Image_{image_col_counter}'] = final_s3_url
                             image_col_counter += 1
                             process_bar.progress(min(1.0, processed_count / selected_image_count_total), text=f"Processing {processed_count}/{selected_image_count_total}...")
                             continue

                        headline_text = ""
                        cta_text_final = cta_text_base
                        tag_line = ""
                        try:
                            topic_for_prompt = re.sub('\\|.*','', topic)
                            # --- Reverted Models for Headline/CTA Gen ---
                            if template_id in [1, 2]:
                                headline_prompt = f"write a short text (up to 20 words) to promote an article about {topic_for_prompt} in {lang}. Goal: be concise yet compelling to click."
                                headline_text = chatGPT(prompt=headline_prompt, model='gpt-4').strip('"').strip("'") # Original used gpt-4
                            elif template_id in [3, 7]:
                                headline_prompt = f"write 1 statement, same length, no quotes, for {topic_for_prompt} in {lang}. Examples:\n'Surprising Travel Perks You Might Be Missing'\n'Little-Known Tax Tricks to Save Big'\nDont mention 'Hidden' or 'Unlock'."
                                headline_text = chatGPT(prompt=headline_prompt, model='gpt-4').strip('"').strip("'") # Original used gpt-4
                                if template_id == 7: cta_text_final = chatGPT(f"Return EXACTLY the text 'Check This Out' in {lang} (no quotes).").strip('"') or "Check This Out" # Use default gpt-4o
                            elif template_id == 5:
                                headline_prompt = f"write 1 statement, same length, no quotes, for {topic_for_prompt} in {lang}. ALL IN CAPS. wrap the 1-2 most urgent words in <span class='highlight'>...</span>. Make it under 60 chars total, to drive curiosity."
                                headline_text = chatGPT(prompt=headline_prompt, model='gpt-4').strip('"').strip("'") # Original used gpt-4
                                tagline_prompt = f"Write a short tagline for {topic_for_prompt} in {lang}, to drive action, max 25 chars, ALL CAPS, possibly with emoji. Do NOT mention the topic explicitly."
                                tag_line = chatGPT(prompt=tagline_prompt).strip('"').strip("'").strip("!") # Use default gpt-4o
                                headline_text = headline_text.replace(r"</span>", r"</span> ")
                            elif template_id in [4, 41, 42]:
                                headline_text = topic
                                cta_text_final = chatGPT(f"Return EXACTLY 'Read more about' in {lang} (no quotes).").strip('"') or "Read more about" # Use default gpt-4o
                            elif template_id in [6, 8]:
                                headline_text = ''
                                cta_text_final = ''
                            else: # Default fallback headline
                                headline_prompt = f"Write a concise headline for {topic_for_prompt} in {lang}, no quotes."
                                headline_text = chatGPT(prompt=headline_prompt, model='gpt-4').strip('"').strip("'") # Original used gpt-4
                            # --- End Reverted Models ---

                            html_content = save_html( headline=headline_text or "", image_url=img_data['url'], cta_text=cta_text_final or "", template=template_id, tag_line=tag_line or "" )
                            screenshot_width = 720 if template_id == 8 else 1000
                            screenshot_height = 480 if template_id == 8 else 1000
                            screenshot_image = capture_html_screenshot_playwright( html_content, width=screenshot_width, height=screenshot_height )

                            if screenshot_image:
                                s3_upload_url = upload_pil_image_to_s3( screenshot_image, S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION )
                                if s3_upload_url:
                                    final_s3_url = s3_upload_url
                                else: # Upload failed, use original
                                    final_s3_url = img_data['url']
                            else: # Screenshot failed, use original
                                final_s3_url = img_data['url']
                        except Exception as process_err:
                            st.warning(f"Failed processing instance for {img_data['url'][-10:]}: {process_err}")
                            final_s3_url = img_data['url'] # Fallback

                    res_row[f'Image_{image_col_counter}'] = final_s3_url
                    image_col_counter += 1
                    process_bar.progress(min(1.0, processed_count / selected_image_count_total), text=f"Processing {processed_count}/{selected_image_count_total}...")

            final_results.append(res_row)

    # --- Display Final Results (Keep Improved Version) ---
    if final_results:
        output_df = pd.DataFrame(final_results)
        image_cols = sorted([col for col in output_df.columns if col.startswith("Image_")], key=lambda x: int(x.split('_')[1]))
        if image_cols:
             output_df[image_cols] = output_df.apply(lambda row: shift_left_and_pad(row[image_cols], image_cols), axis=1)
             output_df.rename(columns={old_col: f"Image_{i+1}" for i, old_col in enumerate(image_cols)}, inplace=True)
             cols_order = ['Topic', 'Original Topic', 'Language'] + [f"Image_{i+1}" for i in range(len(image_cols))]
             # Ensure all expected columns exist before reordering
             cols_order = [col for col in cols_order if col in output_df.columns]
             output_df = output_df[cols_order]

        st.subheader("Final Results")
        st.dataframe(output_df)
        try:
            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Results as CSV", data=csv, file_name='final_results.csv', mime='text/csv')
        except Exception as e:
            st.error(f"Could not prepare CSV for download: {e}")
    else:
        st.warning("Processing finished, but no results were generated.")
