import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
import boto3
from botocore.exceptions import NoCredentialsError
import random
import string
import httpx # Use httpx for async requests
import google.generativeai as genai # Use the official library which supports async
from google.generativeai.types import GenerationConfig # For generation config
import anthropic # Anthropic library supports async
import json
import base64
import os
import time
import asyncio # Import asyncio
from playwright.async_api import async_playwright # Use async playwright
from tempfile import NamedTemporaryFile
import re
import math
from google_images_search import GoogleImagesSearch # This library is synchronous
import openai # For DALL-E variations (Needs async version)
from openai import AsyncOpenAI # Use AsyncOpenAI
import logging
import aiobotocore.session # Use aiobotocore for async S3
from functools import wraps


# Configure logging (optional, uncomment if needed)
# logging.basicConfig(
#     format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
#     level=logging.DEBUG
# )
# logger = logging.getLogger(__name__)

# --- Load Secrets ---
# It's generally better to load secrets once at the top
try:
    AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
    S3_BUCKET_NAME = st.secrets["S3_BUCKET_NAME"]
    AWS_REGION = st.secrets.get("AWS_REGION", "us-east-1")
    GPT_API_KEY = st.secrets["GPT_API_KEY"]
    FLUX_API_KEY = st.secrets["FLUX_API_KEY"] # Assuming this is a list or tuple
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] # Assuming this is a list or tuple
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] # Assuming this is a list or tuple
    GOOGLE_CX = st.secrets["GOOGLE_CX"]

    # --- Initialize Async Clients ---
    # Initialize clients once to reuse connections
    async_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    async_anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    # No persistent client needed for genai library, it's handled per call
    # No persistent client needed for aiobotocore, typically created with async with
    # No persistent client needed for httpx, typically created with async with

except KeyError as e:
    st.error(f"Missing secret: {e}. Please check your Streamlit secrets.")
    st.stop()
except Exception as e:
    st.error(f"Error loading secrets or initializing clients: {e}")
    st.stop()


# --- Utility Functions ---

def shift_left_and_pad(row):
    """
    Utility function to left-shift non-null values in each row
    and pad with empty strings, preserving columns' order.
    """
    valid_values = [x for x in row if pd.notna(x)]
    # Need image_cols defined globally or passed in. Define later.
    padded_values = valid_values + [''] * (len(st.session_state.get('image_cols', [])) - len(valid_values))
    return pd.Series(padded_values[:len(st.session_state.get('image_cols', []))])

def log_function_call(func):
    """
    Decorator to log function calls and return values (works for async too).
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # logger.info(f"CALL: {func.__name__} - args: {args}, kwargs: {kwargs}")
        print(f"CALL: {func.__name__}") # Simplified logging for example
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            # Allow decorating sync functions if needed, run in thread
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, func, *args, **kwargs)
        # logger.info(f"RETURN: {func.__name__} -> {result}")
        print(f"RETURN: {func.__name__} -> Type: {type(result)}") # Simplified
        return result
    return wrapper


# --- Asynchronous API/IO Functions ---

# @log_function_call # Decorator might add overhead, use if debugging needed
async def fetch_google_images_async(query, num_images=3, max_retries=5):
    """
    Fetch images from Google Images using google_images_search asynchronously.
    Uses asyncio.to_thread to run the synchronous library code without blocking.
    """
    terms_list = query.split('~')
    all_image_urls = []
    loop = asyncio.get_running_loop()

    gis_instances = {} # Cache instances per API key if needed, though random choice might negate benefit

    async def search_term(term):
        for trial in range(max_retries):
            try:
                # Select API key randomly for each attempt or term
                api_key = random.choice(GOOGLE_API_KEY)
                cx = GOOGLE_CX

                # Create GIS instance (synchronous part)
                # Run synchronous GIS setup and search in a separate thread
                def sync_search():
                    # Consider caching GIS instance based on api_key if retries are frequent
                    # if api_key not in gis_instances:
                    #     gis_instances[api_key] = GoogleImagesSearch(api_key, cx)
                    # gis = gis_instances[api_key]
                    gis = GoogleImagesSearch(api_key, cx) # Create fresh instance

                    search_params = {
                        'q': term.strip(),
                        'num': num_images,
                        # Add other params like safe search if needed: 'safe': 'medium'
                    }
                    gis.search(search_params=search_params)
                    return [result.url for result in gis.results()]

                # Execute the synchronous search function in the executor
                term_urls = await loop.run_in_executor(None, sync_search)
                return term_urls # Return successfully found URLs for this term
            except Exception as e:
                st.warning(f"Error fetching Google Images for '{term}' (Attempt {trial+1}/{max_retries}): {e}")
                if trial < max_retries - 1:
                    await asyncio.sleep(random.uniform(3, 7)) # Exponential backoff might be better
                else:
                    st.error(f"Failed to fetch Google Images for '{term}' after {max_retries} attempts.")
                    return [] # Return empty list on failure for this term
        return [] # Should not be reached if loop completes, but safety return

    # Gather results for all terms concurrently
    results_per_term = await asyncio.gather(*(search_term(term) for term in terms_list))

    # Flatten the list of lists and remove duplicates
    for url_list in results_per_term:
        all_image_urls.extend(url_list)

    return list(set(all_image_urls))


def play_sound(audio_file_path):
    """Plays an audio file in the Streamlit app. (Remains Synchronous)"""
    try:
        with open(audio_file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            Your browser does not support the audio element.
            </audio>
        """
        st.markdown(md, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Audio file not found: {audio_file_path}")
    except Exception as e:
        st.error(f"Error playing sound: {e}")


# @log_function_call
def install_playwright_browsers():
    """
    Install Playwright browsers (Chromium) if not installed yet. (Remains Synchronous)
    Best run once at startup.
    """
    st.info("Checking and installing Playwright browser dependencies...")
    try:
        # Use subprocess for better error handling if needed
        result_deps = os.system('playwright install-deps')
        if result_deps != 0:
             st.warning("Playwright install-deps might have had issues (non-zero exit code).")
        result_install = os.system('playwright install chromium')
        if result_install != 0:
            raise OSError("Playwright install chromium failed.")
        st.success("Playwright browsers should be installed.")
        return True
    except Exception as e:
        st.error(f"Failed to install Playwright browsers: {str(e)}")
        return False

# Initialize playwright state
if 'playwright_installed' not in st.session_state:
    st.session_state.playwright_installed = install_playwright_browsers()


# @log_function_call
async def upload_pil_image_to_s3_async(
    image,
    bucket_name,
    aws_access_key_id,
    aws_secret_access_key,
    object_name='',
    region_name='us-east-1',
    image_format='PNG'
):
    """
    Upload a PIL image to S3 asynchronously using aiobotocore.
    """
    session = aiobotocore.session.get_session()
    # Use async with to ensure the client is closed properly
    async with session.create_client(
        's3',
        region_name=region_name.strip(),
        aws_access_key_id=aws_access_key_id.strip(),
        aws_secret_access_key=aws_secret_access_key.strip()
    ) as s3_client:
        try:
            if not object_name:
                object_name = f"image_{int(time.time())}_{random.randint(1000, 9999)}.{image_format.lower()}"

            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format=image_format)
            img_byte_arr.seek(0)

            await s3_client.put_object(
                Bucket=bucket_name.strip(),
                Key=object_name,
                Body=img_byte_arr,
                ContentType=f'image/{image_format.lower()}'
            )

            # Construct URL (consider using get_presigned_url if needed, but public URL is often simpler)
            # Ensure your bucket policy allows public reads if using this format
            url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}"
            return url

        except NoCredentialsError:
            st.error("S3 Upload Error: AWS credentials not found.")
            return None
        except Exception as e:
            st.error(f"Error in S3 upload async: {str(e)}")
            return None

# @log_function_call
async def gemini_text_lib_async(prompt, model='gemini-1.5-flash', is_pd_policy=False, predict_policy=""):
    """ Generate text using Google GenAI async library """
    if is_pd_policy:
        prompt += predict_policy

    api_key = random.choice(GEMINI_API_KEY)
    genai.configure(api_key=api_key)

    try:
        # Choose an appropriate async model if available, or use the specified one
        async_model = genai.GenerativeModel(model)
        response = await async_model.generate_content_async(
            prompt,
            generation_config=GenerationConfig(
                # response_mime_type="text/plain", # Often default
                temperature=0.7 # Example config
            )
        )
        # Accessing response parts safely
        if response.candidates and response.candidates[0].content.parts:
             return response.candidates[0].content.parts[0].text.replace('```', '')
        else:
             # Log or handle cases with no response or unexpected structure
             st.warning(f"Gemini response structure unexpected or empty for prompt: {prompt[:100]}...")
             # Attempt to access text directly if structure is simpler
             try:
                 return response.text.replace('```', '')
             except AttributeError:
                 st.error("Could not extract text from Gemini response.")
                 return None

    except Exception as e:
        st.error(f'gemini_text_lib_async error: {e}')
        await asyncio.sleep(random.uniform(3, 5)) # Sleep before potential retry
        return None # Indicate failure

# @log_function_call
async def chatGPT_async(prompt, model="gpt-4o", temperature=1.0, reasoning_effort='', is_pd_policy=False, predict_policy=""):
    """ Generate text using OpenAI async library """
    if is_pd_policy:
        prompt += predict_policy

    try:
        messages = [{"role": "user", "content": prompt}]
        # Base parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        # Conditional parameters (Note: OpenAI API might not support 'reasoning' or 'input' directly like this for standard chat completions)
        # Adapt based on the actual API endpoint and parameters if using a non-standard one.
        # For standard chat completions, only include standard parameters.
        # if reasoning_effort: params['reasoning'] = {"effort": reasoning_effort} # This looks non-standard
        if temperature == 0: del params['temperature'] # Or set to a very low value like 0.01

        # Use the standard async chat completion endpoint
        response = await async_openai_client.chat.completions.create(**params)

        # Extract content safely
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            return content.strip()
        else:
            st.error("Invalid response structure from OpenAI.")
            return None

    except Exception as e:
        st.error(f"Error in chatGPT_async: {str(e)}")
        # Log the actual response if available and helpful
        # if 'response' in locals() and hasattr(response, 'text'):
        #     st.text(f"Response text: {response.text}")
        return None


# @log_function_call
async def claude_async(prompt, model="claude-3-haiku-20240307", temperature=1, is_thinking=False, max_retries=5, is_pd_policy=False, predict_policy=""):
    if is_pd_policy:
        prompt += predict_policy

    tries = 0
    while tries < max_retries:
        try:
            messages = [{"role": "user", "content": prompt}]

            # Base parameters
            params = {
                "model": model,
                "messages": messages,
                "max_tokens": 4000, # Reduced from 20000, adjust as needed
                "temperature": temperature,
                "top_p": 0.8 # Example, adjust as needed
            }

            # Conditional parameters for 'thinking' - Check Anthropic docs for current implementation
            # The original 'thinking' structure seems non-standard for the messages API.
            # If 'thinking' is a specific feature, consult docs. Assuming it's not standard here.
            # if is_thinking:
                # This structure might be incorrect for the standard API
                # params["thinking"] = {"type": "enabled", "budget_tokens": 16000}

            message = await async_anthropic_client.messages.create(**params)

            # Extract content safely
            if message.content and isinstance(message.content, list) and message.content[0].type == "text":
                # If is_thinking modified the structure, adjust index (e.g., [1] ?)
                # Assuming standard response structure here:
                return message.content[0].text.strip()
            else:
                 st.warning(f"Claude response structure unexpected or empty for prompt: {prompt[:100]}...")
                 # Maybe try accessing attributes differently if needed
                 try:
                      if isinstance(message.content, str): # Handle simpler string response if API changes
                           return message.content.strip()
                 except Exception:
                      pass # Ignore if alternative access fails
                 st.error("Could not extract text from Claude response.")
                 return None # Indicate failure


        except anthropic.APIConnectionError as e:
             st.warning(f"Claude Connection Error (Attempt {tries+1}/{max_retries}): {e}. Retrying...")
             tries += 1
             await asyncio.sleep(random.uniform(2**tries, 2**(tries+1))) # Exponential backoff
        except anthropic.APIStatusError as e:
             st.warning(f"Claude API Status Error (Attempt {tries+1}/{max_retries}): {e.status_code} - {e.response}. Retrying...")
             tries += 1
             await asyncio.sleep(random.uniform(2**tries, 2**(tries+1))) # Exponential backoff
        except anthropic.RateLimitError as e:
             st.warning(f"Claude Rate Limit Error (Attempt {tries+1}/{max_retries}): {e}. Retrying after delay...")
             tries += 1
             await asyncio.sleep(random.uniform(10, 20)) # Longer wait for rate limits
        except Exception as e:
            st.error(f"Claude Error (Attempt {tries+1}/{max_retries}): {e}")
            tries += 1
            if tries >= max_retries:
                 st.error(f"Failed to get response from Claude after {max_retries} attempts.")
                 return None # Indicate failure after all retries
            await asyncio.sleep(random.uniform(5, 10)) # General retry delay

    return None # Failed after retries


# @log_function_call
async def gen_flux_img_async(prompt, height=784, width=960, model="black-forest-labs/FLUX.1-schnell-Free"):
    """
    Generate images using FLUX model via Together.xyz API asynchronously.
    """
    url = "https://api.together.xyz/v1/images/generations"
    payload = {
        "prompt": prompt,
        "model": model,
        "steps": 4, # Adjust as needed
        "n": 1,
        "height": height,
        "width": width,
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {random.choice(FLUX_API_KEY)}"
    }
    max_retries = 5
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=90.0) as client: # Increased timeout for image gen
                 response = await client.post(url, json=payload, headers=headers)
                 response.raise_for_status() # Raise HTTPStatusError for bad responses (4xx or 5xx)
                 data = response.json()
                 if data.get("data") and len(data["data"]) > 0 and data["data"][0].get("url"):
                      return data["data"][0]["url"]
                 elif data.get("error"):
                      st.warning(f"Flux API returned error: {data['error']}")
                      # Check for specific errors like NSFW
                      if "NSFW" in str(data['error']):
                           st.warning(f"NSFW content detected by Flux API for prompt: {prompt[:100]}...")
                           return None # Indicate NSFW failure
                      # Other errors might be retriable
                 else:
                      st.warning(f"Unexpected response format from Flux API: {data}")

        except httpx.HTTPStatusError as e:
            st.warning(f"Flux API HTTP Error (Attempt {attempt+1}/{max_retries}): {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 429: # Rate limit
                await asyncio.sleep(random.uniform(5, 10) * (attempt + 1)) # Exponential backoff
            elif e.response.status_code >= 500: # Server error
                await asyncio.sleep(random.uniform(3, 7) * (attempt + 1))
            else: # Other client errors (4xx) - likely not retriable
                st.error(f"Flux API client error: {e.response.status_code}")
                return None # Non-retriable client error
        except httpx.RequestError as e:
            st.warning(f"Flux API Request Error (Attempt {attempt+1}/{max_retries}): {e}")
            await asyncio.sleep(random.uniform(3, 7) * (attempt + 1))
        except Exception as e:
            st.error(f"Unexpected error in gen_flux_img_async (Attempt {attempt+1}/{max_retries}): {e}")
            # Check if it's a known non-retriable error based on string content
            if "NSFW" in str(e): # Example check
                return None
            await asyncio.sleep(random.uniform(3, 7) * (attempt + 1))

    st.error(f"Failed to generate Flux image for prompt '{prompt[:100]}...' after {max_retries} attempts.")
    return None # Failed after all retries


# @log_function_call
async def gen_gemini_image_async(prompt, max_retries=5):
    """ Generate image using Gemini Image API async """
    api_key = random.choice(GEMINI_API_KEY)
    # Correct endpoint might vary, check latest Gemini docs. This looks plausible.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent?key={api_key}" # Using pro-vision as example, check docs for specific image gen model endpoint
    # url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}" # Or maybe flash directly?

    headers = {"Content-Type": "application/json"}

    # Construct the payload carefully based on API docs
    # The original payload mixes user roles and has an empty part, which seems wrong.
    # A more typical structure for multimodal might be:
    data = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}] # Send the prompt as text
        }],
        "generationConfig": {
             "temperature": 0.65,
             "topK": 40,
             "topP": 0.95,
             "maxOutputTokens": 1024, # Adjust token limits as needed
            # "responseMimeType": "application/json", # Usually default
            # Requesting an image modality might need specific parameters
            # "output_modality": "image/png" # Hypothetical, check docs
        },
        # Add safety settings if needed
        # "safetySettings": [...]
    }

    # Simplified payload for potential image model (check exact API)
    # data_img_gen = {
    #     "prompt": {"text": prompt},
    #     "n": 1,
    #     "size": "1024x1024", # Example
    #     "response_format": "b64_json"
    # }


    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client: # Longer timeout for image generation
                response = await client.post(url, headers=headers, json=data) # Using multimodal payload structure
                response.raise_for_status()
                res_json = response.json()

                # --- Safely extract image data ---
                # The path might differ based on the actual API response structure
                # Path 1: From original code (might be correct for some versions)
                try:
                    image_b64 = res_json['candidates'][0]["content"]["parts"][0]["inlineData"]['data']
                    image_data = base64.b64decode(image_b64) # Use b64decode
                    return Image.open(BytesIO(image_data))
                except (KeyError, IndexError, TypeError) as e1:
                    # Path 2: Alternative structure (e.g., direct image data)
                    try:
                        # Check if 'image_data' or similar key exists directly
                        if 'image_bytes' in res_json: # Hypothetical key
                             image_data = base64.b64decode(res_json['image_bytes'])
                             return Image.open(BytesIO(image_data))
                        # Add other potential paths based on Gemini API docs
                    except (KeyError, IndexError, TypeError) as e2:
                        st.warning(f"Could not extract Gemini image data (Attempt {attempt+1}). Structure: {res_json}. Errors: {e1}, {e2}")
                        # Fall through to retry logic

        except httpx.HTTPStatusError as e:
            st.warning(f"Gemini Image API HTTP Error (Attempt {attempt+1}/{max_retries}): {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 429: # Rate limit
                await asyncio.sleep(random.uniform(5, 10) * (attempt + 1))
            elif e.response.status_code >= 500: # Server error
                await asyncio.sleep(random.uniform(3, 7) * (attempt + 1))
            elif "API key not valid" in e.response.text:
                 st.error(f"Invalid Gemini API Key used. Please check secrets.")
                 return None # Non-retriable
            else: # Other client errors (4xx)
                st.error(f"Gemini Image API client error: {e.response.status_code}. Response: {e.response.text}")
                return None # Likely non-retriable
        except httpx.RequestError as e:
            st.warning(f"Gemini Image API Request Error (Attempt {attempt+1}/{max_retries}): {e}")
            await asyncio.sleep(random.uniform(3, 7) * (attempt + 1))
        except Exception as e:
            st.error(f"Unexpected error in gen_gemini_image_async (Attempt {attempt+1}/{max_retries}): {e}")
            await asyncio.sleep(random.uniform(3, 7) * (attempt + 1))

        if attempt == max_retries - 1:
            st.error(f"Failed to generate Gemini image for prompt '{prompt[:100]}...' after {max_retries} attempts.")

    return None # Failed


# @log_function_call
async def gen_flux_img_lora_async(prompt, height=784, width=960, lora_path="https://huggingface.co/ddh0/FLUX-Amateur-Photography-LoRA/resolve/main/FLUX-Amateur-Photography-LoRA-v2.safetensors?download=true", lora_scale=0.99):
    """ Generate image using Flux with LoRA via Together.xyz API async """
    url = "https://api.together.xyz/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {random.choice(FLUX_API_KEY)}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "black-forest-labs/FLUX.1-dev-lora", # Ensure this model ID is correct
        "prompt": "candid unstaged taken with iphone 8 : " + prompt, # Prepending style
        "width": width,
        "height": height,
        "steps": 20, # Adjust as needed
        "n": 1,
        "response_format": "url",
        "image_loras": [
            {"path": lora_path, "scale": lora_scale}
        ],
        # "update_at": "2025-03-04T16:25:21.474Z" # update_at seems unnecessary for generation
    }
    max_retries = 5
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client: # Longer timeout
                 response = await client.post(url, headers=headers, json=data)
                 response.raise_for_status()
                 response_data = response.json()
                 if response_data.get("data") and len(response_data["data"]) > 0 and response_data["data"][0].get("url"):
                      image_url = response_data['data'][0]['url']
                      # print(f"Flux LoRA Image URL: {image_url}") # Debug
                      return image_url
                 else:
                      st.warning(f"Unexpected response format from Flux LoRA API: {response_data}")

        except httpx.HTTPStatusError as e:
            st.warning(f"Flux LoRA API HTTP Error (Attempt {attempt+1}/{max_retries}): {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 429:
                await asyncio.sleep(random.uniform(5, 10) * (attempt + 1))
            elif e.response.status_code >= 500:
                await asyncio.sleep(random.uniform(3, 7) * (attempt + 1))
            else:
                st.error(f"Flux LoRA API client error: {e.response.status_code}")
                return None
        except httpx.RequestError as e:
            st.warning(f"Flux LoRA API Request Error (Attempt {attempt+1}/{max_retries}): {e}")
            await asyncio.sleep(random.uniform(3, 7) * (attempt + 1))
        except Exception as e:
            st.error(f"Unexpected error in gen_flux_img_lora_async (Attempt {attempt+1}/{max_retries}): {e}")
            await asyncio.sleep(random.uniform(3, 7) * (attempt + 1))

    st.error(f"Failed to generate Flux LoRA image after {max_retries} attempts.")
    return None


# @log_function_call
async def capture_html_screenshot_playwright_async(html_content, width=1000, height=1000):
    """
    Use async Playwright to capture a screenshot of the given HTML snippet.
    """
    if not st.session_state.get('playwright_installed', False):
        st.error("Playwright browsers not installed properly")
        return None

    screenshot_bytes = None
    temp_html_path = None

    try:
        # async with playwright context manager
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu'] # Added disable-gpu
            )
            page = await browser.new_page(viewport={'width': width, 'height': height})

            # Use NamedTemporaryFile correctly (synchronous part, but acceptable)
            # Needs to be created outside the async with block if cleaned up later
            with NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
                 f.write(html_content)
                 temp_html_path = f.name # Get the path

            # Go to the local file URL
            await page.goto(f'file://{temp_html_path}')
            # Wait for potential JS or rendering to settle
            await page.wait_for_timeout(1500) # Slightly longer wait

            # Take screenshot
            screenshot_bytes = await page.screenshot(type='png') # Specify type

            # Close browser
            await browser.close()

        # Clean up the temporary file *after* browser is closed
        if temp_html_path and os.path.exists(temp_html_path):
            os.unlink(temp_html_path)

        if screenshot_bytes:
             return Image.open(BytesIO(screenshot_bytes))
        else:
             st.error("Screenshot capture failed (no bytes returned).")
             return None

    except Exception as e:
        st.error(f"Async screenshot capture error: {str(e)}")
        # Ensure cleanup happens even on error
        if temp_html_path and os.path.exists(temp_html_path):
            try:
                os.unlink(temp_html_path)
            except Exception as cleanup_e:
                st.warning(f"Could not delete temp file {temp_html_path}: {cleanup_e}")
        return None


# @log_function_call # This function is CPU-bound (HTML string formatting), not IO-bound. Async doesn't help here.
def save_html(headline, image_url, cta_text, template, tag_line='', output_file="advertisement.html"):
    """
    Returns an HTML string based on the chosen template ID. (Remains Synchronous)
    """
    # Basic input validation
    if not isinstance(template, int):
         try:
              template = int(template)
         except (ValueError, TypeError):
              st.error(f"Invalid template format: {template}. Must be an integer.")
              return f"<p>Invalid Template ID: {template}</p>" # Return error HTML

    # Template definitions (Keep existing HTML, ensure template IDs match usage)
    # Template 1
    if template == 1:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template 1</title>
            <style>
                body {{
                    font-family: 'Gisha', sans-serif; /* Ensure font is available or use fallback */
                    font-weight: 550; margin: 0; padding: 0; display: flex;
                    justify-content: center; align-items: center; height: 100vh;
                    background-color: #f0f0f0; /* Added background for context */
                }}
                .ad-container {{
                    width: 1000px; height: 1000px; border: 1px solid #ddd;
                    border-radius: 20px; box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
                    display: flex; flex-direction: column; justify-content: space-between;
                    align-items: center; padding: 30px;
                    background: url('{image_url}') no-repeat center center; /* Ensure URL is valid */
                    background-size: contain; /* Changed from cover for potentially better text visibility */
                    text-align: center; position: relative; /* Added relative for potential overlay */
                    overflow: hidden; /* Prevents content spillover */
                }}
                 /* Added overlay for better text readability on diverse images */
                .text-overlay {{
                    background-color: rgba(255, 255, 255, 0.85); /* Semi-transparent white */
                    padding: 20px 40px; border-radius: 15px; margin-top: 20px; /* Adjusted margin */
                    display: inline-block; /* Fit content width */
                    max-width: 90%; /* Prevent excessive width */
                 }}
                .ad-title {{
                    font-size: 3.2em; color: #333; margin: 0; /* Removed margin-top from here */
                }}
                .cta-button {{
                    font-weight: 400; display: inline-block; padding: 30px 50px; /* Adjusted padding */
                    font-size: 2.8em; /* Adjusted size */ color: white; background-color: #FF5722;
                    border: none; border-radius: 20px; text-decoration: none;
                    cursor: pointer; transition: background-color 0.3s ease;
                    margin-bottom: 40px; /* Adjusted margin */ box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                }}
                .cta-button:hover {{ background-color: #E64A19; }}
            </style>
        </head>
        <body>
            <div class="ad-container">
                <div class="text-overlay">
                     <div class="ad-title">{headline}</div>
                 </div>
                <a href="#" class="cta-button">{cta_text}</a>
            </div>
        </body>
        </html>
        """
    # Template 2
    elif template == 2:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8"> <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template 2</title>
            <style>
                body {{ font-family: 'Gisha', sans-serif; font-weight: 550; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #f0f0f0; }}
                .ad-container {{ width: 1000px; height: 1000px; border: 2px solid black; box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2); display: flex; flex-direction: column; overflow: hidden; position: relative; background-color: white; }}
                .ad-title {{ font-size: 3em; /* Adjusted size */ color: #333; background-color: white; padding: 25px; /* Adjusted padding */ text-align: center; flex: 0 0 auto; /* Don't grow/shrink, use content height */ border-bottom: 1px solid #eee; /* Separator */ }}
                .ad-image {{ flex: 1 1 auto; /* Grow and shrink */ background: url('{image_url}') no-repeat center center; background-size: cover; /* Changed to cover */ position: relative; }}
                .cta-button {{ font-weight: 400; display: inline-block; padding: 25px 45px; /* Adjusted */ font-size: 3em; /* Adjusted */ color: white; background-color: #FF5722; border: none; border-radius: 20px; text-decoration: none; cursor: pointer; transition: background-color 0.3s ease; position: absolute; bottom: 8%; /* Adjusted position */ left: 50%; transform: translateX(-50%); z-index: 10; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }}
                .cta-button:hover {{ background-color: #E64A19; }}
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
    # Template 3 or 7 (Consolidated - Check if visual difference was intended for 7)
    elif template in [3, 7]:
        # If template 7 needs specific style changes, add conditional CSS or duplicate block
        button_class = "cta-button" if template == 3 else "c1ta-button" # Use different class for 7 if needed
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ad Template {template}</title>
            <link href="https://fonts.googleapis.com/css2?family=Boogaloo&display=swap" rel="stylesheet">
            <style>
                body {{ margin: 0; padding: 0; font-family: 'Boogaloo', 'Segoe UI Emoji', 'Apple Color Emoji', sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; background: #f0f0f0; }}
                .container {{ position: relative; width: 1000px; height: 1000px; margin: 0; padding: 0; overflow: hidden; box-shadow: 0 0 20px rgba(0,0,0,0.2); background-color: #ccc; /* BG if image fails */}}
                .image {{ width: 100%; height: 100%; object-fit: cover; filter: saturate(120%) contrast(105%); /* Adjusted filter */ transition: transform 0.3s ease; display: block; /* Remove potential bottom space */ }}
                .image:hover {{ transform: scale(1.03); /* Reduced hover effect */ }}
                .overlay {{ position: absolute; top: 0; left: 0; width: 100%; background: rgba(255, 0, 0, 0.85); /* Red overlay with alpha */ display: flex; justify-content: center; align-items: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); padding: 25px; /* Adjusted padding */ box-sizing: border-box; min-height: 12%; /* Adjusted */ }}
                .overlay-text {{ color: #FFFFFF; font-size: 3.8em; /* Adjusted size */ text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); /* Softer shadow */ letter-spacing: 1px; /* Adjusted */ margin: 0; word-wrap: break-word; }}
                .{button_class} {{ /* Dynamic class based on template */ position: absolute; bottom: 8%; /* Adjusted position */ left: 50%; transform: translateX(-50%); padding: 25px 50px; /* Adjusted padding */ background: blue; color: white; border: none; border-radius: 50px; font-size: 3.2em; /* Adjusted */ cursor: pointer; transition: all 0.3s ease; font-family: 'Boogaloo', sans-serif; text-transform: uppercase; letter-spacing: 1.5px; /* Adjusted */ box-shadow: 0 5px 15px rgba(0,0,255,0.4); /* Blue shadow */ }}
                .{button_class}:hover {{ background: #007bff; /* Lighter blue */ transform: translateX(-50%) translateY(-3px); box-shadow: 0 8px 20px rgba(0,123,255,0.6); }}
                /* Keyframes for shine effect (if used) */
                /* Example: Add a pseudo-element to the button if shine is desired */
            </style>
        </head>
        <body>
            <div class="container">
                <img src="{image_url}" class="image" alt="Ad Image for {headline}">
                <div class="overlay">
                    <h1 class="overlay-text">{headline}</h1>
                </div>
                <button class="{button_class}">{cta_text}</button>
            </div>
        </body>
        </html>
        """
        # Specific CSS override for template 7 if button class is different
        # if template == 7:
        #    html_template = html_template.replace(
        #         f'.{button_class} {{',
        #         f'.{button_class} {{ /* Add specific styles for c1ta-button here */'
        #    )

    # Template 4, 41, 42 (Consolidated - Adjust positioning)
    elif template in [4, 41, 42]:
        if template == 4: top_position = "50%"
        elif template == 41: top_position = "15%"
        elif template == 42: top_position = "85%" # Changed from 90% for better spacing
        else: top_position = "50%" # Default

        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8"> <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ad Template {template}</title>
            <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Bebas+Neue&display=swap" rel="stylesheet">
            <style>
                /* Define Calibre font if used, otherwise fallback */
                /* @font-face {{ font-family: 'Calibre'; src: url('path-to-calibre-font.woff2') format('woff2'); }} */
                body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #F4F4F4; }}
                .container {{ position: relative; width: 1000px; height: 1000px; background-image: url('{image_url}'); background-size: cover; background-position: center
