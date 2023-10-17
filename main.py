import os
import openai
import warnings
from PIL import Image
from io import BytesIO
from typing import List
from dotenv import load_dotenv
from models import get_image_caption
from fastapi import FastAPI, File, UploadFile

# load env variables
load_dotenv();

# ignore warnings
warnings.filterwarnings("ignore")

# set openai keys
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_prompt(list: List[dict]):
    prompt = f"""
    Generate a creative brief for an advertisement campaign based on the following list of images, captions, and descriptions:

    Image List:
    {list}

    Your creative brief should include the following steps:

    1. Define the target audience for the campaign.
    2. Identify the core message or theme you want to convey.
    3. Describe the key features or benefits of the product or service.
    4. Suggest the tone and style of the advertisement (e.g., humorous, emotional, informative).
    5. Recommend the primary media channels for the campaign (e.g., TV, social media, print).
    6. Outline any specific visual or auditory elements that should be included (e.g., a catchy jingle, a memorable logo).
    7. Propose a call to action or next steps for the audience.
    8. Provide any additional notes or ideas that could enhance the campaign.

    Feel free to be creative and think outside the box in your brief. Be sure to incorporate the unique elements of each image and its accompanying captions and descriptions into the campaign idea.

    [You can start with "The creative brief for our campaign is as follows:"]
    """
    return prompt

# create an app server
app = FastAPI()

# define your endpoints
@app.post("/generate")
async def generate_brief(images: List[UploadFile] = File(...)):
    image_files = []
    for image in images:
        image_rgb = Image.open(BytesIO(image.file.read())).convert("RGB")
        image_caption = get_image_caption(image_rgb)
        image_files.append({"image": image.filename, "caption": image_caption})
    
    # prepare prompt
    prompt = get_prompt(image_files)

    # prepare creative brief
    completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    print(completion.choices[0].message.content)
    return {"data": {"images": image_files, "brief": completion.choices[0].message.content}}
