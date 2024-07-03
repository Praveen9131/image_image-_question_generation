import os
import base64
import asyncio
import aiohttp
import logging
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import openai
import requests

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Function to generate an image using DALL-E 3
def generate_image(prompt: str):
    logger.info(f"Generating image with prompt: {prompt}")
    response = openai.Image.create(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    logger.info(f"Image generated: {image_url}")
    return image_url

# Async function to download, resize, and save an image
async def download_resize_save_image(url: str, size: tuple, filename: str):
    logger.info(f"Downloading image from URL: {url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.read()
                image = Image.open(BytesIO(data))
                image = image.resize(size, Image.Resampling.LANCZOS)
                image.save(filename, format="JPEG", quality=10)  # Save resized image with reduced quality
                logger.info(f"Image saved: {filename}")
                return filename
    logger.error(f"Failed to download image from URL: {url}")
    return None

# Function to generate a detailed description of the image using a POST request
def generate_image_description(image_url: str):
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you think the person in the image is doing?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            }
        ],
        "max_tokens": 300
    }

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    logger.info(f"Generating description for image: {image_url}")
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)
    response_data = response.json()

    if response.status_code == 200:
        description = response_data["choices"][0]["message"]["content"]
        logger.info(f"Generated description: {description}")
        return description
    else:
        error_message = response_data.get("error", {}).get("message", "Unknown error")
        logger.error(f"Error generating description: {error_message}")
        return None

# Function to generate a question from a detailed description using GPT-4
def generate_mcq_from_description(description: str, tone: str, subject: str):
    prompt = [
        {"role": "system", "content": "You are an expert in generating educational content."},
        {"role": "user", "content": f"Based on the following detailed description of an image related to the topic '{subject}', generate a multiple-choice question in a {tone} tone with 4 options and provide the correct answer: {description}"}
    ]
    logger.info("Generating MCQ from description")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=prompt,
            max_tokens=1000,
            temperature=0.5
        )
        mcq = response.choices[0].message['content']
        logger.info(f"Generated MCQ: {mcq}")
        return mcq
    except Exception as e:
        logger.error(f"Error generating MCQ: {e}")
        return "Unable to generate a multiple-choice question at this time."

# Consolidated function to generate image and MCQ
async def generate_image_mcq(number: int, subject: str, tone: str):
    images_and_questions = []
    for i in range(number):
        # Generate image
        image_prompt = f"An illustration representing the topic: {subject}"
        image_url = generate_image(image_prompt)
        
        # Resize and save image
        image_filename = f"static/question_image_{i+1}.jpg"
        resized_image_path = await download_resize_save_image(image_url, (750, 319), image_filename)
        
        if resized_image_path:
            # Generate a detailed description of the image
            description = generate_image_description(image_url)
            
            if description:
                # Generate MCQ based on the detailed description
                mcq_text = generate_mcq_from_description(description, tone, subject)
            else:
                mcq_text = f"As there is no detailed description of the image related to the topic '{subject}', I'm unable to generate a multiple-choice question. Please provide the image description for me to assist you better."
            
            images_and_questions.append({
                'question_image_url': image_url,
                'question_image_url_resized': resized_image_path,
                'mcq': mcq_text
            })

    return images_and_questions

@app.route('/generate_content', methods=['GET'])
async def generate_content():
    number = request.args.get('number')
    subject = request.args.get('subject')
    tone = request.args.get('tone')

    if not number or not subject or not tone:
        logger.error("Missing required parameters: number, subject, and tone")
        return jsonify({"error": "Missing required parameters: number, subject, and tone"}), 400

    try:
        number = int(number)
    except ValueError:
        logger.error("Number must be an integer")
        return jsonify({"error": "number must be an integer"}), 400

    try:
        content = await generate_image_mcq(number, subject, tone)
        return jsonify(content)
    except Exception as e:
        logger.error(f"Internal server error: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
