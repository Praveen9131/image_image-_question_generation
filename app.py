import os
import base64
import asyncio
import aiohttp
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import openai
import logging

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Directory to store images
image_dir = "images"
os.makedirs(image_dir, exist_ok=True)

# Function to generate an image using DALL-E 3
def generate_image(prompt: str):
    response = openai.Image.create(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    return response['data'][0]['url']

# Async function to download and resize an image
async def download_and_resize_image(url: str, size: tuple, image_path: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.read()
                image = Image.open(BytesIO(data))
                resized_image = image.resize(size, Image.Resampling.LANCZOS)
                resized_image.save(image_path, format="PNG")
                with open(image_path, "rb") as image_file:
                    img_str = base64.b64encode(image_file.read()).decode()
                return img_str
    return None

# Function to generate a detailed description of the image using GPT-4 Vision
def generate_image_description(image_base64: str):
    prompt = [
        {"role": "user", "content": "Describe the image in detail."},
        {"role": "user", "content": {"type": "image", "data": image_base64}}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=prompt,
        max_tokens=300,
    )
    return response.choices[0].message['content']

# Function to generate a question from a detailed description using GPT-4
def generate_mcq_from_description(description: str, tone: str, subject: str):
    prompt = [
        {"role": "system", "content": "You are an expert in generating educational content."},
        {"role": "user", "content": f"Based on the following detailed description of an image related to the topic '{subject}', generate a multiple-choice question in a {tone} tone with 4 options and provide the correct answer: {description}"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=prompt,
        max_tokens=1000,
        temperature=0.5
    )
    return response.choices[0].message['content']

# Consolidated function to generate image and MCQ
async def generate_image_mcq(number: int, subject: str, tone: str):
    images_and_questions = []
    for i in range(number):
        try:
            # Generate image
            image_prompt = f"An illustration representing the topic: {subject}"
            image_url = generate_image(image_prompt)
            
            # Resize and encode image
            image_path = os.path.join(image_dir, f"resized_image_{i}.png")
            resized_image_base64 = await download_and_resize_image(image_url, (750, 319), image_path)
            
            if resized_image_base64:
                # Generate a detailed description of the image
                description = generate_image_description(resized_image_base64)
                
                # Generate MCQ based on the detailed description
                mcq_text = generate_mcq_from_description(description, tone, subject)
                
                images_and_questions.append({
                    'image_url': image_url,
                    'resized_image_base64': resized_image_base64,
                    'description': description,
                    'mcq': mcq_text
                })
        except Exception as e:
            logging.error(f"Error processing image {i}: {e}")
            images_and_questions.append({
                'error': f"Error processing image {i}: {e}"
            })

    return images_and_questions

@app.route('/generate_content', methods=['GET'])
async def generate_content():
    number = request.args.get('number')
    subject = request.args.get('subject')
    tone = request.args.get('tone')

    if not number or not subject or not tone:
        return jsonify({"error": "Missing required parameters: number, subject, and tone"}), 400

    try:
        number = int(number)
    except ValueError:
        return jsonify({"error": "number must be an integer"}), 400

    try:
        content = await generate_image_mcq(number, subject, tone)
        return jsonify(content)
    except Exception as e:
        logging.error(f"Internal server error: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
