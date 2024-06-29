import os
import logging
from flask import Flask, request, jsonify, send_file
import openai
from dotenv import load_dotenv
from PIL import Image
import requests
from io import BytesIO
import base64

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable not set")
openai.api_key = openai_api_key

app = Flask(__name__)

# In-memory storage for images
image_store = {}

# Function to download and resize image
def download_and_resize_image(image_url, target_size):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        
        original_size = image.size
        logger.info(f"Original image size: {original_size}")
        
        resized_image = image.resize(target_size, Image.LANCZOS)
        resized_size = resized_image.size
        logger.info(f"Resized image size: {resized_size}")
        
        output = BytesIO()
        resized_image.save(output, format='PNG')
        output.seek(0)
        
        # Generate a unique key for storing the image in memory
        image_key = f"image_{len(image_store) + 1}.png"
        image_store[image_key] = output
        
        return image_key
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return None

# Function to generate an image using DALL-E 3
def generate_image(prompt: str):
    try:
        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return response['data'][0]['url']
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None

# Function to describe an image using GPT-4
def describe_image(image_url: str):
    image_prompt = [
        {"role": "system", "content": "You are an expert in describing images."},
        {"role": "user", "content": f"Describe the content of the following image URL in detail: {image_url}"}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=image_prompt,
            max_tokens=500,
            temperature=0.5
        )
        description = response.choices[0].message['content'].strip()
        return description
    except Exception as e:
        logger.error(f"Error describing image: {e}")
        return None

# Function to generate multiple image options based on a prompt
def generate_image_options(prompts: list):
    options = []
    for prompt in prompts:
        image_url = generate_image(prompt)
        if image_url:
            options.append(image_url)
        else:
            logger.error(f"Failed to generate image for prompt: {prompt}")
    return options

# Function to generate a question with image options based on a description
def generate_mcq_with_image_options(topic: str, description: str):
    description_prompt = [
        {"role": "system", "content": "You are an expert in generating educational content."},
        {"role": "user", "content": f"Generate a multiple-choice question with four options based on the following description and topic. Ensure the options are closely related to the question. Use the following format:\n\n**Question:** [Question based on the description]\n\n**Options:**\n1. [Option 1]\n2. [Option 2]\n3. [Option 3]\n4. [Option 4]\n\n**Correct Answer:** [Correct Option]\n\nTopic: {topic}\n\nDescription: {description}"}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=description_prompt,
            max_tokens=1000,
            temperature=0.5
        )
        content = response.choices[0].message['content']
    except Exception as e:
        logger.error(f"Error generating MCQ with image options: {e}")
        return {"error": "Failed to generate MCQ"}
    
    try:
        question_section = content.split("**Question:**")[1].split("**Options:**")[0].strip()
        options_section = content.split("**Options:**")[1].split("**Correct Answer:**")[0].strip()
        correct_answer = content.split("**Correct Answer:**")[1].strip()

        options = options_section.split('\n')
        option_prompts = [option.split('. ')[1] for option in options]

        option_images = generate_image_options(option_prompts)
        
        if correct_answer not in option_prompts:
            raise ValueError(f"Correct answer '{correct_answer}' not found in options: {option_prompts}")

        correct_answer_index = option_prompts.index(correct_answer)
        
        return {
            "question": question_section,
            "options": {
                "Option 1": option_images[0],
                "Option 2": option_images[1],
                "Option 3": option_images[2],
                "Option 4": option_images[3]
            },
            "correct_answer": f"Option {correct_answer_index + 1}"
        }
    except IndexError as e:
        logger.error(f"Error processing response: {e}")
        logger.error(f"Response content: {content}")
        return {
            "error": "Failed to parse the response from OpenAI",
            "response_content": content
        }
    except ValueError as e:
        logger.error(f"Error: {e}")
        return {
            "error": str(e),
            "response_content": content
        }

@app.route('/generate_content', methods=['GET'])
def generate_content():
    try:
        topic = request.args.get('topic')
        num_questions = int(request.args.get('num_questions'))

        images_and_questions = []
        for _ in range(num_questions):
            image_prompt = f"An educational illustration representing the topic: {topic}. The illustration should be clear and informative."
            question_image_url = generate_image(image_prompt)
            if not question_image_url:
                return jsonify({"error": "Failed to generate question image"}), 500

            # Describe the image
            description = describe_image(question_image_url)
            if not description:
                return jsonify({"error": "Failed to describe the image"}), 500

            # Generate MCQ based on the description
            mcq_with_images = generate_mcq_with_image_options(topic, description)
            if "error" in mcq_with_images:
                return jsonify(mcq_with_images), 500

            mcq_with_images["question_image_url"] = question_image_url
            images_and_questions.append(mcq_with_images)

        # Resize images and store in memory
        for item in images_and_questions:
            question_image_url = item["question_image_url"]
            question_image_key = download_and_resize_image(question_image_url, (750, 319))
            if question_image_key:
                item["question_image_url"] = f"http://127.0.0.1:5000/image/{question_image_key}"

            for option_key in item["options"]:
                option_image_url = item["options"][option_key]
                option_image_key = download_and_resize_image(option_image_url, (270, 140))
                if option_image_key:
                    item["options"][option_key] = f"http://127.0.0.1:5000/image/{option_image_key}"

        return jsonify(images_and_questions)
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/image/<image_key>', methods=['GET'])
def get_image(image_key):
    if image_key in image_store:
        return send_file(
            BytesIO(image_store[image_key].getvalue()),
            mimetype='image/png'
        )
    else:
        return jsonify({"error": "Image not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
