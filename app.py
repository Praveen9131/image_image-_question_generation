import os
import logging
from quart import Quart, request, jsonify, send_file
import openai
from dotenv import load_dotenv
from PIL import Image
import aiohttp
import asyncio
from io import BytesIO
from pathlib import Path

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable not set")
else:
    logger.info(f"OpenAI API Key Loaded: {openai_api_key[:5]}...")  # Log partial key for security

openai.api_key = openai_api_key

app = Quart(__name__)

# Directory to store images
image_dir = Path("image_store")
image_dir.mkdir(parents=True, exist_ok=True)

# Async function to download and resize image
async def download_and_resize_image(image_url, target_size, image_name):
    try:
        logger.info(f"Downloading image from URL: {image_url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                response.raise_for_status()
                image_data = await response.read()
                image = Image.open(BytesIO(image_data))

        original_size = image.size
        logger.info(f"Original image size: {original_size}")

        resized_image = image.resize(target_size, Image.LANCZOS)
        resized_size = resized_image.size
        logger.info(f"Resized image size: {resized_size}")

        image_path = image_dir / image_name
        resized_image.save(image_path, format='PNG')

        logger.info(f"Image resized and saved to file: {image_path}")
        return str(image_path)
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return None

# Function to generate an image using DALL-E 3
async def generate_image(prompt: str):
    try:
        logger.info(f"Generating image with prompt: {prompt}")
        response = await asyncio.to_thread(openai.Image.create,
                                           model="dall-e-3",
                                           prompt=prompt,
                                           n=1,
                                           size="1024x1024")
        image_url = response['data'][0]['url']
        logger.info(f"Image generated with URL: {image_url}")
        return image_url
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None

# Async function to generate multiple image options based on a prompt
async def generate_image_options(prompts: list):
    tasks = [generate_image(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)

# Async function to generate a question with image options based on a description
async def generate_mcq_with_image_options(description: str):
    description_prompt = [
        {"role": "system", "content": "You are an expert in generating educational content."},
        {"role": "user", "content": f"Generate a multiple-choice question with four options based on the following description. Use the following format:\n\n**Question:** [Question based on the description]\n\n**Options:**\n1. [Option 1]\n2. [Option 2]\n3. [Option 3]\n4. [Option 4]\n\n**Correct Answer:** [Correct Option]\n\nDescription: {description}"}
    ]

    try:
        response = await asyncio.to_thread(openai.ChatCompletion.create,
                                           model="gpt-4",
                                           messages=description_prompt,
                                           max_tokens=1000,
                                           temperature=0.5)
        content = response.choices[0].message['content']
        logger.info(f"MCQ content generated: {content}")
    except Exception as e:
        logger.error(f"Error generating MCQ with image options: {e}")
        return {"error": "Failed to generate MCQ"}

    try:
        question_section = content.split("**Question:**")[1].split("**Options:**")[0].strip()
        options_section = content.split("**Options:**")[1].split("**Correct Answer:**")[0].strip()
        correct_answer = content.split("**Correct Answer:**")[1].strip()

        options = options_section.split('\n')
        option_prompts = [option.split('. ')[1] for option in options]

        option_images = await generate_image_options(option_prompts)

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
async def generate_content():
    try:
        topic = request.args.get('topic')
        num_questions = int(request.args.get('num_questions'))

        logger.info(f"Generating content for topic: {topic} with {num_questions} questions")

        images_and_questions = []
        tasks = []

        for i in range(num_questions):
            image_prompt = f"An illustration representing the topic: {topic}"
            tasks.append(generate_image(image_prompt))

        question_image_urls = await asyncio.gather(*tasks)

        for i, question_image_url in enumerate(question_image_urls):
            if not question_image_url:
                return jsonify({"error": "Failed to generate question image"}), 500

            description = f"This is an illustration representing the topic '{topic}'."
            mcq_with_images = await generate_mcq_with_image_options(description)
            if "error" in mcq_with_images:
                return jsonify(mcq_with_images), 500

            mcq_with_images["question_image_url"] = question_image_url
            images_and_questions.append(mcq_with_images)

        # Resize images and store in file system
        resize_tasks = []

        for i, item in enumerate(images_and_questions):
            question_image_url = item["question_image_url"]
            resize_tasks.append(download_and_resize_image(question_image_url, (750, 319), f"question_image_{i+1}.png"))

        resized_question_images = await asyncio.gather(*resize_tasks[:num_questions])

        for i, item in enumerate(images_and_questions):
            item["question_image_url_resized"] = f"/image/question_image_{i+1}.png"

        resize_option_tasks = []

        for i, item in enumerate(images_and_questions):
            for j, option_key in enumerate(item["options"]):
                option_image_url = item["options"][option_key]
                resize_option_tasks.append((option_key, download_and_resize_image(option_image_url, (270, 140), f"option_image_{i+1}_{j+1}.png")))

        resized_option_images = await asyncio.gather(*[task[1] for task in resize_option_tasks])

        option_counter = 0
        for i, item in enumerate(images_and_questions):
            for j, option_key in enumerate(list(item["options"].keys())):
                item["options"][option_key + "_resized"] = f"/image/option_image_{i+1}_{j+1}.png"
                option_counter += 1

        return jsonify(images_and_questions)
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/image/<filename>', methods=['GET'])
async def get_image(filename):
    logger.info(f"Fetching image with filename: {filename}")
    image_path = image_dir / filename
    if image_path.exists():
        return await send_file(
            str(image_path),
            mimetype='image/png'
        )
    else:
        logger.error(f"Image not found: {filename}")
        return jsonify({"error": "Image not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
