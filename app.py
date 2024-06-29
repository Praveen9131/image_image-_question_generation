import os
import logging
from quart import Quart, request, jsonify, send_file
import openai
from dotenv import load_dotenv
from PIL import Image
import aiohttp
from io import BytesIO
import asyncio

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

app = Quart(__name__)

# In-memory storage for images
image_store = {}

# Function to download and resize image
async def download_and_resize_image(image_url, target_size):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, timeout=10) as response:
                response.raise_for_status()
                image_data = await response.read()
                
        image = Image.open(BytesIO(image_data))
        
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
async def generate_image(prompt: str, retries=3):
    for attempt in range(retries):
        try:
            response = await openai.Image.create(
                model="dall-e-3",
                prompt=f"Create an educational illustration about {prompt}. The illustration should be clear and informative.",
                n=1,
                size="1024x1024"
            )
            return response['data'][0]['url']
        except openai.error.OpenAIError as e:
            logger.error(f"Error generating image: {e}")
            if "safety system" in str(e):
                return None
            await asyncio.sleep(2 ** attempt)
    return None

# Function to describe an image using GPT-4
async def describe_image(image_url: str):
    image_prompt = [
        {"role": "system", "content": "You are an expert in describing images."},
        {"role": "user", "content": f"Describe the content of the following image URL in detail: {image_url}"}
    ]
    
    try:
        response = await openai.ChatCompletion.create(
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
async def generate_image_options(prompts: list):
    tasks = [generate_image(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    options = []
    for result in results:
        if result:
            options.append(result)
        else:
            logger.error(f"Failed to generate image for prompt")
            options.append("https://via.placeholder.com/270x140.png?text=Image+Unavailable")
    return options

# Function to generate a question with image options based on a description
async def generate_mcq_with_image_options(topic: str, description: str):
    description_prompt = [
        {"role": "system", "content": "You are an expert in generating educational content."},
        {"role": "user", "content": f"Generate a multiple-choice question with four options based on the following description and topic. Ensure the options are closely related to the question. Use the following format:\n\n**Question:** [Question based on the description]\n\n**Options:**\n1. [Option 1]\n2. [Option 2]\n3. [Option 3]\n4. [Option 4]\n\n**Correct Answer:** [Correct Option]\n\nTopic: {topic}\n\nDescription: {description}"}
    ]
    
    try:
        response = await openai.ChatCompletion.create(
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
        num_questions = request.args.get('num_questions')

        if not topic or not num_questions:
            logger.error("Missing 'topic' or 'num_questions' parameter")
            return jsonify({"error": "Missing 'topic' or 'num_questions' parameter"}), 400

        try:
            num_questions = int(num_questions)
        except ValueError:
            logger.error("'num_questions' must be an integer")
            return jsonify({"error": "'num_questions' must be an integer"}), 400

        images_and_questions = []
        for _ in range(num_questions):
            image_prompt = f"An educational illustration representing the topic: {topic}. The illustration should be clear and informative."
            question_image_url = await generate_image(image_prompt)
            if not question_image_url:
                logger.error("Failed to generate question image")
                return jsonify({"error": "Failed to generate question image"}), 500

            # Describe the image
            description = await describe_image(question_image_url)
            if not description:
                logger.error("Failed to describe the image")
                return jsonify({"error": "Failed to describe the image"}), 500

            # Generate MCQ based on the description
            mcq_with_images = await generate_mcq_with_image_options(topic, description)
            if "error" in mcq_with_images:
                logger.error(f"Error in generating MCQ with image options: {mcq_with_images['error']}")
                return jsonify(mcq_with_images), 500

            mcq_with_images["question_image_url"] = question_image_url
            images_and_questions.append(mcq_with_images)

        # Resize images and store in memory
        for item in images_and_questions:
            question_image_url = item["question_image_url"]
            question_image_key = await download_and_resize_image(question_image_url, (750, 319))
            if question_image_key:
                item["question_image_url"] = f"http://127.0.0.1:5000/image/{question_image_key}"

            for option_key in item["options"]:
                option_image_url = item["options"][option_key]
                option_image_key = await download_and_resize_image(option_image_url, (270, 140))
                if option_image_key:
                    item["options"][option_key] = f"http://127.0.0.1:5000/image/{option_image_key}"

        return jsonify(images_and_questions)
    except Exception as e:
        logger.error(f"Error generating content: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/image/<image_key>', methods=['GET'])
async def get_image(image_key):
    if image_key in image_store:
        return await send_file(
            BytesIO(image_store[image_key].getvalue()),
            mimetype='image/png'
        )
    else:
        logger.error("Image not found")
        return jsonify({"error": "Image not found"}), 404

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
