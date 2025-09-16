from dotenv import load_dotenv
import os
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

image_prompt = "Children building a snowman in a snowy village with trees and lights"
response = openai.images.generate(
    model="dall-e-3",
    prompt=image_prompt,
    size="1024x1024",  
    quality="standard",  
    n=1
)

image_url = response.data[0].url
print("Generated Image URL:", image_url)
 