import boto3
import os
import dotenv
import json
from botocore.exceptions import ClientError

# ✅ Load AWS credentials from .env file
dotenv.load_dotenv()

# ✅ Initialize Amazon Bedrock client
bedrock_client = boto3.client(
    'bedrock-runtime',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name="us-east-1"
)

# ✅ Initialize Amazon Polly client
polly_client = boto3.client(
    "polly",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="us-east-1"
)

# ✅ Correct Model ID for Amazon Bedrock
model_id = "us.meta.llama3-3-70b-instruct-v1:0"

# ✅ Master Prompt for Amazon Bedrock
master_prompt = """
You are a storyteller.
Tell a story only about Harry Potter.
Be familiar with all of cast in Harry Potter.
Start the story with once upon a time.
"""

# ✅ Function to Convert Text to MP3
def text_to_mp3(text, file_name="output.mp3"):
    try:
        response = polly_client.synthesize_speech(
            Text=text,
            Engine="neural",
            OutputFormat="mp3",
            VoiceId="Amy"  # Change to preferred voice
        )

        # ✅ Save MP3 file correctly
        with open(file_name, "wb") as file:
            file.write(response["AudioStream"].read())

        print(f"\n🎧 MP3 saved successfully as {file_name}")

    except ClientError as e:
        print(f"❌ ERROR: Polly synthesis failed. Reason: {e}")

# ✅ Function to Generate Text from Amazon Bedrock
def generate_text_from_bedrock(user_prompt):
    conversation = [
        {
            "role": "user",
            "content": [{"text": user_prompt}],
        }
    ]

    try:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=conversation,
            system=[{"text": master_prompt}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
        )

        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text

    except (ClientError, Exception) as e:
        print(f"❌ ERROR: Bedrock failed. Reason: {e}")
        return None

# ✅ Main Execution
if __name__ == "__main__":
    user_message = input("\nEnter your prompt: ")

    if user_message.lower() in ['quit', 'exit', 'bye']:
        print("\nGoodbye!")
    else:
        # ✅ Generate Story from Amazon Bedrock
        story_text = generate_text_from_bedrock(user_message)

        if story_text:
            print(f"\nAssistant: {story_text}")

            # ✅ Convert AI-generated Story to Speech
            text_to_mp3(story_text, "harry_potter_story.mp3")
