from openai import OpenAI
import json
from logging import getLogger

MODEL_NAME = "gpt-5-nano"
SYSTEM_PROMPT = "You are a curmudgeonly assistant who is very direct and to the point. If you don't know something, say you don't know."
logger = getLogger(__name__)

def run_agent_loop():
    pass

def inference(input, api_key):
    """Run inference on the input using the OpenAI API."""
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=MODEL_NAME,
        instructions=SYSTEM_PROMPT,
        input=input,
    )
    logger.debug(response.model_dump_json(indent=2))
    return response.output_text