import asyncio
from dotenv import load_dotenv
load_dotenv()
from browser_use import Agent
from langchain_openai import ChatOpenAI
import os

OPENAI_API_KEY="sk-c64iuwxrlh3irht6"
OPENAI_API_BASE_URL="https://cloud.infini-ai.com/maas/v1"

async def main():
    agent = Agent(
        task="Search for the Nature journal's Scientific Reports conference proceedings from 2012 to locate the relevant article that does not mention plasmons or plasmonics",
        llm=ChatOpenAI(
            model="gpt-4o-2024-11-20",
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_BASE_URL
        ),
    )
    await agent.run()

asyncio.run(main())