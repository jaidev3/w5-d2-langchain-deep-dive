# Auto Email Responder

This project is an intelligent email response system integrated with Gmail using LangChain. It retrieves company policies, generates responses, and handles batch processing with caching.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Obtain Gmail API credentials:
   - Follow [Google's guide](https://developers.google.com/gmail/api/quickstart/python) to download `credentials.json`.
   - Place it in the project directory.
3. Set OpenAI API key: Export `OPENAI_API_KEY` environment variable.

## Usage

Run `python main.py` to start the system. It will periodically check for new emails and respond accordingly.
