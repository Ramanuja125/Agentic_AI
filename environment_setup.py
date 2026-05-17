from dotenv import load_dotenv
import os

load_dotenv()  # reads .env from current folder
api_key = os.getenv("GOOGLE_API_KEY")