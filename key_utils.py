import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Collect API keys
API_KEYS = [
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4"),
    os.getenv("GOOGLE_API_KEY_5"),
    os.getenv("GOOGLE_API_KEY_6"),
    os.getenv("GOOGLE_API_KEY_7"),
    os.getenv("GOOGLE_API_KEY_8"),
    os.getenv("GOOGLE_API_KEY_9"),
    os.getenv("GOOGLE_API_KEY_10"),
    os.getenv("GOOGLE_API_KEY_11"),
    os.getenv("GOOGLE_API_KEY_12"),
]
GMAIL_NAMES = [
    "sijanpaudel",
    "sijan.paudel10",
    "paudelsijan15",
    "aayushkafle",
    "aasutoshregmi",
    "pshreesha30",
    "nepalcric4",
    "prabeshsubedi",
    "sandeshtiwari",
    "iamengineer",
    "birajmohanta",
    "saharamanandhar",
    # 12th key if exists, or truncate list
]

# Create mapping: key -> gmail
KEY_TO_GMAIL = dict(zip(API_KEYS, GMAIL_NAMES[:len(API_KEYS)]))

def get_api_keys():
    """Return dictionary mapping API key -> Gmail name."""
    return KEY_TO_GMAIL


# Round-robin key generator
def key_generator():
    while True:
        for key in API_KEYS:
            yield key

_key_rotator = key_generator()

def get_next_key():
    """Return the next API key and its Gmail name as a tuple."""
    key = next(_key_rotator)
    gmail = KEY_TO_GMAIL.get(key, "Unknown Gmail")
    return key, gmail

def get_api_keys():
    """Return dictionary mapping API key -> Gmail name."""
    return KEY_TO_GMAIL.copy()