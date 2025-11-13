import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Load API keys
API_KEYS = [os.getenv(f"GOOGLE_API_KEY_{i}") for i in range(1, 13)]

# Load Gmail names
GMAIL_NAMES = [os.getenv(f"GMAIL_NAME_{i}") for i in range(1, 13)]

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