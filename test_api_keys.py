#!/usr/bin/env python3
"""
Simple script to test all Google API keys with a basic prompt: "2+2"
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError
import time

# Load environment variables
load_dotenv()

# Test configuration
TEST_PROMPT = "What is 2+2? Just give me the number."
MODEL_NAME = "gemini-2.5-flash"


def test_single_key(api_key, key_name, key_number):
    """Test a single API key"""
    print(f"\n{'='*60}")
    print(f"Testing Key #{key_number}: {key_name}")
    print(f"{'='*60}")

    if not api_key:
        print(f"❌ FAILED: API key is empty or not found")
        return False

    try:
        # Initialize the LLM with the API key
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=api_key
        )

        # Create a simple message
        messages = [HumanMessage(content=TEST_PROMPT)]

        # Make the API call
        print(f"📤 Sending prompt: '{TEST_PROMPT}'")
        response = llm.invoke(messages)

        # Check response
        if response and response.content:
            print(f"✅ SUCCESS: Got response - '{response.content.strip()}'")
            return True
        else:
            print(f"❌ FAILED: Empty response received")
            return False

    except ResourceExhausted as e:
        print(f"❌ FAILED: Quota exceeded or rate limit hit")
        print(f"   Error details: {str(e)[:100]}")
        return False

    except GoogleAPICallError as e:
        print(f"❌ FAILED: Google API error")
        print(f"   Error details: {str(e)[:100]}")
        return False

    except ValueError as e:
        print(f"❌ FAILED: Invalid API key or configuration error")
        print(f"   Error details: {str(e)[:100]}")
        return False

    except Exception as e:
        print(f"❌ FAILED: Unexpected error - {type(e).__name__}")
        print(f"   Error details: {str(e)[:100]}")
        return False


def main():
    """Main function to test all API keys"""
    print("\n" + "="*60)
    print("🔑 GOOGLE API KEY TESTING SCRIPT")
    print("="*60)
    print(f"Test Prompt: '{TEST_PROMPT}'")
    print(f"Model: {MODEL_NAME}")

    # Load all API keys
    api_keys = []
    gmail_names = []

    for i in range(1, 14):  # Testing keys 1-13
        key = os.getenv(f"GOOGLE_API_KEY_{i}")
        name = os.getenv(f"GMAIL_NAME_{i}", f"Unknown_{i}")

        if key:
            api_keys.append((i, key, name))

    print(f"\n📊 Found {len(api_keys)} API keys to test\n")

    # Test each key
    results = []
    working_keys = []
    failed_keys = []

    for key_number, api_key, gmail_name in api_keys:
        success = test_single_key(api_key, gmail_name, key_number)
        results.append((key_number, gmail_name, success))

        if success:
            working_keys.append((key_number, gmail_name))
        else:
            failed_keys.append((key_number, gmail_name))

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Total Keys Tested: {len(results)}")
    print(f"✅ Working Keys: {len(working_keys)}")
    print(f"❌ Failed Keys: {len(failed_keys)}")
    print(f"Success Rate: {len(working_keys)/len(results)*100:.1f}%")

    if working_keys:
        print("\n✅ Working Keys:")
        for key_num, name in working_keys:
            print(f"   - Key #{key_num}: {name}")

    if failed_keys:
        print("\n❌ Failed Keys:")
        for key_num, name in failed_keys:
            print(f"   - Key #{key_num}: {name}")

    print("\n" + "="*60)
    print("Testing completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
