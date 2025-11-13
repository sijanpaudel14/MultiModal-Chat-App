import os
from LLMHandler import LLMHandler
import gradio as gr
import base64
from PIL import Image
import io
import tempfile
import scipy.io.wavfile
handler = LLMHandler()


# ==============================================================================
# Step 2: Define the Core Gradio Functions (Refactored)
# ==============================================================================

# --- NEW, SEPARATED HANDLER for Text Chat ---
def handle_chat(message, history):
    # 1. Add user message to history
    history.append({"role": "user", "content": message})

    # 2. Call the handler for a text response
    response = handler.run(message, task_type="text")
    assistant_reply = response['data'] if response else "Sorry, an error occurred."

    # 3. Add assistant response to history
    history.append({"role": "assistant", "content": assistant_reply})

    # The return signature must match the outputs for the chat event listener.
    # It needs to update the shared history, all chatbots, and clear the input textbox.
    return history, history, history, history, history, history, history, ""

# --- NEW, SEPARATED HANDLER for Image Generation ---


def handle_image_generation(prompt, history):
    image_output_display = None
    image_output_path = None

    # 1. Add user's request to the conversation history for context
    history.append(
        {"role": "user", "content": f"(User wants to generate an image with the prompt: '{prompt}')"})

    # 2. Call the handler for an image response
    if prompt:
        response = handler.run(prompt, task_type="image")
        if response and response.get('type') == 'image' and response.get('data'):
            img_bytes = base64.b64decode(response['data'])
            image_output_display = Image.open(io.BytesIO(img_bytes))

            # Save to a temporary file for the download link
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    image_output_display.save(temp_file.name)
                    image_output_path = temp_file.name
            except Exception as e:
                print(f"❌ Error saving temporary image file for download: {e}")

            history.append(
                {"role": "assistant", "content": "Here is the image you requested."})
        else:
            history.append(
                {"role": "assistant", "content": "Sorry, I couldn't generate the image."})

    # The return signature must match the outputs for the image generation event listener.
    # It updates the image display, the download file, the shared history, and all chatbots.
    return image_output_display, image_output_path, history, history, history, history, history, history, history

# --- HANDLER 2: Image Analysis ---


def handle_image_analysis(image, prompt, history):
    # This is a defensive check in case the button is clicked without an image
    if image is None:
        history.append(
            {"role": "user", "content": "(User tried to analyze an image but provided none)"})
        history.append(
            {"role": "assistant", "content": "Please upload an image first to analyze it."})
        return history, history, history, history, history, history, history

    # --- THE FIX IS HERE ---
    # 1. Create a temporary file to save the PIL image object.
    #    'delete=False' is important so the file persists after the 'with' block.
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            image.save(temp_file.name)
            temp_file_path = temp_file.name  # Get the path of the saved file
    except Exception as e:
        print(f"❌ Error saving temporary image file: {e}")
        history.append(
            {"role": "assistant", "content": "Sorry, there was an error processing the uploaded image."})
        return history, history, history, history, history, history, history
    # --- END OF FIX ---

    # Now, use the temporary file path with the handler, as it expects
    prompt = prompt or "Describe this image in detail."
    history.append(
        {"role": "user", "content": f"(User uploaded an image with the prompt: '{prompt}')"})

    # Call the handler with the correct file path
    response = handler.generate_text_from_image(
        image_source=temp_file_path, prompt=prompt)

    # Clean up the temporary file after it has been used
    os.remove(temp_file_path)

    # Process the response as before
    assistant_reply = response['data'] if response and 'data' in response else "Sorry, I couldn't analyze the image."
    history.append({"role": "assistant", "content": assistant_reply})

    return history, history, history, history, history, history, history


# --- HANDLER 3 & 4: Transcription ---
def handle_audio_transcription(audio, history):
    temp_path = audio
    if not isinstance(audio, str):
        sr, y = audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            scipy.io.wavfile.write(temp_audio.name, sr, y)
            temp_path = temp_audio.name

    history.append(
        {"role": "user", "content": "(User uploaded audio for transcription)"})
    transcript = handler.generate_text_from_audio(file_name=temp_path)
    if temp_path != audio:
        os.remove(temp_path)

    if transcript:
        # Add the full transcript to the conversation history for context
        history.append(
            {"role": "assistant", "content": f"Here is the transcript:\n\n---\n\n{transcript}"})
    else:
        transcript = "Transcription failed."
        history.append({"role": "assistant", "content": transcript})

    return transcript, history, history, history, history, history, history, history

# --- HANDLER 5: Follow-up Chatting ---


def handle_follow_up_chat(message, history):
    history.append({"role": "user", "content": message})
    response = handler.run(message, task_type="text")
    assistant_reply = response['data'] if response else "Sorry, an error occurred."
    history.append({"role": "assistant", "content": assistant_reply})
    # Clear the input box
    return history, history, history, history, history, history, history, ""

# --- HANDLER 6: Dedicated Audio Generation ---


def handle_text_to_audio(text_to_speak):
    if not text_to_speak or not text_to_speak.strip():
        return None, None
    response = handler.run(text_to_speak, task_type="audio")
    if response and response.get('type') == 'audio' and response.get('data'):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(response['data'])
            # Return path to both Audio and File components
            return temp_audio.name, temp_audio.name
    return None, None


# ==============================================================================
# Step 2 & 4: Define Handlers and Wire Events (Final, Final Version)
# ==============================================================================

# ... (Keep existing handlers: handle_chat_and_image_gen, handle_image_analysis, handle_audio_transcription) ...

# --- HANDLER for Video Transcription ---
def handle_video_transcription(video, history):
    if video is None:
        return None, history, history, history, history, history, history, history
    history.append(
        {"role": "user", "content": f"(User uploaded video for AUDIO transcription: {os.path.basename(video)})"})
    transcript = handler.generate_transcript_from_video(file_name=video)
    if transcript:
        history.append(
            {"role": "assistant", "content": f"Here is the audio transcript:\n\n---\n\n{transcript}"})
    else:
        transcript = "Audio transcription failed."
        history.append({"role": "assistant", "content": transcript})
    return transcript, history, history, history, history, history, history, history
# --- NEW HANDLER for Video Analysis ---


def handle_video_analysis(video, prompt, history):
    if video is None:
        return "Please upload a video first.", history, history, history, history, history, history, history
    prompt = prompt or "Describe this video in detail."
    history.append(
        {"role": "user", "content": f"(User requested VISUAL analysis for video '{os.path.basename(video)}' with prompt: '{prompt}')"})
    analysis_result = handler.analyze_video(file_name=video, prompt=prompt)

    if analysis_result and analysis_result.get('data'):
        analysis_text = analysis_result['data']
        history.append(
            {"role": "assistant", "content": f"Here is the video analysis:\n\n---\n\n{analysis_text}"})
    else:
        analysis_text = "Video analysis failed."
        history.append({"role": "assistant", "content": analysis_text})

    return analysis_text, history, history, history, history, history, history, history

# ... (Keep existing handlers: handle_follow_up_chat, handle_text_to_audio) ...

# --- FINAL, UPDATED Clear All Handler ---


# The number of 'None' or '[]' values returned now exactly matches the length of the list above.
def clear_all_inputs():
    """Clears the handler's memory and all UI components."""
    print("--- Clearing Chat and UI ---")
    handler.clear_history()
    return (
        # Live Chat Tab (2 components)
        [], None,
        # Image Generation Tab (4 components)
        None, None, None, [],
        # Image Analysis Tab (3 components)
        None, None, [],
        # Audio Transcription Tab (4 components)
        None, None, [], None,
        # Video Transcription Tab (4 components)
        None, None, [], None,
        # Video Analysis Tab (5 components)
        None, None, None, [], None,
        # Audio Generation Tab (3 components)
        None, None, None,
        # Shared State (1 component)
        []
    )


# ... (tabs.select for clearing history remains the same) ...
# Don't forget to update the handle_tab_change function to return the correct number of empty lists
def handle_tab_change(evt: gr.SelectData):
    print(f"Tab changed to index: {evt.index}. Clearing conversation history.")
    handler.clear_history()
    empty_history = []
    # shared_history + 5 chatbots
    return empty_history, empty_history, empty_history, empty_history, empty_history, empty_history, empty_history


# ==============================================================================
# Step 3 & 4: Build the Gradio UI and Wire Up Event Listeners (Beautified)
# ==============================================================================
custom_css = """
.avatar-image {
    width: 60px !important;
    height: 60px !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Multimodal AI Assistant", css=custom_css) as ui:
    gr.Markdown("# 🤖 Multimodal AI Assistant")
    gr.Markdown(
        "A unified interface for text, image, audio, and video interaction with AI.")

    shared_chat_history = gr.State([])

    with gr.Tabs() as tabs:
        # ===============================================
        # 💬 TAB: Live Chat (NEW)
        # ===============================================
        with gr.TabItem("💬 Live Chat", id=0):
            gr.Markdown("## Live Conversation")
            gr.Markdown(
                "*Have a conversation with the AI or ask it to use its tools (e.g., 'check flight price to Lisbon').*")
            chatbot_main = gr.Chatbot(
                label="Conversation", height=450, type="messages",
                avatar_images=(
                    None, "https://img.freepik.com/free-vector/graident-ai-robot-vectorart_78370-4114.jpg")
            )
            text_prompt_main = gr.Textbox(
                label="Your Message", placeholder="Type your message here...", scale=1
            )

        # ===============================================
        # 🎨 TAB: Image Generation (NEW)
        # ===============================================
        with gr.TabItem("🎨 Image Generation", id=1):
            gr.Markdown("## Generate an Image from a Prompt")
            gr.Markdown(
                "*Describe the image you want to create. The conversation history will be updated with your request.*")
            with gr.Row():
                with gr.Column(scale=3):
                    image_display = gr.Image(
                        label="Generated Image", height=450, interactive=False)
                    image_download_file = gr.File(
                        label="Download Generated Image")
                with gr.Column(scale=2):
                    text_prompt_image_gen = gr.Textbox(
                        label="Image Prompt", lines=4, placeholder="e.g., A watercolor painting of a robot reading a book...")
                    submit_image_gen_btn = gr.Button(
                        "Generate Image", variant="primary")
                    chatbot_image_gen_display = gr.Chatbot(label="Conversation Context", height=400, type="messages", avatar_images=(
                        None, "https://img.freepik.com/free-vector/graident-ai-robot-vectorart_78370-4114.jpg"))

        # ===============================================
        # 🎤 TAB: Generate & Download Audio
        # ===============================================
        with gr.TabItem("🎤 Generate Audio", id=2):
            gr.Markdown("## Generate Speech from Text")
            with gr.Group():
                text_prompt_audio_gen = gr.Textbox(
                    label="Text to Synthesize", lines=5,
                    placeholder="e.g., 'Hello, welcome to the future of AI.'"
                )
                generate_audio_btn = gr.Button(
                    "Generate Audio", variant="primary")
                audio_output_gen = gr.Audio(
                    label="Playback", show_download_button=True)
                audio_download_file = gr.File(
                    label="Download Audio File (.wav)")

        # ===============================================
        # 🖼️ TAB: Image Analysis
        # ===============================================
        with gr.TabItem("🖼️ Analyze Image", id=3):
            gr.Markdown("## Analyze Uploaded Image")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("### Step 1: Upload & Ask")
                        image_upload = gr.Image(
                            type="pil", label="Upload Image Here")
                        text_prompt_image = gr.Textbox(
                            label="Your Question", placeholder="e.g., 'What is in this image?'...", lines=2
                        )
                        submit_image_btn = gr.Button(
                            "Analyze Image", variant="primary", scale=1)

                with gr.Column(scale=3):
                    gr.Markdown("### Step 2: View Conversation")
                    chatbot_image_display = gr.Chatbot(
                        label="Conversation Context", height=450, type="messages",
                        avatar_images=(
                            None, "https://img.freepik.com/free-vector/graident-ai-robot-vectorart_78370-4114.jpg")
                    )

        # ===============================================
        # 👁️ TAB: Video Visual Analysis
        # ===============================================
        with gr.TabItem("👁️ Analyze Video", id=4):
            gr.Markdown("## Analyze Video Content")
            with gr.Accordion("Step 1: Upload Video & Ask a Question", open=True):
                video_upload_analyze = gr.Video(label="Input Video")
                text_prompt_video_analyze = gr.Textbox(
                    label="Your Question About the Video", lines=3,
                    placeholder="e.g., 'Describe the scene at 0:15' or 'What is the main color of the car?'"
                )
                submit_video_analyze_btn = gr.Button(
                    "Analyze Video Content", variant="primary")

            with gr.Accordion("Step 2: View Analysis & Ask Follow-up Questions", open=True):
                analysis_output_video = gr.Textbox(
                    label="Full Analysis Result", lines=10, interactive=False)
                chatbot_video_analyze = gr.Chatbot(
                    label="Conversation about Video Content", height=300, type="messages", avatar_images=(None, "https://img.freepik.com/free-vector/graident-ai-robot-vectorart_78370-4114.jpg"))
                text_prompt_video_analyze_chat = gr.Textbox(
                    label="Ask a follow-up question...", placeholder="e.g., 'Elaborate on the second point.'")

        # ===============================================
        # 🎙️ TAB: Audio Transcription & Chat
        # ===============================================
        with gr.TabItem("🎙️ Transcribe (Audio)", id=5):
            gr.Markdown("## Transcribe Audio and Chat")
            with gr.Accordion("Step 1: Upload or Record Audio", open=True):
                audio_upload = gr.Audio(type="numpy", label="Input Audio", sources=[
                                        "upload", "microphone"])
                submit_audio_btn = gr.Button(
                    "Transcribe Audio", variant="primary")

            with gr.Accordion("Step 2: View Transcript & Ask Questions", open=True):
                transcript_output_audio = gr.Textbox(
                    label="Full Transcription Result", lines=10, interactive=False)
                chatbot_audio = gr.Chatbot(
                    label="Conversation about Audio", height=300, type="messages", avatar_images=(None, "https://img.freepik.com/free-vector/graident-ai-robot-vectorart_78370-4114.jpg"))
                text_prompt_audio = gr.Textbox(
                    label="Ask about the audio...", placeholder="e.g., 'Summarize this in 3 points.'")

        # ===============================================
        # 📹 TAB: Video Audio Transcription
        # ===============================================
        with gr.TabItem("📹 Transcribe Video Audio", id=6):
            gr.Markdown("## Transcribe Video Audio")
            with gr.Accordion("Step 1: Upload Video for Audio Transcription", open=True):
                video_upload_transcribe = gr.Video(label="Input Video")
                submit_video_transcribe_btn = gr.Button(
                    "Transcribe Video Audio", variant="primary")

            with gr.Accordion("Step 2: View Transcript & Ask Questions", open=True):
                transcript_output_video = gr.Textbox(
                    label="Full Transcription Result", lines=10, interactive=False)
                chatbot_video_transcribe = gr.Chatbot(
                    label="Conversation about Video Transcript", height=300, type="messages", avatar_images=(None, "https://img.freepik.com/free-vector/graident-ai-robot-vectorart_78370-4114.jpg"))
                text_prompt_video_transcribe = gr.Textbox(
                    label="Ask about the transcript...", placeholder="e.g., 'What were the key topics discussed?'")

    # =================================================
    # --- Global Controls ---
    # =================================================
    clear_btn = gr.Button("🗑️ Clear All Conversations & Reset", variant="stop")

    # =================================================
    # --- Event Listener Wiring (NO CHANGES NEEDED HERE) ---
    # =================================================
    all_chatbots = [
        chatbot_main, chatbot_image_gen_display, chatbot_image_display,
        chatbot_audio, chatbot_video_transcribe, chatbot_video_analyze
    ]

    # This list now accurately reflects all components in the new UI layout.
    all_clearable_components = [
        # Live Chat Tab
        chatbot_main, text_prompt_main,
        # Image Generation Tab
        image_display, image_download_file, text_prompt_image_gen, chatbot_image_gen_display,
        # Image Analysis Tab
        image_upload, text_prompt_image, chatbot_image_display,
        # Audio Transcription Tab
        audio_upload, transcript_output_audio, chatbot_audio, text_prompt_audio,
        # Video Transcription Tab
        video_upload_transcribe, transcript_output_video, chatbot_video_transcribe, text_prompt_video_transcribe,
        # Video Analysis Tab
        video_upload_analyze, text_prompt_video_analyze, analysis_output_video, chatbot_video_analyze, text_prompt_video_analyze_chat,
        # Audio Generation Tab
        text_prompt_audio_gen, audio_output_gen, audio_download_file,
        # Shared State
        shared_chat_history
    ]

# --- EVENT for the new Live Chat tab ---
    text_prompt_main.submit(
        fn=handle_chat,
        inputs=[text_prompt_main, shared_chat_history],
        outputs=[shared_chat_history] + all_chatbots + [text_prompt_main]
    )

    # --- EVENT for the new Image Generation tab ---
    submit_image_gen_btn.click(
        fn=handle_image_generation,
        inputs=[text_prompt_image_gen, shared_chat_history],
        outputs=[image_display, image_download_file,
                 shared_chat_history] + all_chatbots
    )

    submit_image_btn.click(
        fn=handle_image_analysis,
        inputs=[image_upload, text_prompt_image, shared_chat_history],
        outputs=[shared_chat_history] + all_chatbots
    )
    submit_audio_btn.click(
        fn=handle_audio_transcription,
        inputs=[audio_upload, shared_chat_history],
        outputs=[transcript_output_audio, shared_chat_history] + all_chatbots
    )
    submit_video_transcribe_btn.click(
        fn=handle_video_transcription,
        inputs=[video_upload_transcribe, shared_chat_history],
        outputs=[transcript_output_video, shared_chat_history] + all_chatbots
    )
    submit_video_analyze_btn.click(
        fn=handle_video_analysis,
        inputs=[video_upload_analyze,
                text_prompt_video_analyze, shared_chat_history],
        outputs=[analysis_output_video, shared_chat_history] + all_chatbots
    )
    generate_audio_btn.click(
        fn=handle_text_to_audio,
        inputs=[text_prompt_audio_gen],
        outputs=[audio_output_gen, audio_download_file]
    )
    text_prompt_audio.submit(
        fn=handle_follow_up_chat,
        inputs=[text_prompt_audio, shared_chat_history],
        outputs=[shared_chat_history] + all_chatbots + [text_prompt_audio]
    )
    text_prompt_video_transcribe.submit(
        fn=handle_follow_up_chat,
        inputs=[text_prompt_video_transcribe, shared_chat_history],
        outputs=[shared_chat_history] + all_chatbots +
        [text_prompt_video_transcribe]
    )
    text_prompt_video_analyze_chat.submit(
        fn=handle_follow_up_chat,
        inputs=[text_prompt_video_analyze_chat, shared_chat_history],
        outputs=[shared_chat_history] + all_chatbots +
        [text_prompt_video_analyze_chat]
    )
    clear_btn.click(fn=clear_all_inputs, inputs=None,
                    outputs=all_clearable_components)

    tabs.select(
        fn=handle_tab_change,
        inputs=None,
        outputs=[shared_chat_history] + all_chatbots
    )
PORT = int(os.getenv("PORT", 7860))
# You would then launch the UI as before
ui.launch(server_name="0.0.0.0", server_port=PORT)
