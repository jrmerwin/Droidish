# LD-1 operating system v 1.0 (LLM Droid 1)

import sys
import queue
import threading
import sounddevice as sd
import vosk
import json
import time
import os
import pygame
import re
from autogen import ConversableAgent

# ----------------------------------------------------------------
# 0. Instantiate Droidish Encoder
# ----------------------------------------------------------------
import soundfile
from scipy.signal import find_peaks

from droidish import (
    generate_tone_dictionaries,
    Encoder,
    Decoder,
    MIN_INDEX_DIFFERENCE,
    NUM_FREQUENCIES,
    MIN_FREQ,
    MAX_FREQ,
    NUM_WORDS_NEEDED,
    SAMPLE_RATE,
    FADE_DURATION,
    VOLUME,
    WHISTLE_THRESHOLD,
    CHIRP_THRESHOLD,
    SEED,
    TONE_LENGTH,
    WHISTLE_LENGTH
)

TONE_LENGTH = 0.1
WHISTLE_LENGTH = 0.1

# Generate dictionaries
common_words_dict, character_dict = generate_tone_dictionaries(
    SEED, NUM_WORDS_NEEDED, NUM_FREQUENCIES, MIN_FREQ, MAX_FREQ, MIN_INDEX_DIFFERENCE
)

# Create an encoder instance
encoder = Encoder(
    common_words_dict,
    character_dict,
    tone_length=TONE_LENGTH,
    sample_rate=44100,
    fade_duration=0.01,
    volume=0.5
)

# ------------------------------------------------------------
# 1. Set up Vosk model
# ------------------------------------------------------------
MODEL_PATH = "/home/lb1/droid/droids/vosk-model"
SAMPLE_RATE = 44100

vosk.SetLogLevel(-1)  # Optional: reduce log spam
model = vosk.Model(MODEL_PATH)
recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)

# ------------------------------------------------------------
# 2. Set up GPT config. Swap this for local LLM later
# ------------------------------------------------------------

openAI_config = [
    {
        "model": "gpt-4",  
        "api_key": "sk-proj-XXXX",  
        "base_url": "https://api.openai.com/v1/"
    }
]


gpt_con = {"config_list": openAI_config}

agent = ConversableAgent(
    name="chatbot",
    llm_config=gpt_con,  
    human_input_mode="NEVER"
)

# ------------------------------------------------------------
# 3. Producer thread: Continuously capture & transcribe audio
# ------------------------------------------------------------
def audio_capture(recognized_text_queue, stop_event):
    """
    Continuously listens on the mic, sends recognized text to the queue
    only if it has at least 3 words.
    """
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        
        # Put raw audio data in the recognizer
        if recognizer.AcceptWaveform(bytes(indata)):
            result = recognizer.Result()
            text = json.loads(result).get("text", "").strip()

            # Create prompt
            prepend = (
                "You are an astromech droid from Star Wars. "
                "You repair starships. Your response to this prompt "
                "should be 5-6 words total. Here is the prompt:  "
            )
            message = prepend + text
            
            if text:
                # Split the text into words
                words = text.split()
                word_count = len(words)
                
                # Check if the recognized text has at least 3 words
                if word_count >= 3:
                    recognized_text_queue.put(message)
                    print(f"Added to queue: '{text}' (Word count: {word_count})")  
                else:
                    print(f"Ignored short text: '{text}' (Word count: {word_count})")  

    print("Listening... Press Ctrl+C, close the window, or press any key to stop.")
    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=8000,
            dtype='int16',
            channels=1,
            callback=audio_callback
        ):
            # Wait until the stop_event is set
            while not stop_event.is_set():
                time.sleep(0.1)  
    except Exception as e:
        print(f"Error in audio_capture: {e}")
    finally:
        print("Audio capture stopped.")

# ------------------------------------------------------------
# 4. Consumer thread: Use recognized text to prompt the LLM
# ------------------------------------------------------------
def llm_responder(recognized_text_queue, ui_queue, audio_queue, stop_event):
    while not stop_event.is_set():
        try:
            text = recognized_text_queue.get(timeout=0.5)

           
            reply = agent.generate_reply(messages=[{"content": text, "role": "user"}])
            ui_queue.put(reply)

            # Encode the TTS data
            encoded_data = encoder.encode_sentence(reply)

            duration_seconds = (len(encoded_data)*.5) / float(SAMPLE_RATE)
            audio_queue.put("playing")

            # Start audio playback
            encoder.play_audio(encoded_data)

            time.sleep(duration_seconds)
            audio_queue.put("finished")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in llm_responder: {e}")


# ------------------------------------------------------------
# 5. Text Wrapping Function
# ------------------------------------------------------------
def wrap_text(text, font, max_width):
    """
    Wraps text to fit within a specified width when rendered with the given font.
    
    :param text: The text string to wrap.
    :param font: The Pygame font object used to render the text.
    :param max_width: The maximum width in pixels for each line.
    :return: A list of wrapped lines.
    """
    lines = []
    words = text.split(' ')
    current_line = ""
    
    for word in words:
        test_line = current_line + word + " "
        line_width, _ = font.size(test_line)
        if line_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    
    if current_line:
        lines.append(current_line.strip())
    
    return lines

# ------------------------------------------------------------
# 6. Main function
# ------------------------------------------------------------
def main():
    recognized_text_queue = queue.Queue()
    ui_queue = queue.Queue()       
    audio_queue = queue.Queue()    
    stop_event = threading.Event()

    # Start audio capture in one thread
    capture_thread = threading.Thread(
        target=audio_capture,
        args=(recognized_text_queue, stop_event),
        daemon=True 
    )
    capture_thread.start()

    # Start LLM responder in another thread
    responder_thread = threading.Thread(
        target=llm_responder,
        args=(recognized_text_queue, ui_queue, audio_queue, stop_event),
        daemon=True 
    )
    responder_thread.start()

    # -------------- Setup Pygame Window --------------
    pygame.init()
    screen = pygame.display.set_mode((800, 400))
    pygame.display.set_caption("LD-1")

    # requires aurebesh font is installed on RBPi
    aurebesh_font_path = "/usr/share/fonts/truetype/custom/Aurebesh.otf"
    try:
        aurebesh_font = pygame.font.Font(aurebesh_font_path, 32)
        print("Aurebesh font loaded successfully.")
    except:
        aurebesh_font = pygame.font.SysFont(None, 32)  
        print("Aurebesh font not found. Using default font.")

    arial_font_path = "/usr/share/fonts/truetype/custom/Arial.ttf"
    try:
        arial_font = pygame.font.Font(arial_font_path, 32)
        print("Arial font loaded successfully.")
    except:
        arial_font = pygame.font.SysFont(None, 32)  
        print("Arial font not found. Using default font.")

    # Define colors
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLACK = (0, 0, 0)

    # Define maximum width for text rendering (e.g., window width minus margins)
    MAX_TEXT_WIDTH = 700 

    # Initialize display variables
    full_text = ""
    current_text = ""
    letter_index = 0
    phase = 'idle'  
    last_letter_time = 0
    display_complete_time = 0

    # Initialize wrapped lines
    wrapped_lines = []

    clock = pygame.time.Clock()  # To control the frame rate

    try:
        while True:
            current_time = pygame.time.get_ticks()  # Current time in milliseconds

            # Process any Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Quit event detected. Stopping threads...")
                    stop_event.set()  # Signal all threads to stop
                    pygame.quit()
                    sys.exit(0)
                elif event.type == pygame.KEYDOWN:
                    print("Key press detected. Stopping threads...")
                    stop_event.set()  # Signal all threads to stop
                    pygame.quit()
                    sys.exit(0)

            # Check for new LLM replies
            try:
                while True:  # Drain the queue
                    new_reply = ui_queue.get_nowait()
                    full_text = new_reply
                    current_text = ""
                    letter_index = 0
                    phase = 'typing'
                    last_letter_time = current_time
                    wrapped_lines = []  # Reset wrapped lines
                    print(f"New reply received: '{full_text}'")
            except queue.Empty:
                pass

            # Check for audio status updates
            try:
                while True:  # Drain the audio queue
                    audio_status = audio_queue.get_nowait()
                    if audio_status == "playing":
                        print("Audio is playing...")
                     
                    elif audio_status == "finished" and phase == 'waiting':
                        phase = 'final'
                        print("Finished, switching to final phase (green font).")
            except queue.Empty:
                pass

            # Handle typing effect
            if phase == 'typing':
                if current_time - last_letter_time >= 100:  # 0.1 seconds per letter
                    if letter_index < len(full_text):
                        current_text += full_text[letter_index]
                        letter_index += 1
                        last_letter_time = current_time
                        
                        wrapped_lines = wrap_text(current_text, aurebesh_font, MAX_TEXT_WIDTH)

                    if letter_index >= len(full_text):
                        phase = 'waiting'
                        display_complete_time = current_time
                        print("Typing complete. Waiting for audio to finish.")
            elif phase == 'waiting':
                pass

            # Fill screen with black
            screen.fill(BLACK)

            # Render the text based on the current phase
            if phase == 'typing':
                # Render each wrapped line in red, Aurebesh
                y_offset = 50  
                for line in wrapped_lines:
                    text_surface = aurebesh_font.render(line, True, RED)
                    screen.blit(text_surface, (50, y_offset))
                    y_offset += aurebesh_font.get_height() + 5
            elif phase == 'final':
                final_wrapped = wrap_text(full_text, arial_font, MAX_TEXT_WIDTH)
                y_offset = 50
                for line in final_wrapped:
                    text_surface = arial_font.render(line, True, GREEN)
                    screen.blit(text_surface, (50, y_offset))
                    y_offset += arial_font.get_height() + 5

            # Flip/update the display
            pygame.display.flip()

            # Control the frame rate
            clock.tick(30)  # 30 FPS

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Stopping threads...")
        stop_event.set()
        pygame.quit()
        sys.exit(0)

if __name__ == "__main__":
    main()