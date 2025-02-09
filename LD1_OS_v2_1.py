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
import string

# set up the GPIO state indicator lights
import subprocess

GPIOCHIP = "gpiochip0"
LED1_PIN = 17
LED2_PIN = 18

# Track which light is currently ON (0 = LED1, 1 = LED2)
current_state = 0  

def switch_lights():
    """Toggle between LED1 and LED2, keeping one ON and the other OFF."""
    global current_state
    if current_state == 0:
        # Switch to LED2 ON, LED1 OFF
        subprocess.Popen(["gpioset", GPIOCHIP, f"{LED1_PIN}=0", f"{LED2_PIN}=1"])
        current_state = 1
    else:
        # Switch to LED1 ON, LED2 OFF
        subprocess.Popen(["gpioset", GPIOCHIP, f"{LED1_PIN}=1", f"{LED2_PIN}=0"])
        current_state = 0

def lights_off():
    """Turn both LEDs OFF."""
    subprocess.Popen(["gpioset", GPIOCHIP, f"{LED1_PIN}=0", f"{LED2_PIN}=0"])

# ----------------------------------------------------------------
# 0. Instantiate Droidish Encoder
# ----------------------------------------------------------------
import soundfile
from scipy.signal import find_peaks
import droidish_v2_1 as dr

from droidish_v2_1 import (
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

ONE_LENGTH = 0.1   
WHISTLE_LENGTH = .6     
NUM_WORDS_NEEDED = 2000

COMMON_WORDS = dr.load_common_words("2000_words.txt")

common_words_dict, character_dict = generate_tone_dictionaries(
    SEED, NUM_WORDS_NEEDED, NUM_FREQUENCIES, MIN_FREQ, MAX_FREQ, MIN_INDEX_DIFFERENCE, COMMON_WORDS
)

# Create an encoder instance
encoder = Encoder(common_words_dict, character_dict,
                    tone_length=TONE_LENGTH,
                    whistle_length=WHISTLE_LENGTH,
                    sample_rate=SAMPLE_RATE,
                    fade_duration=FADE_DURATION,
                    volume=VOLUME)


# ------------------------------------------------------------
# 1. Set up your Vosk model
# ------------------------------------------------------------
MODEL_PATH = "/home/lb1/droid/droids/vosk-model"
SAMPLE_RATE = 44100

vosk.SetLogLevel(-1)  # Optional: reduce log spam
model = vosk.Model(MODEL_PATH)
recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)

# ------------------------------------------------------------
# 2. Set up your LLM configuration
# ------------------------------------------------------------

openAI_config = [
    {
        "model": "gpt-4",  # Replace with your agent's model name if different
        "api_key": "sk-proj-XXXX",  # Replace with your actual API key
        "base_url": "https://api.openai.com/v1/"
    }
]

# Choose which LLM to use
gpt_con = {"config_list": openAI_config}

agent = ConversableAgent(
    name="chatbot",
    llm_config=gpt_con,  # or use your ChatGPT config instead
    human_input_mode="NEVER"
)

# ------------------------------------------------------------
# 3. Producer thread: Continuously capture & transcribe audio
# ------------------------------------------------------------

# Define the desired microphone device index
DESIRED_MIC_INDEX = 0  # Corresponds to hw:2,0 (USB PnP Sound Device)

def audio_capture(recognized_text_queue, stop_event):
    """
    Continuously listens on the specified microphone, sends recognized text to the queue
    only if it has at least 3 words.
    """
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Audio Status: {status}", file=sys.stderr)
        
        # Put raw audio data in the recognizer
        if recognizer.AcceptWaveform(bytes(indata)):
            result = recognizer.Result()
            text = json.loads(result).get("text", "").strip()

            if text:
                # Split the text into words
                words = text.split()
                word_count = len(words)
                
                # Check if the recognized text has at least 3 words
                if word_count >= 3:
                    recognized_text_queue.put(text)  # Enqueue only the user input
                    print(f"DEBUG: Added to queue: '{text}' (Word count: {word_count})")  # Debug print
                else:
                    print(f"DEBUG: Ignored short text: '{text}' (Word count: {word_count})")  # Debug print

    print("Listening... Press Ctrl+C, close the window, or press any key to stop.")
    switch_lights()
    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=8000,
            dtype='int16',
            channels=1,  # Mono recording
            device=DESIRED_MIC_INDEX,  # Specify the desired microphone
            callback=audio_callback
        ):
            # Wait until the stop_event is set
            while not stop_event.is_set():
                time.sleep(0.1)  # Sleep briefly to allow checking the event
    except Exception as e:
        print(f"Error in audio_capture: {e}")
    finally:
        print("Audio capture stopped.")



# ------------------------------------------------------------
# 4. Consumer thread: Use recognized text to prompt the LLM
# ------------------------------------------------------------
def llm_responder(recognized_text_queue, ui_queue, audio_queue, stop_event):
    while not stop_event.is_set():
        switch_lights()
        try:
            text = recognized_text_queue.get(timeout=0.5)
            # Remove leading/trailing whitespace, convert to lowercase, and remove punctuation
            clean_text = text.strip().lower().translate(str.maketrans('', '', string.punctuation))

            # Debug: Print the exact recognized text after cleaning
            print(f"DEBUG: Recognized text after cleaning: '{clean_text}'")

            # --------------------------
            # 1) Check for shutdown phrase
            # --------------------------
            if clean_text == "shut down now":
                print("DEBUG: Shutdown command detected.")
                goodbye_text = "Good Bye"
                lights_off()
                
                # Send text to UI queue for display
                ui_queue.put(goodbye_text)

                # Encode the TTS data in "Droidish"
                encoded_data = encoder.encode_sentence(goodbye_text)

                # Calculate playback duration
                duration_seconds = (len(encoded_data)*0.5) / float(SAMPLE_RATE)

                # Notify main loop audio is playing
                audio_queue.put("playing")

                # Play audio
                encoder.play_audio(encoded_data)

                # Sleep for the duration
                time.sleep(duration_seconds)

                # Notify main loop audio is finished
                audio_queue.put("finished")

                # Signal to exit
                stop_event.set()

            # --------------------------
            # 2) Check for special welcome token
            # --------------------------
            elif clean_text == "welcome":
                welcome_text = "Hello, my name is LD-1"

                # Send text to UI queue
                ui_queue.put(welcome_text)

                # Encode in "Droidish"
                encoded_data = encoder.encode_sentence(welcome_text)

                # Calculate playback duration
                duration_seconds = (len(encoded_data)*0.5) / float(SAMPLE_RATE)

                # Notify main loop audio is playing
                audio_queue.put("playing")

                # Play audio
                encoder.play_audio(encoded_data)
                time.sleep(duration_seconds)

                # Notify main loop audio is finished
                audio_queue.put("finished")

                # Skip LLM reply, continue listening
                continue

            # --------------------------
            # 3) Normal flow: call LLM
            # --------------------------
            else:
                # Prepend instructions before sending to LLM
                prompt = (
                    "You are an astromech droid from Star Wars. "
                    "You repair starships. Your response to this prompt "
                    "should be 5-6 words total. Here is the prompt: " + text
                )
                
                # Generate LLM reply
                reply = agent.generate_reply(messages=[{"content": prompt, "role": "user"}])
                ui_queue.put(reply)  # send to UI

                # Encode in "Droidish"
                encoded_data = encoder.encode_sentence(reply)

                # Calculate playback duration
                duration_seconds = (len(encoded_data)*0.5) / float(SAMPLE_RATE)

                # Notify main loop audio is playing
                audio_queue.put("playing")

                # Play audio
                encoder.play_audio(encoded_data)
                time.sleep(duration_seconds)

                # Notify main loop audio is finished
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
    ui_queue = queue.Queue()       # For passing the LLM's reply to the UI
    audio_queue = queue.Queue()    # For receiving audio playback status
    stop_event = threading.Event()

    # Start audio capture in one thread
    capture_thread = threading.Thread(
        target=audio_capture,
        args=(recognized_text_queue, stop_event),
        daemon=True  # Daemonize thread to exit with the main program
    )
    capture_thread.start()

    # Start LLM responder in another thread
    responder_thread = threading.Thread(
        target=llm_responder,
        args=(recognized_text_queue, ui_queue, audio_queue, stop_event),
        daemon=True  # Daemonize thread to exit with the main program
    )
    responder_thread.start()

    # ------------------------------------------------
    # NEW: Send a special signal to produce the welcome
    recognized_text_queue.put("welcome")
    # ------------------------------------------------

    # -------------- Setup Pygame Window --------------
    pygame.init()
    screen = pygame.display.set_mode((800, 400), pygame.FULLSCREEN)
    pygame.display.set_caption("LD-1")

    # Initialize fonts
    aurebesh_font_path = "/usr/share/fonts/truetype/custom/Aurebesh.otf"
    try:
        aurebesh_font = pygame.font.Font(aurebesh_font_path, 40)
        print("Aurebesh font loaded successfully.")
    except:
        aurebesh_font = pygame.font.SysFont(None, 40)
        print("Aurebesh font not found. Using default font.")

    arial_font_path = "/usr/share/fonts/truetype/custom/Arial.ttf"
    try:
        arial_font = pygame.font.Font(arial_font_path, 40)
        print("Arial font loaded successfully.")
    except:
        arial_font = pygame.font.SysFont(None, 40)
        print("Arial font not found. Using default font.")

    # Define colors
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLACK = (0, 0, 0)

    # Define maximum width for text rendering
    MAX_TEXT_WIDTH = 700

    # Initialize display variables
    full_text = ""
    current_text = ""
    letter_index = 0
    phase = 'idle'     # 'idle', 'typing', 'waiting', or 'final'
    last_letter_time = 0
    display_complete_time = 0

    # Initialize wrapped lines
    wrapped_lines = []

    clock = pygame.time.Clock()  # To control the frame rate

    try:
        while True:
            current_time = pygame.time.get_ticks()  # Current time in ms

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
                while True:
                    new_reply = ui_queue.get_nowait()
                    full_text = new_reply
                    current_text = ""
                    letter_index = 0
                    phase = 'typing'
                    last_letter_time = current_time
                    wrapped_lines = []
                    print(f"New reply received: '{full_text}'")
            except queue.Empty:
                pass

            # Check for audio status updates
            try:
                while True:
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
                if current_time - last_letter_time >= 100:  # 0.1 seconds/letter
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
                # We'll switch to 'final' once audio playback is done
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
                # Render the entire text in green, Arial
                final_wrapped = wrap_text(full_text, arial_font, MAX_TEXT_WIDTH)
                y_offset = 50
                for line in final_wrapped:
                    text_surface = arial_font.render(line, True, GREEN)
                    screen.blit(text_surface, (50, y_offset))
                    y_offset += arial_font.get_height() + 5

            # Update the display
            pygame.display.flip()

            # Control the frame rate
            clock.tick(30)  # 30 FPS

            # **NEW: Check if stop_event is set to exit the loop**
            if stop_event.is_set():
                print("Stop event detected. Exiting main loop.")
                break

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Stopping threads...")
        lights_off()
        stop_event.set()
        pygame.quit()
        sys.exit(0)

    # **After exiting the loop, perform cleanup**
    print("Exiting program.")
    lights_off()
    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()