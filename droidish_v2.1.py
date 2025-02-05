import numpy as np
from scipy.signal import find_peaks
import random
import sounddevice as sd
import soundfile as sf
import os

# Suggested constants
MIN_INDEX_DIFFERENCE = 3
NUM_FREQUENCIES = 50
MIN_FREQ = 500
MAX_FREQ = 5000
NUM_WORDS_NEEDED = 2000  
SAMPLE_RATE = 44100
FADE_DURATION = 0.01
VOLUME = 0.7
WHISTLE_THRESHOLD = 0.6
CHIRP_THRESHOLD = 0.6
SEED = 73
TONE_LENGTH = 0.1      # Duration for all non whitle tones
WHISTLE_LENGTH = .7    # Duration for whistles

def load_common_words(filename="2000_words.txt"):
    """
    Load words for dictionary from a text file.
    """
    # Get the directory where the current script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    with open(file_path, "r") as f:
        # Read each line, strip whitespace, and ignore empty lines.
        words = [line.strip() for line in f if line.strip()]
    return words


def generate_tone_dictionaries(seed, num_words_needed, num_frequencies, min_freq, max_freq, min_index_difference, COMMON_WORDS):
    """Generates tone dictionaries for common words and characters."""
    CHARACTERS = list('abcdefghijklmnopqrstuvwxyz0123456789 .,!?\'')
    if ' ' not in CHARACTERS:
        CHARACTERS.append(' ')

    random.seed(seed)

    # Generate frequencies
    frequencies = np.linspace(min_freq, max_freq, num_frequencies)

    # Create character dictionary
    character_dict = {char: freq for char, freq in zip(CHARACTERS, frequencies[:len(CHARACTERS)])}

    # Generate word pairs
    word_pairs = []
    for i in range(len(frequencies)):
        for j in range(i + min_index_difference, len(frequencies)):
            pair = (frequencies[i], frequencies[j])
            word_pairs.append(pair)
            word_pairs.append(pair[::-1])  # Reverse the pair

    # Shuffle and truncate word pairs
    random.shuffle(word_pairs)
    word_pairs = word_pairs[:num_words_needed]

    # Create common word dictionary
    common_words_dict = {word: pair for word, pair in zip(COMMON_WORDS[:num_words_needed], word_pairs)}

    return common_words_dict, character_dict

# --- Encoder ---

class Encoder:
    def __init__(self, common_words_dict, character_dict, tone_length, whistle_length, sample_rate=44100, fade_duration=0.01, volume=0.7):
        self.common_words_dict = common_words_dict
        self.character_dict = character_dict
        self.tone_length = tone_length      # For non-whistle tones
        self.whistle_length = whistle_length  # For whistles
        self.sample_rate = sample_rate
        self.fade_duration = fade_duration
        self.volume = volume

        # Generate the whistle using WHISTLE_LENGTH (no padding)
        self.whistle_sound = self.generate_whistle(duration=self.whistle_length, volume=self.volume)
        # Chirp and sine tones remain generated with TONE_LENGTH.
        self.chirp_sound = self.generate_chirp(duration=self.tone_length, volume=self.volume)

    def generate_sine_wave(self, frequency):
        N = int(self.sample_rate * self.tone_length)
        t = np.linspace(0, self.tone_length, N, False)
        tone = np.sin(2 * np.pi * frequency * t)
        fade_in_samples = int(self.sample_rate * self.fade_duration)
        fade_out_samples = int(self.sample_rate * self.fade_duration)
        envelope = np.ones_like(tone)
        envelope[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)
        envelope[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)
        tone *= envelope
        return tone.astype(np.float32)

    def generate_whistle(self, duration, start_freq=500, end_freq=1500, volume=0.7):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        frequencies = np.linspace(start_freq, end_freq, t.size)
        audio = np.sin(2 * np.pi * frequencies * t) * volume
        return audio.astype(np.float32)

    def generate_chirp(self, duration, center_freq=1000, mod_rate=10, volume=0.7):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        modulator = np.sin(2 * np.pi * mod_rate * t) * (center_freq * 0.5)
        carrier = np.sin(2 * np.pi * (center_freq + modulator) * t)
        audio = carrier * volume
        return audio.astype(np.float32)

    def encode_sentence(self, sentence):
        """
        Encode a sentence into an audio sequence.
        
        In system1 (dictionary mode), space tokens are ignored,
        so that the decoder’s segmentation remains aligned. In system2,
        spaces are still encoded.
        """
        # Split the sentence while preserving spaces.
        tokens = []
        words = sentence.lower().split(' ')
        for i, word in enumerate(words):
            tokens.append(word)
            if i < len(words) - 1:
                tokens.append(" ")

        audio_sequence = np.array([], dtype=np.float32)
        current_system = None  # 'system1' for dictionary words, 'system2' for spelled letters

        for token in tokens:
            if token == " ":
                # For dictionary words (system1), ignore the space;
                # the decoder will automatically add a space after decoding each word.
                if current_system == 'system1':
                    continue
                else:
                    # In system2, encode the space as before.
                    if ' ' in self.character_dict:
                        freq = self.character_dict[' ']
                        tone = self.generate_sine_wave(freq) * self.volume
                        audio_sequence = np.concatenate((audio_sequence, tone))
                    else:
                        silence = np.zeros(int(self.sample_rate * self.tone_length), dtype=np.float32)
                        audio_sequence = np.concatenate((audio_sequence, silence))
                continue

            # For non-space tokens:
            if token in self.common_words_dict:
                # Dictionary word → System 1.
                if current_system != 'system1':
                    # Switch to system1 by playing a whistle (as before)
                    whistle_count = random.randint(1, 3)
                    for _ in range(whistle_count):
                        audio_sequence = np.concatenate((audio_sequence, self.whistle_sound))
                    current_system = 'system1'
                # Encode the dictionary word using its two-tone pair.
                f1, f2 = self.common_words_dict[token]
                tone1 = self.generate_sine_wave(f1) * self.volume
                tone2 = self.generate_sine_wave(f2) * self.volume
                audio_sequence = np.concatenate((audio_sequence, tone1, tone2))
            else:
                # Non-dictionary word → System 2.
                if current_system != 'system2':
                    audio_sequence = np.concatenate((audio_sequence, self.chirp_sound))
                    current_system = 'system2'
                for char in token:
                    if char in self.character_dict:
                        freq = self.character_dict[char]
                        tone = self.generate_sine_wave(freq) * self.volume
                        audio_sequence = np.concatenate((audio_sequence, tone))

        # Normalize the final audio sequence.
        max_val = np.max(np.abs(audio_sequence))
        if max_val > 0:
            audio_sequence = audio_sequence / max_val

        return audio_sequence


    def play_audio(self, audio):
        sd.play(audio, self.sample_rate)
        sd.wait()

    def export_audio(self, audio, filename="output.wav"):
        sf.write(filename, audio, self.sample_rate)
        print(f"Audio exported to {filename}")


# --- Decoder ---

class Decoder:
    def __init__(self, 
                 sample_rate=44100, 
                 tone_length=0.125, 
                 whistle_length=1.0, 
                 WHISTLE_THRESHOLD=0.6, 
                 CHIRP_THRESHOLD=0.6, 
                 common_words_dict=None, 
                 character_dict=None):
        self.sample_rate = sample_rate
        self.tone_length = tone_length      # Expected length for non-whistle tones.
        self.whistle_length = whistle_length  # Expected length for whistles.
        self.WHISTLE_THRESHOLD = WHISTLE_THRESHOLD
        self.CHIRP_THRESHOLD = CHIRP_THRESHOLD
        self.common_words_dict = common_words_dict if common_words_dict is not None else {}
        self.character_dict = character_dict if character_dict is not None else {}

        # Generate reference signals without padding.
        self.whistle_ref = self.generate_whistle(duration=self.whistle_length, 
                                                  start_freq=500, 
                                                  end_freq=1500, 
                                                  sample_rate=self.sample_rate, 
                                                  volume=0.7)
        self.chirp_ref = self.generate_chirp(duration=self.tone_length, 
                                             center_freq=1000, 
                                             mod_rate=10, 
                                             sample_rate=self.sample_rate, 
                                             volume=0.7)

    def generate_whistle(self, duration, start_freq=500, end_freq=1500, sample_rate=44100, volume=0.7):
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        frequencies = np.linspace(start_freq, end_freq, t.size)
        audio = np.sin(2 * np.pi * frequencies * t) * volume
        return audio.astype(np.float32)

    def generate_chirp(self, duration, center_freq=1000, mod_rate=10, sample_rate=44100, volume=0.7):
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        modulator = np.sin(2 * np.pi * mod_rate * t) * (center_freq * 0.5)
        carrier = np.sin(2 * np.pi * (center_freq + modulator) * t)
        audio = carrier * volume
        return audio.astype(np.float32)

    def correlate_signals(self, sig1, sig2):
        # Compute normalized dot product.
        length = min(len(sig1), len(sig2))
        sig1 = sig1[:length]
        sig2 = sig2[:length]
        sig1_norm = sig1/(np.linalg.norm(sig1)+1e-10)
        sig2_norm = sig2/(np.linalg.norm(sig2)+1e-10)
        return np.dot(sig1_norm, sig2_norm)

    def detect_whistle(self, segment):
        corr = self.correlate_signals(segment, self.whistle_ref)
        return corr > self.WHISTLE_THRESHOLD

    def detect_chirp(self, segment):
        corr = self.correlate_signals(segment, self.chirp_ref)
        return corr > self.CHIRP_THRESHOLD

    def analyze_tone(self, segment):
        window = np.hanning(len(segment))
        segment = segment * window
        fft_spectrum = np.fft.rfft(segment)
        freq = np.fft.rfftfreq(len(segment), d=1/self.sample_rate)
        magnitude = np.abs(fft_spectrum)
        peaks, _ = find_peaks(magnitude, height=np.max(magnitude)*0.3)
        if len(peaks) == 0:
            return []
        peak_freqs = freq[peaks]
        peak_magnitudes = magnitude[peaks]
        sorted_indices = np.argsort(peak_magnitudes)[::-1]
        dominant_freqs = peak_freqs[sorted_indices]
        return dominant_freqs

    def match_word_pair(self, f1_detected, f2_detected, tolerance_ratio=0.02):
        for word, (f1, f2) in self.common_words_dict.items():
            if abs(f1_detected - f1) <= f1 * tolerance_ratio and abs(f2_detected - f2) <= f2 * tolerance_ratio:
                return word
        return None

    def match_single_tone(self, dominant_freq, tolerance_ratio=0.02):
        best_char = None
        best_diff = float('inf')
        for char, f in self.character_dict.items():
            diff = abs(dominant_freq - f)
            if diff <= f * tolerance_ratio and diff < best_diff:
                best_diff = diff
                best_char = char
        return best_char

    def decode_audio_sequence(self, audio_sequence):
        """
        Walk through the audio stream using a pointer. When a whistle is expected,
        use a window of WHISTLE_LENGTH. For all other tones, use a window of TONE_LENGTH.
        """
        i = 0
        decoded_sentence = ''
        current_system = None  # 'system1' or 'system2'
        spelled_word_chars = ''
        whistle_samples = int(self.sample_rate * self.whistle_length)
        tone_samples = int(self.sample_rate * self.tone_length)

        while i < len(audio_sequence):
            # First, check if a whistle occurs here.
            if i + whistle_samples <= len(audio_sequence):
                seg_whistle = audio_sequence[i:i+whistle_samples]
                if self.detect_whistle(seg_whistle):
                    # System switch to system1.
                    if current_system == 'system2' and spelled_word_chars:
                        decoded_sentence += spelled_word_chars + ' '
                        spelled_word_chars = ''
                    current_system = 'system1'
                    i += whistle_samples
                    continue

            # Check for a chirp signal (system2 marker) using a tone_samples window.
            if i + tone_samples <= len(audio_sequence):
                seg_chirp = audio_sequence[i:i+tone_samples]
                if self.detect_chirp(seg_chirp):
                    if current_system == 'system1' and not decoded_sentence.endswith(' '):
                        decoded_sentence += ' '
                    if current_system == 'system2' and spelled_word_chars:
                        decoded_sentence += spelled_word_chars + ' '
                        spelled_word_chars = ''
                    current_system = 'system2'
                    i += tone_samples
                    continue

            # If no marker was detected, decode based on the current system.
            if current_system is None:
                # No system set yet: mark as unknown.
                decoded_sentence += '?'
                i += tone_samples
                continue

            if current_system == 'system1':
                # Expect two consecutive tones (each tone_samples long) for dictionary words.
                if i + 2 * tone_samples <= len(audio_sequence):
                    seg1 = audio_sequence[i:i+tone_samples]
                    seg2 = audio_sequence[i+tone_samples:i+2*tone_samples]
                    freqs1 = self.analyze_tone(seg1)
                    freqs2 = self.analyze_tone(seg2)
                    # Check if the arrays are empty.
                    if len(freqs1) == 0 or len(freqs2) == 0:
                        decoded_sentence += '? '
                    else:
                        word = self.match_word_pair(freqs1[0], freqs2[0])
                        decoded_sentence += (word + ' ') if word else '? '
                    i += 2 * tone_samples
                else:
                    break

            elif current_system == 'system2':
                # Decode a single tone (tone_samples long) for a character.
                if i + tone_samples <= len(audio_sequence):
                    seg = audio_sequence[i:i+tone_samples]
                    freqs = self.analyze_tone(seg)
                    if not freqs:
                        spelled_word_chars += '?'
                    else:
                        char = self.match_single_tone(freqs[0])
                        if char:
                            if char == ' ':
                                if spelled_word_chars:
                                    decoded_sentence += spelled_word_chars + ' '
                                    spelled_word_chars = ''
                                else:
                                    if not decoded_sentence.endswith(' '):
                                        decoded_sentence += ' '
                            else:
                                spelled_word_chars += char
                        else:
                            spelled_word_chars += '?'
                    i += tone_samples
                else:
                    break

        # Flush any remaining spelled characters.
        if current_system == 'system2' and spelled_word_chars:
            decoded_sentence += spelled_word_chars + ' '
        return decoded_sentence.strip()
    
###############################################################################
############################### EXAMPLE USAGE #################################
###############################################################################

ONE_LENGTH = 0.1   
WHISTLE_LENGTH = .5     
NUM_WORDS_NEEDED = 2000

if __name__ == '__main__':

    COMMON_WORDS = load_common_words("2000_words.txt")

    common_words_dict, character_dict = generate_tone_dictionaries(
        SEED, NUM_WORDS_NEEDED, NUM_FREQUENCIES, MIN_FREQ, MAX_FREQ, MIN_INDEX_DIFFERENCE, COMMON_WORDS
    )

    encoder = Encoder(common_words_dict, character_dict,
                      tone_length=TONE_LENGTH,
                      whistle_length=WHISTLE_LENGTH,
                      sample_rate=SAMPLE_RATE,
                      fade_duration=FADE_DURATION,
                      volume=VOLUME)
    decoder = Decoder(sample_rate=SAMPLE_RATE,
                      tone_length=TONE_LENGTH,
                      whistle_length=WHISTLE_LENGTH,
                      WHISTLE_THRESHOLD=WHISTLE_THRESHOLD,
                      CHIRP_THRESHOLD=CHIRP_THRESHOLD,
                      common_words_dict=common_words_dict,
                      character_dict=character_dict)

    test_sentence = "The odds of survival are 725 to 1"

    print("Encoding sentence:", test_sentence)
    audio = encoder.encode_sentence(test_sentence)
    encoder.play_audio(audio)
    # And decode it:
    decoded = decoder.decode_audio_sequence(audio)
    print("Decoded sentence:", decoded)