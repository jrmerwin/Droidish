import numpy as np
from scipy.signal import find_peaks
import random
import sounddevice as sd

# suggested constants
MIN_INDEX_DIFFERENCE = 3
NUM_FREQUENCIES = 50
MIN_FREQ = 500
MAX_FREQ = 5000
NUM_WORDS_NEEDED = 1000  
SAMPLE_RATE = 44100
FADE_DURATION = 0.01
VOLUME = 0.7
WHISTLE_THRESHOLD = 0.6
CHIRP_THRESHOLD = 0.6
SEED = 73
TONE_LENGTH = 0.125
WHISTLE_LENGTH = 0.125

def generate_tone_dictionaries(seed, num_words_needed, num_frequencies, min_freq, max_freq, min_index_difference):
    """Generates tone dictionaries for common words and characters.
    1000 common words are used here but more can be added for further compression.
    """

    CHARACTERS = list('abcdefghijklmnopqrstuvwxyz0123456789 .,!?\'')
    if ' ' not in CHARACTERS:
        CHARACTERS.append(' ')

    COMMON_WORDS = [
    'a','ability','able','about','above','accept','according','account','across','act',
    'action','activity','actually','add','address','administration','admit','adult','affect',
    'after','again','against','age','agency','agent','ago','agree','agreement','ahead','air',
    'all','allow','almost','alone','along','already','also','although','always','American',
    'among','amount','analysis','and','animal','another','answer','any','anyone','anything',
    'appear','apply','approach','area','argue','arm','around','arrive','art','article','artist',
    'as','ask','assume','at','attack','attention','attorney','audience','author','authority',
    'available','avoid','away','baby','back','bad','bag','ball','bank','bar','base','be','beat'
    ,'beautiful','because','become','bed','before','begin','behavior','behind','believe',
    'benefit','best','better','between','beyond','big','bill','billion','bit','black','blood',
    'blue','board','body','book','born','both','box','boy','break','bring','brother','budget',
    'build','building','business','but','buy','by','call','camera','campaign','can','cancer',
    'candidate','capital','car','card','care','career','carry','case','catch','cause','cell',
    'center','central','century','certain','certainly','chair','challenge','chance','change',
    'character','charge','check','child','choice','choose','church','citizen','city','civil',
    'claim','class','clear','clearly','close','coach','cold','collection','college','color',
    'come','commercial','common','community','company','compare','computer','concern','condition',
    'conference','Congress','consider','consumer','contain','continue','control','cost','could'
    ,'country','couple','course','court','cover','create','crime','cultural','culture','cup',
    'current','customer','cut','dark','data','daughter','day','dead','deal','death','debate',
    'decade','decide','decision','deep','defense','degree','Democrat','democratic','describe'
    ,'design','despite','detail','determine','develop','development','die','difference',
    'different','difficult','dinner','direction','director','discover','discuss','discussion',
    'disease','do','doctor','dog','door','down','draw','dream','drive','drop','drug','during'
    ,'each','early','east','easy','eat','economic','economy','edge','education','effect','effort',
    'eight','either','election','else','employee','end','energy','enjoy','enough','enter',
    'entire','environment','environmental','especially','establish','even','evening','event',
    'ever','every','everybody','everyone','everything','evidence','exactly','example','executive'
    ,'exist','expect','experience','expert','explain','eye','face','fact','factor','fail','fall',
    'family','far','fast','father','fear','federal','feel','feeling','few','field','fight',
    'figure','fill','film','final','finally','financial','find','fine','finger','finish','fire',
    'firm','first','fish','five','floor','fly','focus','follow','food','foot','for','force',
    'foreign','forget','form','former','forward','four','free','friend','from','front','full',
    'fund','future','game','garden','gas','general','generation','get','girl','give','glass',
    'go','goal','good','government','great','green','ground','group','grow','growth','guess',
    'gun','guy','hair','half','hand','hang','happen','happy','hard','have','he','head','health',
    'hear','heart','heat','heavy','help','her','here','herself','high','him','himself','his',
    'history','hit','hold','home','hope','hospital','hot','hotel','hour','house','how','however',
    'huge','human','hundred','husband','I','idea','identify','if','image','imagine','impact',
    'important','improve','in','include','including','increase','indeed','indicate','individual',
    'industry','information','inside','instead','institution','interest','interesting',
    'international','interview','into','investment','involve','issue','it','item','its','itself',
    'job','join','just','keep','key','kid','kill','kind','kitchen','know','knowledge','land',
    'language','large','last','late','later','laugh','law','lawyer','lay','lead','leader','learn',
    'least','leave','left','leg','legal','less','let','letter','level','lie','life','light','like',
    'likely','line','list','listen','little','live','local','long','look','lose','loss','lot',
    'love','low','machine','magazine','main','maintain','major','majority','make','man','manage',
    'management','manager','many','market','marriage','material','matter','may','maybe','me',
    'mean','measure','media','medical','meet','meeting','member','memory','mention','message',
    'method','middle','might','military','million','mind','minute','miss','mission','model',
    'modern','moment','money','month','more','morning','most','mother','mouth','move','movement',
    'movie','Mr','Mrs','much','music','must','my','myself','name','nation','national','natural',
    'nature','near','nearly','necessary','need','network','never','new','news','newspaper','next',
    'nice','night','no','none','nor','north','not','note','nothing','notice','now','number',
    'occur','of','off','offer','office','officer','official','often','oh','oil','ok','old',
    'on','once','one','only','onto','open','operation','opportunity','option','or','order',
    'organization','other','others','our','out','outside','over','own','owner','page','pain',
    'painting','paper','parent','part','participant','particular','particularly','partner',
    'party','pass','past','patient','pattern','pay','peace','people','per','perform','performance',
    'perhaps','period','person','personal','phone','physical','pick','picture','piece','place',
    'plan','plant','play','player','PM','point','police','policy','political','politics','poor',
    'popular','population','position','positive','possible','power','practice','prepare','present',
    'president','pressure','pretty','prevent','price','private','probably','problem','process',
    'produce','product','production','professional','professor','program','project','property',
    'protect','prove','provide','public','pull','purpose','push','put','quality','question',
    'quickly','quite','race','radio','raise','range','rate','rather','reach','read','ready',
    'real','reality','realize','really','reason','receive','recent','recently','recognize',
    'record','red','reduce','reflect','region','relate','relationship','religious','remain',
    'remember','remove','report','represent','Republican','require','research','resource',
    'respond','response','responsibility','rest','result','return','reveal','rich','right',
    'rise','risk','road','rock','role','room','rule','run','safe','same','save','say','scene',
    'school','science','scientist','score','sea','season','seat','second','section','security',
    'see','seek','seem','sell','send','senior','sense','series','serious','serve','service',
    'set','seven','several','sex','sexual','shake','share','she','shoot','short','shot','should',
    'shoulder','show','side','sign','significant','similar','simple','simply','since','sing',
    'single','sister','sit','site','situation','six','size','skill','skin','small','smile','so',
    'social','society','soldier','some','somebody','someone','something','sometimes','son','song',
    'soon','sort','sound','source','south','southern','space','speak','special','specific',
    'speech','spend','sport','spring','staff','stage','stand','standard','star','start','state',
    'statement','station','stay','step','still','stock','stop','store','story','strategy',
    'street','strong','structure','student','study','stuff','style','subject','success',
    'successful','such','suddenly','suffer','suggest','summer','support','sure','surface',
    'system','table','take','talk','task','tax','teach','teacher','team','technology',
    'television','tell','ten','tend','term','test','than','thank','that','the','their','them',
    'themselves','then','theory','there','these','they','thing','think','third','this','those',
    'though','thought','thousand','threat','three','through','throughout','throw','thus','time',
    'to','today','together','tonight','too','top','total','tough','toward','town','trade',
    'traditional','training','travel','treat','treatment','tree','trial','trip','trouble',
    'TRUE','truth','try','turn','TV','two','type','under','understand','unit','until','up',
    'upon','us','use','usually','value','various','very','victim','view','violence','visit',
    'voice','vote','wait','walk','wall','want','war','watch','water','way','we','weapon','wear',
    'week','weight','well','west','western','what','whatever','when','where','whether','which',
    'while','white','who','whole','whom','whose','why','wide','wife','will','win','wind',
    'window','wish','with','within','without','woman','wonder','word','work','worker','world',
    'worry','would','write','writer','wrong','yard','yeah','year','yes','yet','you','young',
    'your','yourself'
    ]

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

# encoder class
class Encoder:
    def __init__(self, common_words_dict, character_dict, tone_length, sample_rate=44100, fade_duration=0.01, volume=0.7):
        self.common_words_dict = common_words_dict
        self.character_dict = character_dict
        self.tone_length = tone_length
        self.sample_rate = sample_rate
        self.fade_duration = fade_duration
        self.volume = volume

        # Generate reference signals similar to the function version
        self.whistle_sound = self.generate_whistle(duration=self.tone_length, volume=self.volume)
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
        
        words = []
        for i, word in enumerate(sentence.lower().split(' ')):
            words.append(word)
            if i < len(sentence.lower().split(' ')) - 1:  # Add a space after each word except the last one
                words.append(' ')

        audio_sequence = np.array([], dtype=np.float32)
        current_system = None  # 'system1' or 'system2'

        for i, word in enumerate(words):
            if word in self.common_words_dict:
                # Dictionary word => System 1
                if current_system != 'system1':
                    # Play whistle 1 to 3 times
                    whistle_count = random.randint(1, 3)
                    for _ in range(whistle_count):
                        audio_sequence = np.concatenate((audio_sequence, self.whistle_sound))
                    current_system = 'system1'

                f1, f2 = self.common_words_dict[word]
                tone1 = self.generate_sine_wave(f1) * self.volume
                tone2 = self.generate_sine_wave(f2) * self.volume
                audio_sequence = np.concatenate((audio_sequence, tone1, tone2))
            else:
                # Non-dictionary word => System 2
                if current_system != 'system2':
                    audio_sequence = np.concatenate((audio_sequence, self.chirp_sound))
                    current_system = 'system2'

                for char in word:
                    if char in self.character_dict:
                        freq = self.character_dict[char]
                        tone = self.generate_sine_wave(freq) * self.volume
                        audio_sequence = np.concatenate((audio_sequence, tone))

        # Normalize the final audio sequence
        max_val = np.max(np.abs(audio_sequence))
        if max_val > 0:
            audio_sequence = audio_sequence / max_val

        return audio_sequence

    def play_audio(self, audio):
        sd.play(audio, self.sample_rate)
        sd.wait()


# decoder class
class Decoder:
    def __init__(self, 
                 sample_rate=44100, 
                 tone_length=0.1, 
                 whistle_length=0.1, 
                 WHISTLE_THRESHOLD=0.6, 
                 CHIRP_THRESHOLD=0.6, 
                 common_words_dict=None, 
                 character_dict=None):
        
        self.sample_rate = sample_rate
        self.tone_length = tone_length
        self.whistle_length = whistle_length
        self.WHISTLE_THRESHOLD = WHISTLE_THRESHOLD
        self.CHIRP_THRESHOLD = CHIRP_THRESHOLD
        self.common_words_dict = common_words_dict if common_words_dict is not None else {}
        self.character_dict = character_dict if character_dict is not None else {}

        # Pre-generate reference tones
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

    def pad_audio_sequence(self, audio_sequence, segment_length):
        total_length = len(audio_sequence)
        remainder = total_length % segment_length
        if remainder != 0:
            padding = np.zeros(segment_length - remainder, dtype=np.float32)
            audio_sequence = np.concatenate((audio_sequence, padding))
        return audio_sequence

    def segment_audio_sequence(self, audio_sequence, sample_rate, tone_length):
        segment_length = int(sample_rate * tone_length)
        total_length = len(audio_sequence)
        segments = [audio_sequence[i:i+segment_length] for i in range(0, total_length, segment_length)]
        return segments

    def analyze_tone(self, segment, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        window = np.hanning(len(segment))
        segment = segment * window
        fft_spectrum = np.fft.rfft(segment)
        freq = np.fft.rfftfreq(len(segment), d=1/sample_rate)
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
            t1 = f1 * tolerance_ratio
            t2 = f2 * tolerance_ratio
            if abs(f1_detected - f1) <= t1 and abs(f2_detected - f2) <= t2:
                return word
        return None

    def match_single_tone(self, dominant_freq, tolerance_ratio=0.02):
        best_char = None
        best_diff = float('inf')
        for char, f in self.character_dict.items():
            t = f * tolerance_ratio
            diff = abs(dominant_freq - f)
            if diff <= t and diff < best_diff:
                best_diff = diff
                best_char = char
        return best_char

    def decode_audio_sequence(self, audio_sequence):
        segment_length = int(self.sample_rate * self.tone_length)
        audio_sequence = self.pad_audio_sequence(audio_sequence, segment_length)
        segments = self.segment_audio_sequence(audio_sequence, self.sample_rate, self.tone_length)
        decoded_sentence = ''
        current_system = None  # 'system1' or 'system2'

        spelled_word_chars = ''  # buffer for system2 spelled words
        i = 0
        while i < len(segments):
            segment = segments[i]

            # Check for system switches
            if self.detect_whistle(segment):
                # Switch to System 1
                if current_system == 'system2' and spelled_word_chars:
                    # Flush spelled_word_chars and add a space unconditionally
                    decoded_sentence += spelled_word_chars
                    spelled_word_chars = ''
                    if not decoded_sentence.endswith(' '):
                        decoded_sentence += ' '
                current_system = 'system1'
                # Skip all consecutive whistles
                while i < len(segments) and self.detect_whistle(segments[i]):
                    i += 1
                continue

            if self.detect_chirp(segment):
                # Switch to System 2
                if current_system == 'system1' and not decoded_sentence.endswith(' '):
                    decoded_sentence += ' '
                if current_system == 'system2' and spelled_word_chars:
                    # Flush spelled_word_chars and add a space unconditionally
                    decoded_sentence += spelled_word_chars
                    spelled_word_chars = ''
                    if not decoded_sentence.endswith(' '):
                        decoded_sentence += ' '
                current_system = 'system2'
                # Skip all consecutive chirps
                while i < len(segments) and self.detect_chirp(segments[i]):
                    i += 1
                continue

            if current_system is None:
                # No system set yet - unknown
                decoded_sentence += '?'
                i += 1
                continue

            if current_system == 'system1':
                # System 1 uses pairs of tones for words
                if i + 1 >= len(segments):
                    # Not enough segments for a full word
                    decoded_sentence += '?'
                    break

                seg1 = segments[i]
                seg2 = segments[i + 1]
                freqs1 = self.analyze_tone(seg1, self.sample_rate)
                freqs2 = self.analyze_tone(seg2, self.sample_rate)

                if len(freqs1) < 1 or len(freqs2) < 1:
                    decoded_sentence += '? '
                    i += 2
                    continue

                f1_detected = freqs1[0]
                f2_detected = freqs2[0]
                word = self.match_word_pair(f1_detected, f2_detected)
                if word:
                    # Always add a space after a recognized dictionary word
                    decoded_sentence += word + ' '
                else:
                    decoded_sentence += '? '
                i += 2

            elif current_system == 'system2':
                # System 2 uses single tones for characters
                freqs = self.analyze_tone(segment, self.sample_rate)
                if len(freqs) < 1:
                    spelled_word_chars += '?'
                else:
                    char = self.match_single_tone(freqs[0])
                    if char:
                        if char == ' ':
                            # Space signals end of spelled word
                            if spelled_word_chars:
                                # Flush spelled_word_chars and add space
                                decoded_sentence += spelled_word_chars + ' '
                                spelled_word_chars = ''
                            else:
                                # If there were no spelled chars yet, ensure at least one space
                                if not decoded_sentence.endswith(' '):
                                    decoded_sentence += ' '
                        else:
                            spelled_word_chars += char
                    else:
                        spelled_word_chars += '?'
                i += 1

        # Flush remaining spelled characters if in System 2
        if current_system == 'system2' and spelled_word_chars:
            decoded_sentence += spelled_word_chars
            if not decoded_sentence.endswith(' '):
                decoded_sentence += ' '

        return decoded_sentence.strip()
    
# # example usage:

# from droid_speak import (
#     generate_tone_dictionaries,
#     Encoder,
#     Decoder,
#     MIN_INDEX_DIFFERENCE,
#     NUM_FREQUENCIES,
#     MIN_FREQ,
#     MAX_FREQ,
#     NUM_WORDS_NEEDED ,
#     SAMPLE_RATE,
#     FADE_DURATION,
#     VOLUME,
#     WHISTLE_THRESHOLD,
#     CHIRP_THRESHOLD,
#     SEED,
#     TONE_LENGTH,
#     WHISTLE_LENGTH
# )

# # generate dictionaries
# common_words_dict, character_dict = generate_tone_dictionaries(SEED, NUM_WORDS_NEEDED, NUM_FREQUENCIES, MIN_FREQ, MAX_FREQ, MIN_INDEX_DIFFERENCE)

# # Create an encoder instance
# encoder = Encoder(
#     common_words_dict, 
#     character_dict, 
#     tone_length=TONE_LENGTH, 
#     sample_rate=44100, 
#     fade_duration=0.01, 
#     volume=0.7)

# # Encode a sentence
# audio_sequence = encoder.encode_sentence("Hello World!!!")

# # Play the encoded audio
# encoder.play_audio(audio_sequence)

# # Create a decoder instance
# decoder = Decoder(
#     sample_rate=44100,
#     tone_length=TONE_LENGTH,
#     whistle_length=WHISTLE_LENGTH,
#     WHISTLE_THRESHOLD=WHISTLE_THRESHOLD,
#     CHIRP_THRESHOLD=CHIRP_THRESHOLD,
#     common_words_dict=common_words_dict,
#     character_dict=character_dict
# )

# # Decode the audio sequence
# decoded_text = decoder.decode_audio_sequence(audio_sequence)

# print("Decoded text:", decoded_text)
