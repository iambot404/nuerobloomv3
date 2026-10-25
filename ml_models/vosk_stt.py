"""
Offline Speech-to-Text (STT) ML Engine for Dyslexia Screening
Uses Vosk Kaldi-based neural acoustic & language model (100% offline, zero internet dependency).
Model: vosk-model-small-en-us-0.15 (~40MB, loads in <1s, <250MB Vercel serverless friendly).
"""

import os
import io
import wave
import json
import re
from typing import Dict, Any, Optional

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check common extracted locations for the model
CANDIDATE_PATHS = [
    os.path.join(BASE_DIR, "vosk-model-small-en-us-0.15"),
    os.path.join(BASE_DIR, "vosk_model"),
    os.path.join(BASE_DIR, "vosk-model")
]

MODEL_PATH = next((p for p in CANDIDATE_PATHS if os.path.isdir(p)), CANDIDATE_PATHS[0])

# Global singleton model
_vosk_model = None


def get_vosk_model():
    """Load and cache the offline Vosk speech recognition model."""
    global _vosk_model
    if _vosk_model is None and VOSK_AVAILABLE:
        if os.path.exists(MODEL_PATH):
            print(f"[*] Initializing offline Vosk STT Model from: {MODEL_PATH}")
            _vosk_model = Model(MODEL_PATH)
        else:
            print(f"[!] Vosk model directory not found at: {MODEL_PATH}")
    return _vosk_model


# Clinical Contrast Phoneme, Spoken Letter Names, and Word Maps for Dyslexia Screening
CLINICAL_MAP = {
    'A': {
        'letter_names': ['a', 'ay', 'aye', 'ey', 'ae', 'eh'],
        'phonemes': ['a', 'ay', 'ah', 'eh', 'ey', 'ae'],
        'word': 'apple',
        'aliases': ['apple', 'apples', 'able', 'ape', 'aple', 'alpha']
    },
    'B': {
        'letter_names': ['b', 'be', 'bee', 'bea'],
        'phonemes': ['b', 'buh', 'bee', 'be'],
        'word': 'bat',
        'aliases': ['bat', 'bad', 'back', 'bag', 'bet', 'bit', 'bot', 'boy']
    },
    'C': {
        'letter_names': ['c', 'see', 'sea', 'si'],
        'phonemes': ['c', 'k', 'kuh', 'see'],
        'word': 'cat',
        'aliases': ['cat', 'cut', 'cap', 'can', 'cot', 'kite', 'kit']
    },
    'D': {
        'letter_names': ['d', 'de', 'dee'],
        'phonemes': ['d', 'duh', 'dee', 'de'],
        'word': 'dog',
        'aliases': ['dog', 'dock', 'dark', 'dig', 'dot', 'dawn']
    },
    'E': {
        'letter_names': ['e', 'ee', 'ea', 'eh', 'eat'],
        'phonemes': ['e', 'ee', 'eh', 'ea'],
        'word': 'egg',
        'aliases': ['egg', 'echo', 'elephant']
    },
    'F': {
        'letter_names': ['f', 'ef', 'eff'],
        'phonemes': ['f', 'ef', 'fuh'],
        'word': 'fish',
        'aliases': ['fish', 'fog', 'fan']
    },
    'G': {
        'letter_names': ['g', 'gee', 'jee', 'je'],
        'phonemes': ['g', 'guh', 'gee'],
        'word': 'go',
        'aliases': ['go', 'gate', 'get', 'game']
    },
    'H': {
        'letter_names': ['h', 'aitch', 'eitch', 'haych', 'ache', 'age', 'each', 'eight', 'itch', 'he'],
        'phonemes': ['h', 'aitch'],
        'word': 'hat',
        'aliases': ['hat', 'hut', 'hot']
    },
    'I': {
        'letter_names': ['i', 'eye', 'aye', 'ai'],
        'phonemes': ['i', 'eye', 'ih'],
        'word': 'ice',
        'aliases': ['ice', 'igloo', 'ink']
    },
    'J': {
        'letter_names': ['j', 'jay', 'jei'],
        'phonemes': ['j', 'juh', 'jay'],
        'word': 'jam',
        'aliases': ['jam', 'jar', 'job']
    },
    'K': {
        'letter_names': ['k', 'kay', 'kaye'],
        'phonemes': ['k', 'kuh', 'kay'],
        'word': 'kite',
        'aliases': ['kite', 'king', 'kid']
    },
    'L': {
        'letter_names': ['l', 'el', 'ell', 'elle'],
        'phonemes': ['l', 'el', 'luh'],
        'word': 'lion',
        'aliases': ['lion', 'lamp', 'leg']
    },
    'M': {
        'letter_names': ['m', 'em', 'emm', 'emme'],
        'phonemes': ['m', 'em', 'muh'],
        'word': 'man',
        'aliases': ['man', 'mom', 'moon', 'map']
    },
    'N': {
        'letter_names': ['n', 'en', 'enn', 'enne'],
        'phonemes': ['n', 'en', 'nuh'],
        'word': 'net',
        'aliases': ['net', 'nest', 'nut', 'no']
    },
    'O': {
        'letter_names': ['o', 'oh', 'owe', 'oe'],
        'phonemes': ['o', 'oh', 'aw'],
        'word': 'on',
        'aliases': ['on', 'off', 'ox', 'octopus']
    },
    'P': {
        'letter_names': ['p', 'pe', 'pee', 'pea'],
        'phonemes': ['p', 'puh', 'pee', 'pe'],
        'word': 'pig',
        'aliases': ['pig', 'big', 'pick', 'pink', 'pin', 'pen']
    },
    'Q': {
        'letter_names': ['q', 'cue', 'kew', 'queue'],
        'phonemes': ['q', 'cue', 'kwa'],
        'word': 'queen',
        'aliases': ['queen', 'quick', 'quilt']
    },
    'R': {
        'letter_names': ['r', 'ar', 'are', 'our'],
        'phonemes': ['r', 'ar', 'ruh'],
        'word': 'red',
        'aliases': ['red', 'run', 'rat']
    },
    'S': {
        'letter_names': ['s', 'es', 'ess', 'esse', 'suh'],
        'phonemes': ['s', 'ss', 'es', 'see', 'suh'],
        'word': 'sun',
        'aliases': ['sun', 'son', 'sum', 'some', 'star', 'sung', 'song', 'soon', 'san', 'sam']
    },
    'T': {
        'letter_names': ['t', 'te', 'tee', 'tea', 'ti'],
        'phonemes': ['t', 'tuh', 'tee'],
        'word': 'top',
        'aliases': ['top', 'ten', 'tin', 'tag']
    },
    'U': {
        'letter_names': ['u', 'you', 'yu'],
        'phonemes': ['u', 'you', 'uh'],
        'word': 'up',
        'aliases': ['up', 'us', 'umbrella']
    },
    'V': {
        'letter_names': ['v', 've', 'vee'],
        'phonemes': ['v', 'vee', 'vuh'],
        'word': 'van',
        'aliases': ['van', 'vest', 'vet']
    },
    'W': {
        'letter_names': ['w', 'double-u', 'double u', 'doubleu'],
        'phonemes': ['w', 'wuh', 'double u'],
        'word': 'water',
        'aliases': ['water', 'web', 'win']
    },
    'X': {
        'letter_names': ['x', 'ex', 'eks'],
        'phonemes': ['x', 'ex', 'ks'],
        'word': 'box',
        'aliases': ['box', 'fox', 'six']
    },
    'Y': {
        'letter_names': ['y', 'why', 'wai'],
        'phonemes': ['y', 'why', 'yuh'],
        'word': 'yellow',
        'aliases': ['yellow', 'yes', 'yak']
    },
    'Z': {
        'letter_names': ['z', 'zee', 'zed'],
        'phonemes': ['z', 'zee', 'zed', 'zuh'],
        'word': 'zoo',
        'aliases': ['zoo', 'zebra', 'zip']
    }
}


def clean_text(text: str) -> str:
    """Normalize text into lowercase alphanumeric space-separated words."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())


def evaluate_phonetic_match(transcript: str, target_letter: str, target_word: str) -> Dict[str, Any]:
    """
    Evaluate if the spoken audio transcript matches the target letter or clinical word.
    Handles accent variations, phonetic approximations, and letter names directly (e.g. 'H' -> 'aitch', 'E' -> 'e' / 'ee').
    """
    clean_tr = clean_text(transcript)
    words = clean_tr.split()

    target_letter = target_letter.upper() if target_letter else ''
    target_word = clean_text(target_word)

    info = CLINICAL_MAP.get(target_letter, {
        'letter_names': [target_letter.lower()] if target_letter else [],
        'word': target_word,
        'aliases': [target_word] if target_word else [],
        'phonemes': [target_letter.lower()] if target_letter else []
    })

    letter_names = info.get('letter_names', [target_letter.lower()] if target_letter else [])
    phonemes = info.get('phonemes', [])
    word_aliases = info.get('aliases', [])

    # 1. Exact Letter Name Match (e.g., user said "H" -> "aitch", "E" -> "ee", etc.)
    is_letter_match = False
    matched_term = None

    if target_letter:
        target_char = target_letter.lower()
        if clean_tr == target_char or target_char in words:
            is_letter_match = True
            matched_term = target_letter
        else:
            for lname in letter_names:
                if lname == clean_tr or lname in words:
                    is_letter_match = True
                    matched_term = lname
                    break

    # 2. Target Word or Alias Match (when tested with a word prompt like 'apple')
    is_word_match = False
    if not is_letter_match and target_word:
        if target_word in words or target_word in clean_tr:
            is_word_match = True
            matched_term = target_word
        else:
            for alias in word_aliases:
                if alias in words or alias in clean_tr:
                    is_word_match = True
                    matched_term = alias
                    break

    # 3. Phoneme / Letter Sound Match
    is_sound_match = False
    if not is_letter_match and not is_word_match and target_letter:
        for ph in phonemes:
            if ph in words or clean_tr == ph or clean_tr.startswith(ph + ' '):
                is_sound_match = True
                matched_term = ph
                break

    is_overall_match = is_letter_match or is_word_match or is_sound_match
    confidence = 95 if is_letter_match else (90 if is_word_match else (85 if is_sound_match else 20))

    return {
        "is_match": is_overall_match,
        "match_type": "letter" if is_letter_match else ("word" if is_word_match else ("phoneme" if is_sound_match else "none")),
        "matched_term": matched_term or (words[0] if words else "none"),
        "confidence": confidence,
        "clean_transcript": clean_tr,
        "target_letter": target_letter,
        "target_word": target_word
    }


def transcribe_wav_bytes(audio_bytes: bytes, target_letter: str = "", target_word: str = "") -> Dict[str, Any]:
    """
    Transcribe WAV audio bytes using offline Vosk model.
    Accepts standard 16-bit PCM RIFF WAV or raw PCM16 samples.
    """
    if not VOSK_AVAILABLE:
        return {
            "status": "error",
            "message": "Vosk library not installed in Python environment",
            "text": "",
            "is_match": False
        }

    model = get_vosk_model()
    if model is None:
        return {
            "status": "error",
            "message": "Offline Vosk STT model file not found or failed to load",
            "text": "",
            "is_match": False
        }

    # Verify or unpack WAV header
    try:
        wav_io = io.BytesIO(audio_bytes)
        try:
            wf = wave.open(wav_io, "rb")
            sample_rate = wf.getframerate()
            pcm_data = wf.readframes(wf.getnframes())
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            wf.close()
        except wave.Error:
            # Maybe raw PCM 16kHz mono without header
            sample_rate = 16000
            channels = 1
            sampwidth = 2
            pcm_data = audio_bytes

        # If stereo, convert to mono by taking every other 16-bit sample
        if channels == 2 and sampwidth == 2:
            import numpy as np
            samples = np.frombuffer(pcm_data, dtype=np.int16)
            mono_samples = samples[::2]
            pcm_data = mono_samples.tobytes()

        # Initialize recognizer with actual audio sample rate
        # If target_letter is provided, constrain KaldiRecognizer to alphabet grammar so single letters are recognized accurately
        if target_letter:
            alphabet_grammar = [
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                'eight', 'age', 'each', 'ache', 'he', 'ee', 'ay', 'see', 'dee', 'ef',
                'gee', 'jay', 'kay', 'el', 'em', 'en', 'oh', 'pee', 'cue', 'ar',
                'es', 'tee', 'you', 'vee', 'ex', 'why', 'zee', '[unk]'
            ]
            if target_word:
                cw = clean_text(target_word)
                if cw and cw not in alphabet_grammar:
                    alphabet_grammar.append(cw)
            rec = KaldiRecognizer(model, sample_rate, json.dumps(alphabet_grammar))
        else:
            rec = KaldiRecognizer(model, sample_rate)
        rec.SetWords(True)

        # Process in chunks of 4000 bytes
        chunk_size = 4000
        for offset in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[offset:offset + chunk_size]
            rec.AcceptWaveform(chunk)

        final_res = json.loads(rec.FinalResult())
        transcript = final_res.get("text", "")

        match_eval = evaluate_phonetic_match(transcript, target_letter, target_word)

        return {
            "status": "success",
            "engine": "vosk_offline_ml",
            "transcript": transcript,
            "sample_rate": sample_rate,
            "evaluation": match_eval,
            "is_match": match_eval["is_match"],
            "matched_term": match_eval["matched_term"],
            "confidence": match_eval["confidence"]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Transcription error: {str(e)}",
            "transcript": "",
            "is_match": False
        }


if __name__ == "__main__":
    print("[*] Testing Vosk STT Module initialization...")
    model = get_vosk_model()
    if model:
        print("[+] Vosk STT model initialized successfully!")
    # Test evaluation logic
    print("Test match 'apple' ->", evaluate_phonetic_match("apple", "A", "Apple"))
    print("Test match 'bad' ->", evaluate_phonetic_match("bad", "B", "Bat"))
    print("Test match 'ah' ->", evaluate_phonetic_match("ah", "A", "Apple"))
    print("Test match 'dock' ->", evaluate_phonetic_match("dock", "D", "Dog"))
