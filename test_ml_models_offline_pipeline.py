"""
Comprehensive Fact-Checking Verification Suite for Offline ML Models:
1. Offline Vosk Speech-to-Text ML Engine
2. Dysgraphia Neural Network Drawing Classifier (Shapes & Alphabets)
3. Full Assessment Analysis Endpoints (Dyslexia, Dysgraphia, Dyscalculia)
4. Model Size & Vercel Free-Tier Compatibility Check (<250MB)
"""

import os
import io
import wave
import json
import requests
import numpy as np
from ml_models.dysgraphia_recognizer_model import (
    generate_shape_strokes,
    generate_letter_strokes,
    predict_drawing,
    SHAPE_MODEL_PATH,
    LETTER_MODEL_PATH
)
from ml_models.vosk_stt import evaluate_phonetic_match, transcribe_wav_bytes, MODEL_PATH as VOSK_MODEL_PATH

BASE_URL = "http://127.0.0.1:5000"
passed = 0
failed = 0

def check(title: str, condition: bool, extra: str = ""):
    global passed, failed
    if condition:
        print(f"  [PASS] {title} {extra}")
        passed += 1
    else:
        print(f"  [FAIL] {title} {extra}")
        failed += 1


def generate_synthetic_wav(freq: float = 440.0, duration: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Generate a valid 16kHz mono 16-bit PCM WAV in bytes."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Sine wave modulated to int16
    audio = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return wav_buf.getvalue()


def run_tests():
    print("=" * 70)
    print("TEST SUITE: OFFLINE ML MODELS FACT-CHECK & VERIFICATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Model Storage & Vercel Size Compatibility
    # -------------------------------------------------------------
    print("\n--- 1. Model Artifacts & Bundle Size ---")
    check("Shape model file exists", os.path.exists(SHAPE_MODEL_PATH), f"({SHAPE_MODEL_PATH})")
    check("Letter model file exists", os.path.exists(LETTER_MODEL_PATH), f"({LETTER_MODEL_PATH})")
    check("Vosk small model directory exists", os.path.isdir(VOSK_MODEL_PATH), f"({VOSK_MODEL_PATH})")

    shape_size_kb = os.path.getsize(SHAPE_MODEL_PATH) / 1024 if os.path.exists(SHAPE_MODEL_PATH) else 0
    letter_size_kb = os.path.getsize(LETTER_MODEL_PATH) / 1024 if os.path.exists(LETTER_MODEL_PATH) else 0
    check("Shape model is lightweight (<5MB)", shape_size_kb < 5000, f"({shape_size_kb:.1f} KB)")
    check("Letter model is lightweight (<5MB)", letter_size_kb < 5000, f"({letter_size_kb:.1f} KB)")

    # Calculate total size of vosk model
    vosk_total_mb = sum(
        os.path.getsize(os.path.join(root, file))
        for root, dirs, files in os.walk(VOSK_MODEL_PATH)
        for file in files
    ) / (1024 * 1024) if os.path.isdir(VOSK_MODEL_PATH) else 0
    check("Vosk model size < 100MB (Vercel limit is 250MB)", vosk_total_mb < 100, f"({vosk_total_mb:.1f} MB)")

    # -------------------------------------------------------------
    # 2. Phonetic Clinical Matching (Zero Internet Dependency)
    # -------------------------------------------------------------
    print("\n--- 2. Clinical Contrast Word & Phonetic Matcher ---")
    test_cases = [
        ("apple", "A", "Apple", True, "word"),
        ("able", "A", "Apple", True, "word"),
        ("ah", "A", "Apple", True, "phoneme"),
        ("bat", "B", "Bat", True, "word"),
        ("bad", "B", "Bat", True, "word"),
        ("buh", "B", "Bat", True, "phoneme"),
        ("dog", "D", "Dog", True, "word"),
        ("dock", "D", "Dog", True, "word"),
        ("duh", "D", "Dog", True, "phoneme"),
        ("sun", "S", "Sun", True, "word"),
        ("cat", "C", "Cat", True, "word"),
        ("pig", "P", "Pig", True, "word"),
        ("xylophone", "A", "Apple", False, "none")
    ]
    for transcript, target_let, target_wd, expected_match, match_type in test_cases:
        res = evaluate_phonetic_match(transcript, target_let, target_wd)
        check(f"Match '{transcript}' for {target_let} ('{target_wd}')", res["is_match"] == expected_match, f"-> match_type={res['match_type']}")

    # -------------------------------------------------------------
    # 3. Speech-to-Text API Endpoint (/api/speech-to-text)
    # -------------------------------------------------------------
    print("\n--- 3. POST /api/speech-to-text ML Endpoint ---")
    wav_bytes = generate_synthetic_wav(freq=300.0, duration=0.4)
    r_stt = requests.post(
        f"{BASE_URL}/api/speech-to-text?target_letter=A&target_word=Apple",
        data=wav_bytes,
        headers={"Content-Type": "audio/wav"}
    )
    check("POST /api/speech-to-text status 200", r_stt.status_code == 200)
    stt_json = r_stt.json()
    check("STT status is success", stt_json.get("status") == "success")
    check("STT engine identifies as vosk_offline_ml", stt_json.get("engine") == "vosk_offline_ml")
    check("STT includes sample_rate 16000", stt_json.get("sample_rate") == 16000)

    # -------------------------------------------------------------
    # 4. Dysgraphia ML Drawing Recognition (/api/recognize-drawing)
    # -------------------------------------------------------------
    print("\n--- 4. POST /api/recognize-drawing ML Endpoint ---")
    shapes_to_test = [
        ("triangle", "Triangle"),
        ("square", "Square"),
        ("circle", "Circle"),
        ("line", "Line")
    ]
    for shape_key, shape_title in shapes_to_test:
        strokes = generate_shape_strokes(shape_key, jitter=0.01)
        r_shape = requests.post(
            f"{BASE_URL}/api/recognize-drawing",
            json={"strokes": strokes, "target_type": "shape", "target": shape_title}
        )
        check(f"POST /api/recognize-drawing ({shape_title}) HTTP 200", r_shape.status_code == 200)
        s_data = r_shape.json()
        check(f"ML recognizes {shape_title}", s_data.get("recognized").lower() == shape_title.lower() or s_data.get("is_match") is True, f"({s_data.get('recognized')}, {s_data.get('confidence')}%)")
        check(f"ML returns kinematics for {shape_title}", "smoothness" in s_data.get("kinematics", {}))

    letters_to_test = ["A", "B", "C", "P", "R"]
    for char in letters_to_test:
        strokes = generate_letter_strokes(char, jitter=0.01)
        r_letter = requests.post(
            f"{BASE_URL}/api/recognize-drawing",
            json={"strokes": strokes, "target_type": "letter", "target": char}
        )
        check(f"POST /api/recognize-drawing (Letter '{char}') HTTP 200", r_letter.status_code == 200)
        l_data = r_letter.json()
        check(f"ML recognizes Letter '{char}'", l_data.get("recognized").upper() == char or l_data.get("is_match") is True, f"({l_data.get('recognized')}, {l_data.get('confidence')}%)")

    # -------------------------------------------------------------
    # 5. Dysgraphia Assessment Analysis (/api/analyze-dysgraphia)
    # -------------------------------------------------------------
    print("\n--- 5. POST /api/analyze-dysgraphia End-to-End ---")
    mock_dysgraphia_games = {
        "trace_line": {
            "strokes": [[{"x": 50 + i * 5, "y": 150 + np.random.normal(0, 1), "time": 1000 + i * 20} for i in range(50)]],
            "duration_ms": 3200,
            "accuracy": 92,
            "smoothness": 0.90,
            "straightness": 0.92
        },
        "copy_letter": {
            "attempts": 3,
            "accuracy": 88,
            "letter_matches": 3,
            "letter_recognition_rate": 100,
            "smoothness": 0.86
        },
        "shape_draw": {
            "detected_shape": "Triangle",
            "accuracy": 90,
            "confidence": 95,
            "corners_detected": 3,
            "circularity": 0.55
        },
        "write_audio": {
            "attempts": 3,
            "accuracy": 85,
            "smoothness": 0.84
        },
        "timed_write": {
            "duration_ms": 30000,
            "accuracy": 80,
            "smoothness": 0.82
        }
    }
    r_ana = requests.post(f"{BASE_URL}/api/analyze-dysgraphia", json={"games": mock_dysgraphia_games})
    check("POST /api/analyze-dysgraphia status 200", r_ana.status_code == 200)
    ana_data = r_ana.json()
    check("Result has risk level", "risk" in ana_data and ana_data["risk"] != "Unable to analyze", f"({ana_data.get('risk')})")
    check("Result has risk_score", "risk_score" in ana_data)
    check("Result has per_task metrics", "per_task" in ana_data.get("details", {}))

    # -------------------------------------------------------------
    # 6. Dyslexia Assessment Analysis (/api/analyze-dyslexia)
    # -------------------------------------------------------------
    print("\n--- 6. POST /api/analyze-dyslexia End-to-End ---")
    mock_dyslexia_games = {
        "letter_sound": {"correct": 5, "total": 6, "avg_rt": 820},
        "phoneme_delete": {"correct": 5, "total": 6, "avg_rt": 1100},
        "rapid_naming": {"correct": 6, "total": 6, "avg_rt": 750},
        "rhyme_recog": {"correct": 5, "total": 6, "avg_rt": 900},
        "word_scramble": {"correct": 4, "total": 5, "avg_rt": 1200},
        "lexical_decision": {"correct": 7, "total": 8, "avg_rt": 680}
    }
    r_dys = requests.post(f"{BASE_URL}/api/analyze-dyslexia", json={"games": mock_dyslexia_games})
    check("POST /api/analyze-dyslexia status 200", r_dys.status_code == 200)
    dys_data = r_dys.json()
    check("Dyslexia risk level returned", "risk" in dys_data and dys_data["risk"] != "Unable to analyze", f"({dys_data.get('risk')})")

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 70)

    if failed == 0:
        print("[SUCCESS] All offline ML model checks and fact-checks PASSED!")
    else:
        print(f"[FAIL] {failed} test(s) failed. Please review the output above.")


if __name__ == "__main__":
    run_tests()
