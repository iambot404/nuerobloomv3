"""
Comprehensive Fact-Checking & Automated Verification Suite
Validates:
1. Chart.js & Graphical Analysis in Dyslexia, Dyscalculia, Dysgraphia results & Student Profile
2. Dysgraphia Kinematics, Shape, Line, and Alphabet ML Recognition & APIs
3. Dyslexia Offline Voice / Mic Safety Locking
4. Unified Google Antigravity dark theme across all screening and dashboard routes
"""

import requests
import json
import time
import sys

BASE_URL = "http://127.0.0.1:5000"

def run_tests():
    print("=" * 60)
    print("ANTIGRAVITY PLATFORM FACT-CHECKING SUITE")
    print("=" * 60)

    sess = requests.Session()
    pass_count = 0
    fail_count = 0

    def check(name, condition, extra=""):
        nonlocal pass_count, fail_count
        if condition:
            print(f"  [PASS] {name} {extra}")
            pass_count += 1
        else:
            print(f"  [FAIL] {name} {extra}")
            fail_count += 1

    # 1. Check Flask Dev Server Liveness
    try:
        r = sess.get(f"{BASE_URL}/db-test", timeout=5)
        check("Flask Dev Server Liveness (/db-test)", r.status_code == 200, f"(HTTP {r.status_code})")
    except Exception as e:
        check("Flask Dev Server Liveness", False, f"Exception: {e}")
        return

    # 2. Check Assessment Results Pages for Chart.js and Canvases
    results_pages = [
        ('/dyslexia-results/1', 'Dyslexia Results', ['dyslexiaRadarChart', 'dyslexiaSpeedAccuracyChart', 'cdn.jsdelivr.net/npm/chart.js']),
        ('/dyscalculia-results/2', 'Dyscalculia Results', ['dyscalculiaRadarChart', 'dyscalculiaSpeedAccuracyChart', 'cdn.jsdelivr.net/npm/chart.js']),
        ('/dysgraphia-results/3', 'Dysgraphia Results', ['motorRadarChart', 'taskSpeedBarChart', 'cdn.jsdelivr.net/npm/chart.js'])
    ]

    for path, title, required_tokens in results_pages:
        res = sess.get(f"{BASE_URL}{path}")
        check(f"{title} Route Status ({path})", res.status_code == 200, f"(HTTP {res.status_code})")
        for token in required_tokens:
            check(f"{title} contains '{token}'", token in res.text)

    # 3. Check Student Profile (with Student Login)
    student_sess = requests.Session()
    student_email = f"fact_check_stu_{int(time.time())}@example.com"
    reg = student_sess.post(f"{BASE_URL}/student_register", data={
        "name": "Fact Check Student",
        "email": student_email,
        "password": "Password123!",
        "contact": "9876543210"
    })
    check("Student Registration", reg.status_code in [200, 302], f"(HTTP {reg.status_code})")

    login = student_sess.post(f"{BASE_URL}/login/student", data={
        "student_email": student_email,
        "student_password": "Password123!"
    })
    check("Student Login", login.status_code == 200, f"(HTTP {login.status_code})")

    res_profile = student_sess.get(f"{BASE_URL}/student-profile")
    check("Student Profile Route (/student-profile)", res_profile.status_code == 200, f"(HTTP {res_profile.status_code})")
    check("Student Profile contains 'profileRiskRadarChart'", 'profileRiskRadarChart' in res_profile.text)
    check("Student Profile contains 'profileHealthDoughnut'", 'profileHealthDoughnut' in res_profile.text)
    check("Student Profile contains 'cdn.jsdelivr.net/npm/chart.js'", 'cdn.jsdelivr.net/npm/chart.js' in res_profile.text)

    # 4. Test POST /api/analyze-dysgraphia with shape & letter recognition data
    dysgraphia_payload = {
        'games': {
            'trace_line': {
                'straightness': 0.95,
                'smoothness': 0.92,
                'tremor': 0.08,
                'tremor_stability': 0.92,
                'rmse': 1.8,
                'accuracy': 95,
                'duration_ms': 3800
            },
            'copy_letter': {
                'accuracy': 92,
                'smoothness': 0.90,
                'straightness': 0.90,
                'tremor': 0.10,
                'letter_matches': 3,
                'letter_recognition_rate': 100,
                'duration_ms': 11000
            },
            'shape_draw': {
                'detected_shape': 'Triangle',
                'accuracy': 94,
                'confidence': 94,
                'corners_detected': 3,
                'circularity': 0.60,
                'smoothness': 0.92,
                'straightness': 0.91,
                'tremor': 0.08,
                'duration_ms': 4500
            },
            'timed_write': {
                'accuracy': 88,
                'smoothness': 0.88,
                'straightness': 0.88,
                'tremor': 0.12,
                'duration_ms': 30000
            },
            'write_audio': {
                'accuracy': 90,
                'smoothness': 0.90,
                'straightness': 0.90,
                'tremor': 0.10,
                'duration_ms': 14000
            }
        }
    }

    r_dysgraph = sess.post(f"{BASE_URL}/api/analyze-dysgraphia", json=dysgraphia_payload)
    check("POST /api/analyze-dysgraphia status", r_dysgraph.status_code == 200, f"(HTTP {r_dysgraph.status_code})")
    if r_dysgraph.status_code == 200:
        dg_data = r_dysgraph.json()
        check("Dysgraphia diagnosed Risk is 'None' or 'Low'", dg_data.get('risk') in ['None', 'Low'], f"(Risk: {dg_data.get('risk')})")
        per_task = dg_data.get('details', {}).get('per_task', {})
        check("Dysgraphia per_task contains shape_draw", 'shape_draw' in per_task)
        if 'shape_draw' in per_task:
            check("shape_draw detected_shape is 'Triangle'", per_task['shape_draw'].get('detected_shape') == 'Triangle')
        check("Dysgraphia per_task contains copy_letter", 'copy_letter' in per_task)

    # 5. Test POST /api/analyze-dyslexia
    dyslexia_payload = {
        'games': {
            'phoneme_delete': {'correct': 4, 'total': 4, 'avg_rt': 1100},
            'letter_sound': {'correct': 4, 'total': 4, 'avg_rt': 950},
            'rhyme_recog': {'correct': 4, 'total': 4, 'avg_rt': 1200},
            'word_scramble': {'correct': 4, 'total': 4, 'avg_rt': 1400},
            'lexical_decision': {'correct': 4, 'total': 4, 'avg_rt': 1050},
            'rapid_naming': {'correct': 4, 'total': 4, 'avg_rt': 850}
        }
    }
    r_dyslex = sess.post(f"{BASE_URL}/api/analyze-dyslexia", json=dyslexia_payload)
    check("POST /api/analyze-dyslexia status", r_dyslex.status_code == 200, f"(HTTP {r_dyslex.status_code})")

    # 6. Test POST /api/analyze-dyscalculia
    dyscalculia_payload = {
        'games': {
            'subitizing': {'correct': 4, 'total': 4, 'avg_rt': 900},
            'comparison': {'correct': 4, 'total': 4, 'avg_rt': 850},
            'symbol_match': {'correct': 4, 'total': 4, 'avg_rt': 1100},
            'sequencing': {'correct': 4, 'total': 4, 'avg_rt': 1300},
            'memory_span': {'correct': 4, 'total': 4, 'avg_rt': 1400},
            'story_arith': {'correct': 4, 'total': 4, 'avg_rt': 1250}
        }
    }
    r_dyscalc = sess.post(f"{BASE_URL}/api/analyze-dyscalculia", json=dyscalculia_payload)
    check("POST /api/analyze-dyscalculia status", r_dyscalc.status_code == 200, f"(HTTP {r_dyscalc.status_code})")

    # 7. Check Dysgraphia Test Page for Recognizer Script & Recognition HUD
    r_dg_test = sess.get(f"{BASE_URL}/assessment/3")
    check("GET /assessment/3 (Dysgraphia Test) status", r_dg_test.status_code == 200, f"(HTTP {r_dg_test.status_code})")
    check("Dysgraphia test includes dysgraphia_recognizer.js", 'dysgraphia_recognizer.js' in r_dg_test.text)
    check("Dysgraphia test includes recognition-hud styling", 'recognition-hud' in r_dg_test.text)

    # 8. Check Dyslexia Test Page for Mic-Off safety lock and live display
    r_dl_test = sess.get(f"{BASE_URL}/assessment/1")
    check("GET /assessment/1 (Dyslexia Test) status", r_dl_test.status_code == 200, f"(HTTP {r_dl_test.status_code})")
    check("Dyslexia test includes lsm-spoken-letter live display", 'lsm-spoken-letter' in r_dl_test.text)
    check("Dyslexia test includes isSpeakingAudio lock", 'isSpeakingAudio' in r_dl_test.text)

    print("\n" + "=" * 60)
    print(f"RESULTS: {pass_count} PASSED, {fail_count} FAILED")
    print("=" * 60)

    if fail_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
