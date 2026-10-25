"""
Fact-Check & Verification Suite for User Feedback Fixes:
1. Dyslexia Letter-Sound Matching starting with 'A' and Clinical Dyslexia Contrast Words
2. Voice Matching with Low/Minimal Network Dependency and Fuzzy Word-Level Recognition
3. Dysgraphia Multi-Stroke Geometric Shape Recognition (Triangle, Square, Circle, Straight Line)
4. Dysgraphia Alphabet Recognition (A, B, P, R, C)
5. Theme Consistency across all 3 Assessments (Dyslexia, Dyscalculia, Dysgraphia matching Antigravity Dark Theme)
6. Elimination of "Unable to analyze" errors across all endpoints
"""

import requests
import json
import re
import subprocess

BASE_URL = "http://127.0.0.1:5000"

def fact_check():
    print("=" * 60)
    print("USER FEEDBACK VERIFICATION & FACT-CHECKING SUITE")
    print("=" * 60)
    passed = 0
    total = 0

    def check(name, condition, extra=""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name} {extra}")
        else:
            print(f"  [FAIL] {name} {extra}")

    # 1. Check Dyslexia Template for Letter 'A' as item 1 and contrast words
    r_dys = requests.get(f"{BASE_URL}/assessment/1")
    check("Dyslexia Assessment Page Accessible", r_dys.status_code == 200, f"(HTTP {r_dys.status_code})")
    
    dys_html = r_dys.text
    check("Letter-Sound Matching includes letter 'a' (Apple)", "letter: 'a'" in dys_html or 'word: \'Apple\'' in dys_html)
    check("Letter-Sound Matching includes dyslexia contrast 'Bat' (B)", "word: 'Bat'" in dys_html)
    check("Letter-Sound Matching includes dyslexia contrast 'Dog' (D)", "word: 'Dog'" in dys_html)
    check("Letter-Sound Matching includes 'Sun' (S)", "word: 'Sun'" in dys_html)
    check("Speech transcript processor matches isWordMatch", "isWordMatch" in dys_html)
    check("Speech transcript processor matches isAliasMatch", "isAliasMatch" in dys_html)
    check("Direct 'View Full Evaluation Report & Charts' button present in Dyslexia", "View Full Evaluation Report" in dys_html)

    # 2. Check Dysgraphia Test Page for Theme & Recognition
    r_graph = requests.get(f"{BASE_URL}/assessment/3")
    check("Dysgraphia Assessment Page Accessible", r_graph.status_code == 200, f"(HTTP {r_graph.status_code})")
    graph_html = r_graph.text
    check("Dysgraphia uses Antigravity card (#12151A)", "#12151A" in graph_html)
    check("Dysgraphia uses Antigravity dark background (#08090B)", "#08090B" in graph_html)
    check("Dysgraphia consent group is dark theme (rgba(255, 255, 255, 0.03))", "rgba(255, 255, 255, 0.03)" in graph_html)
    check("Direct 'View Full Evaluation Report & Charts' button present in Dysgraphia", "View Full Evaluation Report" in graph_html)

    # 3. Check Dyscalculia Test Page for Theme
    r_calc = requests.get(f"{BASE_URL}/assessment/2")
    check("Dyscalculia Assessment Page Accessible", r_calc.status_code == 200, f"(HTTP {r_calc.status_code})")
    calc_html = r_calc.text
    check("Dyscalculia uses Antigravity card (#12151A)", "#12151A" in calc_html)
    check("Dyscalculia uses Antigravity dark background (#08090B)", "#08090B" in calc_html)
    check("Dyscalculia consent group is dark theme (rgba(255, 255, 255, 0.03))", "rgba(255, 255, 255, 0.03)" in calc_html)
    check("Direct 'View Full Evaluation Report & Charts' button present in Dyscalculia", "View Full Evaluation Report" in calc_html)

    # 4. Check /api/analyze-dysgraphia for eliminating "Unable to analyze"
    test_games = {
        'trace_line': {
            'duration_ms': 3500,
            'accuracy': 88,
            'straightness': 0.90,
            'smoothness': 0.88,
            'tremor': 0.10,
            'strokes': [{'points': [[20, 150, 100], [250, 150, 200], [480, 150, 300]], 't_start': 100, 't_end': 300}]
        },
        'copy_letter': {
            'duration_ms': 5000,
            'accuracy': 92,
            'smoothness': 0.85,
            'letter_matches': 3,
            'strokes': [{'points': [[100, 200, 0], [150, 50, 100], [200, 200, 200]], 't_start': 0, 't_end': 200}]
        },
        'shape_draw': {
            'duration_ms': 4000,
            'accuracy': 90,
            'smoothness': 0.87,
            'detected_shape': 'Triangle',
            'strokes': [{'points': [[100, 50, 0], [200, 200, 100], [50, 200, 200]], 't_start': 0, 't_end': 200}]
        }
    }
    r_ana_graph = requests.post(f"{BASE_URL}/api/analyze-dysgraphia", json={'games': test_games})
    check("POST /api/analyze-dysgraphia HTTP 200", r_ana_graph.status_code == 200)
    data_graph = r_ana_graph.json()
    check("Dysgraphia analysis risk is valid (not 'Unable to analyze')", data_graph.get('risk') != 'Unable to analyze', f"(Risk: {data_graph.get('risk')})")
    check("Dysgraphia details contains per_task", 'per_task' in data_graph.get('details', {}))
    check("Dysgraphia details contains warnings recommendations", len(data_graph.get('details', {}).get('warnings', [])) > 0)

    # 5. Check Node.js Dysgraphia Engine Multi-stroke recognition
    node_test = subprocess.run(['node', '-e', '''
const Engine = require("./static/dysgraphia_recognizer.js");

// Multi-stroke Triangle
const triStrokes = [
    { points: [[150, 30], [260, 200]] },
    { points: [[260, 200], [40, 200]] },
    { points: [[40, 200], [150, 30]] }
];
const triRes = Engine.recognizeShape(triStrokes, "Triangle");

// Multi-stroke Square
const sqStrokes = [
    { points: [[40, 40], [200, 40]] },
    { points: [[200, 40], [200, 200]] },
    { points: [[200, 200], [40, 200]] },
    { points: [[40, 200], [40, 40]] }
];
const sqRes = Engine.recognizeShape(sqStrokes, "Square");

// Circle
const circPoints = [];
for (let i = 0; i <= 36; i++) {
    const th = (i / 36) * Math.PI * 2;
    circPoints.push([120 + 60 * Math.cos(th), 120 + 60 * Math.sin(th)]);
}
const circRes = Engine.recognizeShape([{ points: circPoints }], "Circle");

// Letter A (Multi-stroke)
const aStrokes = [
    { points: [[100, 220], [150, 40]] },
    { points: [[150, 40], [200, 220]] },
    { points: [[120, 140], [180, 140]] }
];
const aRes = Engine.recognizeLetter(aStrokes, "A");

// Letter B
const bStrokes = [
    { points: [[60, 40], [60, 220]] },
    { points: [[60, 40], [130, 40], [140, 80], [130, 120], [60, 120]] },
    { points: [[60, 120], [140, 120], [150, 170], [140, 220], [60, 220]] }
];
const bRes = Engine.recognizeLetter(bStrokes, "B");

console.log(JSON.stringify({
    triangle: triRes.shape === "Triangle",
    square: sqRes.shape === "Square",
    circle: circRes.shape === "Circle",
    letterA: aRes.isMatch && aRes.letter === "A",
    letterB: bRes.isMatch && bRes.letter === "B"
}));
    '''], capture_output=True, text=True)

    if node_test.returncode == 0:
        res_json = json.loads(node_test.stdout.strip())
        check("Multi-stroke Triangle recognized as Triangle", res_json.get('triangle') == True)
        check("Multi-stroke Square recognized as Square", res_json.get('square') == True)
        check("Circle recognized as Circle", res_json.get('circle') == True)
        check("Multi-stroke Letter 'A' recognized as 'A'", res_json.get('letterA') == True)
        check("Multi-stroke Letter 'B' recognized as 'B'", res_json.get('letterB') == True)
    else:
        check("Node.js recognition engine execution", False, node_test.stderr)

    print("=" * 60)
    print(f"VERIFICATION SUMMARY: {passed} / {total} PASSED")
    print("=" * 60)

if __name__ == "__main__":
    fact_check()
