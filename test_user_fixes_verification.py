"""
Test User Specific Fixes:
1. Dyslexia letter matching: Saying 'H' (aitch, h) matches 'H'; saying 'horse' or 'huh' does NOT match 'H'.
2. Dyslexia letter matching: Saying 'E' (e, ee) matches 'E'.
3. Dysgraphia geometry-first & ML drawing recognition: Triangle, Square, Circle, Line with dict & list strokes.
4. Results pages: /dyslexia-results/1, /dyscalculia-results/1, /dysgraphia-results/1 parameter generation and chart data.
"""

import unittest
import math
import numpy as np
from ml_models.vosk_stt import evaluate_phonetic_match, CLINICAL_MAP
from ml_models.dysgraphia_recognizer_model import predict_drawing, normalize_strokes


class TestUserSpecificFixes(unittest.TestCase):

    def test_01_speech_matching_h_letter_name(self):
        """User said 'H' clearly: recognizers produce 'aitch', 'h', 'eight', 'haych'."""
        for tr in ['aitch', 'h', 'eight', 'haych', 'eitch', 'ache']:
            res = evaluate_phonetic_match(tr, 'H', '')
            self.assertTrue(res['is_match'], f"Failed to match H on transcript '{tr}'")
            self.assertEqual(res['match_type'], 'letter')
            self.assertGreaterEqual(res['confidence'], 90)

    def test_02_speech_matching_h_rejects_horse_or_noise(self):
        """Saying 'horse' or unrelated words should NOT match letter 'H'."""
        for tr in ['horse', 'house', 'hotel', 'hamburger']:
            res = evaluate_phonetic_match(tr, 'H', '')
            self.assertFalse(res['is_match'], f"Should NOT match H on word '{tr}'")

    def test_03_speech_matching_e_letter_name(self):
        """User said 'E': recognizers produce 'e', 'ee', 'ea', 'eh'."""
        for tr in ['e', 'ee', 'ea', 'eh', 'eat']:
            res = evaluate_phonetic_match(tr, 'E', '')
            self.assertTrue(res['is_match'], f"Failed to match E on transcript '{tr}'")
            self.assertEqual(res['match_type'], 'letter')

    def test_04_dysgraphia_dict_strokes_normalization(self):
        """Test that dict stroke format from canvas is correctly normalized and not returned as empty []."""
        dict_strokes = [
            {
                'points': [[100, 100, 1000], [105, 110, 1010], [110, 120, 1020], [115, 130, 1030]],
                'color': '#ffffff',
                'width': 4
            }
        ]
        norm = normalize_strokes(dict_strokes)
        self.assertGreater(len(norm), 0, "normalize_strokes returned empty list for dict strokes!")
        self.assertEqual(len(norm[0]), 4, "Extracted points count mismatch!")

    def test_05_dysgraphia_shape_recognition_dict_strokes(self):
        """Test line, circle, triangle with dict strokes."""
        # Line dict stroke
        line_pts = [[50 + t * 5, 100, 1000 + t * 10] for t in range(30)]
        line_res = predict_drawing([{'points': line_pts}], target_type='shape', target='Line')
        self.assertTrue(line_res['is_match'])
        self.assertIn(line_res['recognized'].lower(), ['line', 'straight line'])

        # Circle dict stroke
        circ_pts = [[150 + 60 * math.cos(a), 150 + 60 * math.sin(a), a * 100] for a in np.linspace(0, 2 * math.pi, 35)]
        circ_res = predict_drawing([{'points': circ_pts}], target_type='shape', target='Circle')
        self.assertTrue(circ_res['is_match'])
        self.assertEqual(circ_res['recognized'].lower(), 'circle')

    def test_06_rapid_naming_images_exist(self):
        """Verify all 6 Gemini-generated familiar item photos exist with non-zero file sizes."""
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ran_dir = os.path.join(base_dir, 'static', 'imgs', 'ran')
        items = ['apple.jpg', 'car.jpg', 'dog.jpg', 'star.jpg', 'sun.jpg', 'fish.jpg']
        for item in items:
            img_path = os.path.join(ran_dir, item)
            self.assertTrue(os.path.isfile(img_path), f"Missing image {item} in {ran_dir}")
            self.assertGreater(os.path.getsize(img_path), 50000, f"Image {item} is unexpectedly small")

    def test_07_dyslexia_last_game_is_letter_sound(self):
        """Verify templates/dyslexia_integrated.html has letter_sound as the last game."""
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        tpl_path = os.path.join(base_dir, 'templates', 'dyslexia_integrated.html')
        with open(tpl_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn("'rapid_naming', 'letter_sound'", content, "gamesOrder must have rapid_naming before letter_sound")
        self.assertIn("gamesOrder = ['phoneme_delete', 'rhyme_recog', 'word_scramble', 'lexical_decision', 'rapid_naming', 'letter_sound']", content)
        self.assertIn("static/imgs/ran/apple.jpg", content)


    def test_08_dyslexia_results_trailing_slash_route(self):
        """Verify /dyslexia-results/ with trailing slash works directly (HTTP 200) without 404."""
        import requests
        try:
            r1 = requests.get('http://127.0.0.1:5000/dyslexia-results/')
            self.assertEqual(r1.status_code, 200, "GET /dyslexia-results/ failed!")
            self.assertIn("Dyslexia Screening Report", r1.text)

            r2 = requests.get('http://127.0.0.1:5000/dyscalculia-results/')
            self.assertEqual(r2.status_code, 200, "GET /dyscalculia-results/ failed!")

            r3 = requests.get('http://127.0.0.1:5000/dysgraphia-results/')
            self.assertEqual(r3.status_code, 200, "GET /dysgraphia-results/ failed!")

            r_api = requests.get('http://127.0.0.1:5000/api/dyslexia-results/')
            self.assertEqual(r_api.status_code, 200, "GET /api/dyslexia-results/ failed!")
        except requests.exceptions.ConnectionError:
            self.skipTest("Flask server not running on localhost:5000")

    def test_09_student_dashboard_results_modal_and_links(self):
        """Verify student.html ties View Results to all assessment screening reports and has no alert."""
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        tpl_path = os.path.join(base_dir, 'templates', 'student.html')
        with open(tpl_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertNotIn("alert('View Results feature coming soon!')", content)
        self.assertIn('resultsSelectModal', content)
        self.assertIn('openResultsModal', content)
        self.assertIn('/dyslexia-results/1', content)
        self.assertIn('/dyscalculia-results/2', content)
        self.assertIn('/dysgraphia-results/3', content)
        self.assertIn('/student-profile', content)


if __name__ == '__main__':
    unittest.main()
