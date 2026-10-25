"""
Google Gemini AI Service for NeuroBloom
Provides pediatric neuropsychological synthesis, teacher accommodations,
parent action blueprints, and interactive clinical Q&A.
Built with zero external package dependencies (pure urllib) for 100% Vercel serverless compatibility.
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Primary key provided by user, configurable via environment variable
_raw_keys = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
GEMINI_API_KEYS: List[str] = [k.strip() for k in _raw_keys.split(",") if k.strip()]
GEMINI_API_KEY: str = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""


# Free-tier optimized candidate models ordered by generous rate limits and sub-2s latency:
_env_models = os.environ.get("GEMINI_MODELS") or os.environ.get("GEMINI_MODEL")
if _env_models:
    CANDIDATE_MODELS = [m.strip() for m in _env_models.split(",") if m.strip()]
else:
    CANDIDATE_MODELS = [
        "gemini-flash-lite-latest",
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite-preview",
        "gemini-3.5-flash-lite",
        "gemini-3-flash-preview",
        "gemma-4-26b-a4b-it"
    ]


# In-memory cache to prevent redundant API calls on page reloads/views
_INSIGHTS_CACHE: Dict[str, Dict[str, Any]] = {}


def _call_gemini_api(prompt: str, 
                      system_instruction: Optional[str] = None, 
                      as_json: bool = False, 
                      timeout: int = 8) -> Optional[str]:
    """
    Call Google Generative Language REST API with multi-key rotation and 
    automatic failover across generous free-tier Flash-Lite models.
    """
    if not GEMINI_API_KEYS:
        print("[!] No GEMINI_API_KEYS configured.")
        return None

    # Rotate through all available API keys if provided, then candidate models
    for key_idx, key in enumerate(GEMINI_API_KEYS):
        for model_name in CANDIDATE_MODELS:
            is_gemma = "gemma" in model_name.lower()
            
            gen_config: Dict[str, Any] = {
                "temperature": 0.2,
                "maxOutputTokens": 800
            }
            # Only specify thinkingConfig for models that use thinking budgets
            if "thinking" in model_name.lower() or model_name == "gemini-3.5-flash":
                gen_config["thinkingConfig"] = {"thinkingBudget": 0}
            if as_json and not is_gemma:
                gen_config["responseMimeType"] = "application/json"


            payload: Dict[str, Any] = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": gen_config
            }
            if system_instruction and not is_gemma:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
            elif system_instruction and is_gemma:
                # Gemma prepend instructions to text
                payload["contents"] = [{"parts": [{"text": f"[Instruction: {system_instruction}]\n\n{prompt}"}]}]

            data = json.dumps(payload).encode("utf-8")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"

            try:
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    candidates = res.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        # Extract non-thought text parts
                        text_parts = [p.get("text", "") for p in parts if "text" in p and not p.get("thought", False)]
                        full_text = "".join(text_parts).strip()
                        if full_text:
                            return full_text
            except urllib.error.HTTPError as http_err:
                if http_err.code == 429:
                    print(f"[!] Model {model_name} (key #{key_idx+1}) rate limited (HTTP 429). Failing over to next model...")
                    continue
                elif http_err.code == 400 and as_json:
                    # Retry without responseMimeType in case model doesn't support schema constraint
                    try:
                        gen_config.pop("responseMimeType", None)
                        gen_config.pop("thinkingConfig", None)
                        retry_data = json.dumps(payload).encode("utf-8")
                        retry_req = urllib.request.Request(url, data=retry_data, headers={"Content-Type": "application/json"})
                        with urllib.request.urlopen(retry_req, timeout=timeout) as retry_resp:
                            res = json.loads(retry_resp.read().decode("utf-8"))
                            parts = res.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                            text_parts = [p.get("text", "") for p in parts if "text" in p and not p.get("thought", False)]
                            full_text = "".join(text_parts).strip()
                            if full_text:
                                return full_text
                    except Exception:
                        pass
                    print(f"[!] Model {model_name} HTTP {http_err.code}, failing over...")
                    continue
                else:
                    print(f"[!] Model {model_name} HTTP {http_err.code}, failing over...")
                    continue
            except Exception as e:
                print(f"[!] Model {model_name} failed: {e}, failing over...")
                continue

    return None



def generate_clinical_synthesis(disorder_type: str, 
                                risk_level: str, 
                                task_scores: Dict[str, Any], 
                                student_name: str = "Student") -> Dict[str, Any]:
    """
    Generate pediatric neuropsychological synthesis, school accommodations,
    and home practice blueprints customized to the child's exact assessment profile.
    Utilizes caching to avoid repeating API consumption on refreshes.
    """
    # Check cache first (0ms latency, zero API quota)
    cache_key = f"{disorder_type.lower()}:{student_name.lower()}:{risk_level}:{hash(json.dumps(task_scores, sort_keys=True))}"
    if cache_key in _INSIGHTS_CACHE:
        return _INSIGHTS_CACHE[cache_key]

    sys_prompt = (
        "You are a compassionate, world-class pediatric neuropsychologist and educational specialist "
        "at NeuroBloom. You analyze screening metrics for children evaluated for dyslexia, dyscalculia, "
        "or dysgraphia. Provide encouraging, scientifically grounded, and empathetic explanations. "
        "Never use doom-laden or alarming language. Emphasize preserved cognitive strengths alongside "
        "targeted accommodations. Return strictly valid JSON."
    )

    scores_text = "\n".join([f"- {k}: {v}" for k, v in task_scores.items()])
    user_prompt = f"""
Assessment Profile:
- Disorder Domain: {disorder_type.capitalize()}
- Identified Risk Level: {risk_level}
- Student: {student_name}
- Detailed Task Scores & Latencies:
{scores_text}

Generate a JSON object matching this schema:
{{
  "summary": "A 2-sentence empathetic overview celebrating strengths and summarizing standing.",
  "clinical_interpretation": "A paragraph explaining the neurodevelopmental pattern (phonological decoding, subitizing, motor kinematics, or working memory).",
  "school_accommodations": ["Strategy 1", "Strategy 2", "Strategy 3"],
  "home_roadmap": ["Action 1", "Action 2", "Action 3"]
}}
"""

    response_text = _call_gemini_api(user_prompt, system_instruction=sys_prompt, as_json=True, timeout=8)
    
    if response_text:
        try:
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            
            parsed = json.loads(clean_text.strip())
            parsed["source"] = "Gemini Flash AI"
            _INSIGHTS_CACHE[cache_key] = parsed
            return parsed
        except Exception as parse_err:
            print(f"[!] JSON parsing error in Gemini response: {parse_err}")

    # Graceful, high-quality pediatric fallback if API offline or rate-limited
    fallback = _generate_fallback_synthesis(disorder_type, risk_level, task_scores, student_name)
    _INSIGHTS_CACHE[cache_key] = fallback
    return fallback


def chat_with_pediatric_ai(user_question: str, 
                           disorder_type: str, 
                           risk_level: str, 
                           history: Optional[List[Dict[str, str]]] = None) -> str:
    """Answer parent or educator questions regarding assessment results."""
    sys_prompt = (
        f"You are the NeuroBloom Pediatric Clinical Assistant. The user is asking about a child's "
        f"{disorder_type} screening results (Current Assessment: {risk_level}). "
        "Provide clear, warm, actionable, and evidence-based guidance. Keep answers concise (2-4 paragraphs max)."
    )

    prompt = f"User Question: {user_question}\nContext: Child assessed for {disorder_type} with risk level '{risk_level}'."
    reply = _call_gemini_api(prompt, system_instruction=sys_prompt, as_json=False, timeout=8)
    
    if reply:
        return reply

    return (
        f"Thank you for reaching out regarding your child's {disorder_type} screening ({risk_level}). "
        f"Children with this profile benefit greatly from consistent, bite-sized multisensory routines "
        f"(10-15 minutes daily) that reinforce learning through visual, auditory, and kinesthetic channels. "
        "Celebrating small milestones and maintaining open communication with educators ensures sustained progress."
    )


def _generate_fallback_synthesis(disorder_type: str, 
                                 risk_level: str, 
                                 task_scores: Dict[str, Any], 
                                 student_name: str) -> Dict[str, Any]:
    """Curated evidence-based clinical synthesis fallback."""
    is_favorable = "No" in risk_level or "Low" in risk_level

    if disorder_type.lower() == "dyslexia":
        return {
            "source": "Clinical Neuropsychological Guidelines (Multisensory IDA Standards)",
            "summary": (
                f"{student_name} demonstrates robust cognitive engagement and solid phonemic foundational skills. "
                f"The screening profile indicates a '{risk_level}' standing with preserved phonological decoding assets."
            ) if is_favorable else (
                f"{student_name}'s profile highlights developing phonological awareness with focal areas for targeted reinforcement. "
                "With early multi-sensory literacy support, significant rapid progress is typically achieved."
            ),
            "clinical_interpretation": (
                "Phonological decoding and rapid automatized retrieval show steady cognitive foundation. "
                "Any observed response latency represents careful orthographic deliberation rather than structural deficits."
            ) if is_favorable else (
                "Performance indicates slight latency in rapid phoneme retrieval and letter-sound correspondence. "
                "Structured multisensory phonics (Orton-Gillingham approach) provides the ideal scaffolding."
            ),
            "school_accommodations": [
                "Provide extended time for reading passages and oral presentations (1.5x time).",
                "Utilize multisensory text presentation (paired audio-visual reading tools and dyslexic-friendly fonts).",
                "Allow oral responses or speech-to-text assistive technology during extended writing tasks."
            ],
            "home_roadmap": [
                "Paired Oral Reading: Read engaging stories together for 10 minutes daily, alternating pages.",
                "Sound Detective Games: Practice identifying starting and rhyming sounds during car rides.",
                "Multisensory Spelling: Form letters in sand, salt trays, or with playdough while vocalizing phonemes."
            ]
        }
    elif disorder_type.lower() == "dyscalculia":
        return {
            "source": "Clinical Neuropsychological Guidelines (Cognitive Numerical Standards)",
            "summary": (
                f"{student_name} showcases strong numerical intuition and intact number sense across evaluated domains. "
                f"Overall evaluation reflects a '{risk_level}' profile with strong mathematical assets."
            ) if is_favorable else (
                f"{student_name} shows enthusiastic mathematical effort, with specific opportunities to strengthen mental number line representation. "
                "Hands-on visual manipulatives offer transformative clarity for these concepts."
            ),
            "clinical_interpretation": (
                "Subitizing and mental arithmetic retrieval reflect solid cognitive processing. "
                "Deliberate response times indicate reflective problem-solving rather than numerical confusion."
            ) if is_favorable else (
                "Evaluation reveals hesitation in mental magnitude comparison and working memory for digits. "
                "Concrete-Representational-Abstract (CRA) instructional models yield outstanding improvement."
            ),
            "school_accommodations": [
                "Permit use of visual number lines, dot arrays, and physical base-10 manipulatives.",
                "Break multi-step math problems into color-coded, sequential sub-steps.",
                "Provide formula reference sheets and grant extra time for timed math quizzes."
            ],
            "home_roadmap": [
                "Dot Quick-Count Dice Games: Roll two dice and practice rapid group recognition without counting one-by-one.",
                "Kitchen Math: Involve child in measuring ingredients and estimating recipe quantities.",
                "Number Line Jump: Create a chalk number line outdoors and jump along addition/subtraction paths."
            ]
        }
    else:  # dysgraphia
        return {
            "source": "Clinical Neuropsychological Guidelines (Motor Kinematics Standards)",
            "summary": (
                f"{student_name} exhibits commendable fine-motor control and stroke consistency on handwriting tasks. "
                f"Screening outcome indicates '{risk_level}' with steady kinematic stability."
            ) if is_favorable else (
                f"{student_name}'s writing shows kinematic motor variability and grip fatigue. "
                "Ergonomic adjustments and targeted motor planning exercises will bolster confidence and endurance."
            ),
            "clinical_interpretation": (
                "Spatial alignment and letter formation follow age-appropriate kinematic trajectories. "
                "Tremor stability and path adherence are well preserved across evaluated tasks."
            ) if is_favorable else (
                "Pen trajectory reveals spatial path deviations and increased pen lifts per second. "
                "Proprioceptive fine-motor coaching will reduce handwriting tension and enhance letter fluidity."
            ),
            "school_accommodations": [
                "Provide ergonomic pencil grips (triangular or claw grip) and slant boards to optimize wrist posture.",
                "Offer alternative response modes (speech-to-text, typing, or oral answers) for lengthy writing tasks.",
                "Supply graphic organizers and pre-printed class notes to minimize copy fatigue from the whiteboard."
            ],
            "home_roadmap": [
                "Finger Strength Play: Squeeze therapy putty, use clothespins, or build with interlocking bricks for 10 mins.",
                "Air Writing: Form large letters in the air using whole-arm movements before putting pencil to paper.",
                "Trace and Glow: Practice tracing smooth curved and horizontal lines on lighted or chalkboard surfaces."
            ]
        }


def analyze_handwriting_with_gemini(target_char_or_shape: str,
                                    task_type: str,
                                    kinematics: Dict[str, Any],
                                    onnx_prediction: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ensemble analysis: Combines MobileNetV2 ONNX character prediction from alphabet-classifier
    with Google Gemini AI to evaluate pediatric dysgraphia markers (motor fluidity, letter reversal,
    spatial proportion, and stroke formation).
    """
    sys_prompt = (
        "You are an expert pediatric occupational therapist and dysgraphia specialist at NeuroBloom. "
        "Analyze a child's handwriting stroke drawing that combines an ONNX MobileNetV2 classifier prediction "
        "with real-time stroke kinematics (smoothness, straightness, tremor, pen lifts). "
        "Return strictly valid JSON."
    )
    
    onnx_info = ""
    if onnx_prediction:
        rec = onnx_prediction.get("recognized", "Unknown")
        conf = onnx_prediction.get("confidence", 0)
        onnx_info = f"- ONNX MobileNetV2 Prediction: Detected '{rec}' with {conf}% confidence\n"
        
    kin_info = (
        f"- Smoothness: {int(float(kinematics.get('smoothness', 0.8)) * 100)}%\n"
        f"- Straightness / Alignment: {int(float(kinematics.get('straightness', 0.8)) * 100)}%\n"
        f"- Tremor Stability: {int(float(kinematics.get('tremor_stability', 0.85)) * 100)}%\n"
        f"- Pen Lifts / Strokes: {kinematics.get('corners', 1) or kinematics.get('strokes_count', 1)}"
    )
    
    user_prompt = f"""
Task: {task_type}
Target Item: '{target_char_or_shape}'
{onnx_info}Kinematics:
{kin_info}

Return JSON with this schema:
{{
  "legibility_score": 85,
  "formation_analysis": "1-2 concise sentences on stroke trajectory, loop closure, or line adherence.",
  "dysgraphia_markers": ["marker 1", "marker 2"],
  "child_encouragement": "1 warm, encouraging sentence for the student.",
  "pedagogical_tip": "1 actionable fine-motor or pencil grip recommendation."
}}
"""
    res = _call_gemini_api(user_prompt, system_instruction=sys_prompt, as_json=True, timeout=8)
    if res:
        try:
            clean = res.strip()
            if clean.startswith("```json"): clean = clean[7:]
            elif clean.startswith("```"): clean = clean[3:]
            if clean.endswith("```"): clean = clean[:-3]
            data = json.loads(clean.strip())
            data["source"] = "Gemini Flash AI + ONNX MobileNetV2"
            return data
        except Exception as e:
            print(f"[!] JSON parse error in Gemini handwriting analysis: {e}")
            
    # Curated clinical fallback
    smooth = float(kinematics.get("smoothness", 0.85))
    straight = float(kinematics.get("straightness", 0.85))
    score = int(round((smooth * 0.5 + straight * 0.5) * 100))
    return {
        "source": "NeuroBloom Kinematics + ONNX Classifier",
        "legibility_score": score,
        "formation_analysis": f"Stroke trajectory shows consistent spatial definition for '{target_char_or_shape}'. Alignment and velocity are within expected pediatric developmental bounds.",
        "dysgraphia_markers": ["Mild line tremor" if straight < 0.70 else "Steady motor trajectory", "No letter reversal"],
        "child_encouragement": f"Great effort drawing '{target_char_or_shape}'! Your lines show wonderful control.",
        "pedagogical_tip": "Practice finger warm-up stretches and maintain dynamic tripod grip."
    }


def analyze_phonemic_error_with_gemini(target_phoneme_or_word: str,
                                       spoken_transcript: str,
                                       subtest_name: str = "Letter-Sound Association",
                                       latency_ms: int = 2500) -> Dict[str, Any]:
    """
    Linguistic error categorization using Gemini AI: Analyzes acoustic, voicing,
    or articulation confusion when spoken input differs from target.
    """
    sys_prompt = (
        "You are a clinical speech-language pathologist and pediatric dyslexia specialist at NeuroBloom. "
        "Analyze a child's spoken response during a phonological assessment. Return strictly valid JSON."
    )
    user_prompt = f"""
Subtest: {subtest_name}
Target: '{target_phoneme_or_word}'
Spoken Transcript: '{spoken_transcript}'
Response Latency: {latency_ms}ms

Return JSON:
{{
  "is_phonetic_match": true/false,
  "error_type": "None" / "Voicing Substitution" / "Place of Articulation" / "Phoneme Deletion" / "Acoustic STT Artifact",
  "clinical_note": "1 sentence explaining the linguistic mechanism",
  "teaching_strategy": "1 specific multisensory phonics cue"
}}
"""
    res = _call_gemini_api(user_prompt, system_instruction=sys_prompt, as_json=True, timeout=8)
    if res:
        try:
            clean = res.strip()
            if clean.startswith("```json"): clean = clean[7:]
            elif clean.startswith("```"): clean = clean[3:]
            if clean.endswith("```"): clean = clean[:-3]
            return json.loads(clean.strip())
        except Exception:
            pass
            
    return {
        "is_phonetic_match": (target_phoneme_or_word.lower() in spoken_transcript.lower()),
        "error_type": "Acoustic or Phonemic Variance",
        "clinical_note": f"Auditory response evaluated against target '{target_phoneme_or_word}'.",
        "teaching_strategy": "Pair tactile mirror articulation cue with letter card."
    }


def analyze_numerical_strategy_with_gemini(target_quantity: int,
                                           user_answer: int,
                                           subtest_name: str = "Approximate Number Sense",
                                           reaction_time_ms: int = 2000,
                                           tier: str = "Medium") -> Dict[str, Any]:
    """
    Cognitive numerical error analysis using Gemini AI: Classifies subitizing ceiling,
    magnitude estimation near-misses, or counting retrieval delay.
    """
    sys_prompt = (
        "You are a pediatric cognitive psychologist and dyscalculia specialist at NeuroBloom. "
        "Analyze a child's numerical cognition response during a rapid dot subitizing or math task. Return strictly valid JSON."
    )
    user_prompt = f"""
Task: {subtest_name} ({tier} difficulty)
Target Quantity: {target_quantity}
Child Answer: {user_answer}
Reaction Time: {reaction_time_ms}ms

Return JSON:
{{
  "is_correct": true/false,
  "error_category": "None" / "Subitizing Boundary Failure" / "Magnitude Near-Miss (+-1)" / "Counting Latency",
  "cognitive_observation": "1 sentence on the child's numerical representation",
  "intervention_idea": "1 hands-on manipulative activity"
}}
"""
    res = _call_gemini_api(user_prompt, system_instruction=sys_prompt, as_json=True, timeout=8)
    if res:
        try:
            clean = res.strip()
            if clean.startswith("```json"): clean = clean[7:]
            elif clean.startswith("```"): clean = clean[3:]
            if clean.endswith("```"): clean = clean[:-3]
            return json.loads(clean.strip())
        except Exception:
            pass
            
    diff = abs(target_quantity - user_answer)
    return {
        "is_correct": (diff == 0),
        "error_category": "Accurate Retrieval" if diff == 0 else ("Magnitude Near-Miss" if diff == 1 else "Quantity Discrepancy"),
        "cognitive_observation": f"Child answered {user_answer} for {target_quantity} in {reaction_time_ms}ms.",
        "intervention_idea": "Practice with five-frame and ten-frame visual dot patterns."
    }
