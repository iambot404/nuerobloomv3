# DocHub AI Architecture & Enhancement Plan
## Offline STT, Dysgraphia Shape Recognition, Cognitive Analytics Charts & Vercel Deployment

---

## 1. Executive Summary

This architecture and implementation plan addresses four interconnected enhancements requested for the **DocHub / Neurobloom** platform:

1. **Offline / Zero-Latency Speech-to-Text (STT)**: Eliminating dependence on external cloud STT APIs that introduce network lag during voice screening, while strictly respecting Vercel serverless size and execution limits.
2. **Dysgraphia Machine Learning & Shape Recognition Model**: Replacing dummy drawing evaluations in `dysgraphia_test.html` with an intelligent geometric and kinematic stroke recognition model that detects straight lines, geometric shapes (triangles, squares, circles), and alphabets, while evaluating handwriting biomarkers (tremor, jerk, velocity, straightness).
3. **Comprehensive Charts & Graphical Cognitive Analysis**:
   - **Assessment Results Pages** (`dyslexia_results.html`, `dyscalculia_results.html`, `dysgraphia_results.html`): Adding rich cognitive radar charts, speed-versus-accuracy profiles, and subtest struggle diagnostics.
   - **Student Profile** (`student-profile.html`): Adding a clean, executive cognitive vulnerability chart showing risk percentages and areas of strength vs. struggle at a glance.
4. **Vercel MCP Deployment Protocol**: Establishing a strict local validation workflow, followed by deploying as a new project via the integrated Vercel MCP server. i am using free tier 

---

## 2. Speech-to-Text (STT): Offline & Low-Latency Model Architecture

### 2.1 The Vercel Serverless Constraint
Vercel serverless functions (AWS Lambda under the hood) have strict constraints:
- **Maximum uncompressed bundle size**: **250 MB** (including Python runtime, installed libraries, and models).
- **No dedicated GPU**: Serverless CPU execution with 10s–60s timeouts.
- **Request payload limit**: 4.5 MB.
- **Implication**: Heavy server-side speech models (such as full PyTorch + OpenAI Whisper ~1.5 GB+) **cannot** run directly inside a Vercel Python serverless lambda. Attempting to bundle PyTorch or TensorFlow on Vercel will trigger a `FUNCTION_SIZE_LIMIT_EXCEEDED` build failure.

### 2.2 Model Candidates Comparison

| Strategy | Technology | Size | Latency | Vercel Compatible? | Internet Required? | Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Option A: Client-Side In-Browser WASM / ONNX** *(Recommended)* | **Transformers.js (Whisper-tiny.en)** or **Vosk-browser WebAssembly** | ~39 MB (cached in browser IndexedDB) | 50–150ms on device | **100% Yes** (Zero server load) | **No** (Works 100% offline once cached) | **Best Choice**: Zero server cost, zero Vercel bundle bloat, zero API network delays. |
| **Option B: Specialized Acoustic Keyword Spotter** | **TensorFlow.js SpeechCommands / Lightweight Letter Acoustic Model** | ~4 MB | <30ms | **100% Yes** | **No** (100% offline) | Extremely fast for single letter/sound matches (A, B, C... muh, puh). |
| **Option C: Hybrid Offline-First Web Speech API** | Browser Native Offline Engine (Chrome/Edge on-device speech packs) with auto-fallback | 0 MB | Instant | **100% Yes** | No (if OS speech pack present) / Yes (if fallback) | Already partially configured; needs offline error-recovery & cache bridge. |
| **Option D: Server-side Python Whisper** | `whisper` / `faster-whisper` in Flask backend | >1.2 GB | 1–4s (CPU) | **NO** (Exceeds Vercel 250MB limit) | No | **Incompatible with Vercel deployment**. |

### 2.3 Recommended Architecture: In-Browser Offline Neural Engine
1. **Engine**: In-browser WebAssembly speech recognition using **Transformers.js (`whisper-tiny.en` quantized ONNX)** or **Vosk-browser**.
2. **Offline Caching**: The model files (~39 MB) are cached locally in the student's browser (IndexedDB / CacheStorage) upon first access.
3. **Microphone Lock**: Preserves our verified protocol: the microphone remains strictly **OFF** while the system is speaking questions/instructions, then turns **ON** instantaneously when user input is required.
4. **Zero-Latency Fallback**: If the device has limited hardware capability, it immediately falls back to the local Web Speech API with no user disruption.

---

## 3. Dysgraphia ML Model: Shape, Alphabet & Stroke Analysis

### 3.1 The Problem
In `templates/dysgraphia_test.html`, the drawing tests (`trace_line`, `copy_letter`, `shape_draw`) currently collect raw stroke coordinates and duration without evaluating shape fidelity, geometric correctness, or fine-motor biomarkers.

### 3.2 Recommended Multi-Layer Model Pipeline

```
                     Student Drawing Canvas (Touch / Mouse Coordinates)
                                        │
                                        ▼
             ┌──────────────────────────────────────────────────────┐
             │       Layer 1: Kinematic Biomarker Extraction        │
             │   - Tremor / Jerk Index (d³x/dt³ jerk metric)        │
             │   - Straightness RMSE (Orthogonal regression)        │
             │   - Velocity Profile & Speed Uniformity              │
             │   - Pen-Lift Count & Spatial Boundary Drift          │
             └──────────────────────────┬───────────────────────────┘
                                        │
                                        ▼
             ┌──────────────────────────────────────────────────────┐
             │       Layer 2: Geometric & Shape Recognition         │
             │   - $P Point-Cloud / $1 Dollar Gesture Recognizer    │
             │   - Vertex & Angle Extraction (Triangle 3, Square 4) │
             │   - Aspect Ratio & Circularity Metric (4πA / P²)     │
             │   - Alphabet Stroke Template Matching (A, B, C...)   │
             └──────────────────────────┬───────────────────────────┘
                                        │
                                        ▼
             ┌──────────────────────────────────────────────────────┐
             │     Layer 3: Neural Scoring & Clinical Diagnosis     │
             │   - Integration into `ml_models/dysgraphia_nn_model` │
             │   - Sub-scores: Motor Control, Legibility, Stamina   │
             │   - Risk Level: None / Low / Medium / High           │
             └──────────────────────────────────────────────────────┘
```

### 3.3 Target Test Evaluations

1. **Straight Line Test (`trace_line`)**:
   - **Target**: Follow a horizontal or diagonal guide line.
   - **Evaluated Biomarkers**:
     - *Orthogonal Deviation ($RMSE$)*: Average perpendicular distance from the ideal vector.
     - *Tremor Index*: Frequency and amplitude of directional oscillations (hand shakiness).
     - *Pass Threshold*: $RMSE < 12px$, Tremor stability $> 80\%$.

2. **Geometric Shape Test (`shape_draw`)**:
   - **Target**: Draw a Triangle, Square, and Circle.
   - **Evaluated Biomarkers**:
     - *Vertex Detection*: Douglas-Peucker polygon simplification to identify corner counts (Triangle = 3, Square = 4).
     - *Interior Angle Check*: Verifies triangles have $\approx 60^\circ$ angles and squares have $\approx 90^\circ$ angles.
     - *Circularity Index*: $C = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$ (Circle $\approx 1.0$).
     - *Closure Gap*: Distance between start and end stroke points.

3. **Letter Formation & Copying (`copy_letter`)**:
   - **Target**: Copy target letters (e.g., 'A', 'B', 'P', 'R').
   - **Evaluated Biomarkers**:
     - Point-cloud structural similarity to reference font glyphs.
     - Stroke count and sequence alignment.
     - Spatial scale consistency and baseline alignment.

---

## 4. Charts & Graphical Cognitive Analysis

### 4.1 Assessment Results Pages (`dyslexia_results.html`, `dyscalculia_results.html`, `dysgraphia_results.html`)

Currently, results display textual numbers, badges, and plain percentage bars. We will integrate **Chart.js** with Antigravity dark styling (`#08090B`, `#12151A`, neon purple `#7C3AED`, emerald `#10B981`, amber `#F59E0B`, rose `#EF4444`).

#### Visualizations to Add:
1. **Cognitive Domain Radar Chart**:
   - *Dyslexia*: Phoneme Awareness, Sound Retrieval, Rhyme Recognition, Lexical Decision, Rapid Fluency, Working Memory.
   - *Dyscalculia*: Magnitude Comparison, Arithmetic Fluency, Sequencing, Mental Calculation, Spatial Math.
   - *Dysgraphia*: Motor Smoothness, Line Straightness, Tremor Control, Letter Legibility, Speed Consistency.
   - **Value**: Instantly highlights the student's exact cognitive bottlenecks (e.g., "High vocabulary, but severe struggle in Rapid Naming Retrieval").

2. **Subtest Performance & Speed Comparison (Bar / Grouped Chart)**:
   - Compares the student's accuracy (%) and reaction time ($ms$) against the age-norm baseline for each sub-game.
   - Color-coded: Emerald (Mastered/Passed), Amber (Borderline/Slow), Rose (Struggled/Failed).

3. **Cognitive Struggle Diagnostic Card**:
   - Explicit clinical diagnostic badges:
     - ⚠️ *Struggling Areas*: e.g., "Phonological Retrieval Speed (1820ms vs 800ms norm)", "Fine Motor Tremor during angular stroke transitions".
     - ✅ *Preserved Strengths*: e.g., "High Lexical Accuracy (95%)", "Strong Rhyme Recognition".

### 4.2 Student Profile (`student-profile.html`)

In `student-profile.html`, we will add **executive, clean, limited graphical analysis** so the profile remains focused and elegant without clutter:

1. **Executive Disorder Risk Radar Chart (3-Axis)**:
   - Displays Risk Percentage across all three conditions:
     - **Dyslexia Risk %**
     - **Dyscalculia Risk %**
     - **Dysgraphia Risk %**
   - Shows at a glance if a student has isolated difficulty or co-occurring (comorbid) learning differences.

2. **Cognitive Domain Health Meter**:
   - A multi-segmented horizontal progress bar / mini-doughnut showing overall cognitive health (Low Risk / Moderate / Attention Needed).

---

## 5. Deployment Protocol via Vercel MCP Server

As requested by the user:
> *"took in mind that i need to deploy it on vercel i added vercel mcp cerver you just upload it as new project and make it live after everything is planned and working it should be last step after conferming all on local"*

### Deployment Workflow:
1. **Local Implementation**: Complete all models, charts, and routes locally.
2. **Local End-to-End Testing**: Run the complete automated test suite (`pytest` / Python verification scripts) ensuring:
   - `python app.py` starts cleanly.
   - Voice games, dysgraphia canvas recognition, and chart rendering pass with HTTP 200.
   - Zero unbundled heavy dependencies (bundle size strictly within Vercel's 250MB limit).
3. **Vercel Project Setup via MCP**:
   - Use `mcp_vercel_create_project` to create a dedicated project for `neurobloom` / `neurobloomv2`.
   - Set environment variables via `mcp_vercel_create_project_env` (`SUPABASE_URL`, `SUPABASE_ANON_KEY`, `FLASK_SECRET_KEY`, `DATABASE_URL`).
   - Trigger production deployment and verify the live public URL returns HTTP 200 without 404 or runtime errors.

---

## 6. Implementation Milestones

| Milestone | Deliverables | Verification Method |
| :--- | :--- | :--- |
| **Phase 1: Architecture Approval** | Complete `ARCHITECTURE_AND_ENHANCEMENT_PLAN.md` (This document) | User review & approval |
| **Phase 2: Dysgraphia Shape Recognition Engine** | Geometric recognizer + kinematic tremor/straightness pipeline in `dysgraphia_test.html` and `ml_models/dysgraphia_nn_model.py` | Automated shape stroke test script (`test_dysgraphia_recognition.py`) |
| **Phase 3: Offline / Low-Latency Voice Engine** | In-browser zero-latency STT engine with offline caching and microphone safety locks | Automated speech engine verification test |
| **Phase 4: Graphical Cognitive Analysis & Charts** | Chart.js radar charts and struggle diagnostic bars in all 3 results templates + executive profile chart | Browser testing & route verification |
| **Phase 5: Local Validation & Regression Testing** | Verify full assessment flow, Supabase persistence, and Antigravity styling | Complete test suite run locally |
| **Phase 6: Vercel MCP Deployment** | Create new project, configure env vars, deploy, and verify live URL | Vercel MCP deployment check |
