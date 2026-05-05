# Empathy Chat System

Empathy Chat System is a Streamlit-based application for multi-turn emotion tracking, user-need detection, and empathetic dialogue generation.

## 1. Install Dependencies

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment.

On Windows PowerShell:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install streamlit requests
```

If a `requirements.txt` file is provided, install dependencies with:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` should include:

```text
streamlit
requests
```

Set your Gemini API key before running the system.

On Windows PowerShell:

```bash
$env:GEMINI_API_KEY="your_api_key_here"
```

On macOS/Linux:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

## 2. Run the System

Run the Streamlit web interface:

```bash
streamlit run web_ui.py
```

After running the command, Streamlit will open the system in your browser.

The system provides three modes:

```text
RAG Multi-turn Emotion Tracking
No-RAG Multi-turn Emotion Tracking
Empathy Chat
```

## 3. Example Usage

### Example 1: RAG Multi-turn Emotion Tracking

Select `RAG Multi-turn Emotion Tracking`.

Enter a multi-turn conversation, with one user message per line:

```text
I just broke up with my girlfriend. I feel terrible.
I don't know what to do now.
My life feels so empty.
I can't calm down. Everything feels too much.
```

Click `Analyze Multi-turn Conversation`.

Example output:

```text
Overall Emotional Trajectory: Persistent Negative Escalating
Trajectory Confidence: 0.92

Overall Support Need: Emotional Regulation
Need Transition: emotional_validation → practical_guidance → emotional_validation → emotional_regulation
```

```

### Example 2: Empathy Chat

Select `Empathy Chat`.

Enter a message in the chat box:

```text
I am so worried that I will fail this interview.
```

Example assistant response:

```text
It is understandable to feel worried before an important interview. You have already started preparing, so a good next step is to review your key project experiences and practice answering common questions out loud.
```

The system also displays turn-level analysis:

```text
Current Emotion: Negative
Detected User Need: Reassurance
Emotional Trajectory: Initial
```

## 4. Run Evaluation

To compare RAG and No-RAG performance, run:

```bash
python evaluate_tracking.py
```

To use the small evaluation set, set this in `evaluate_tracking.py`:

```python
CASES_FILE = "evaluation_cases_20.json"
```

The evaluation script outputs table-based results such as:

```text
Trajectory Accuracy
Need Accuracy
Avg Confidence When Trajectory Correct
Avg Confidence When Trajectory Wrong
```
