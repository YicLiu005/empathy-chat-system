# RAG-only Annotation Results in Human Annotation Format

This file reports RAG-only analysis results using a format similar to the simulated human annotation study. Each case is treated as one multi-turn conversation. The RAG system predicts the overall emotional trajectory and the main user support need.

## Label Set

### Emotional Trajectory Labels

| Label | Meaning |
|---|---|
| `improving` | Emotion becomes more positive or more constructive across turns. |
| `planning_progress` | User moves toward concrete goals, planning, or next steps. |
| `stable` | Emotion remains mostly unchanged across turns. |
| `fluctuating` | Emotion changes back and forth without a clear direction. |
| `persistent_negative` | Negative emotion remains across multiple turns. |
| `persistent_negative_escalating` | Negative emotion remains and becomes stronger over time. |
| `worsening` | The latest turns show more distress than earlier turns. |
| `high_risk_worsening` | User expresses severe distress that may require safety-aware support. |
| `unknown` | The emotional trajectory cannot be clearly determined. |

### User Need Labels

| Label | Meaning |
|---|---|
| `emotional_validation` | User mainly needs their feelings to be recognized and understood. |
| `reassurance` | User needs calming encouragement or confidence support. |
| `practical_guidance` | User wants concrete advice or a solution. |
| `emotional_regulation` | User needs help calming down before solving the problem. |
| `action_planning` | User wants structured next steps or a plan. |
| `clarification` | User need is unclear and requires a follow-up question. |
| `unknown` | The user need cannot be clearly determined. |

## Overall RAG Evaluation Summary

| Metric | Value |
|---|---:|
| Number of valid cases | 8 |
| Trajectory accuracy | 0.875 |
| Need accuracy | 0.875 |
| Error cases | 0 |

## Case-level RAG Summary

| Case | Title | Gold Trajectory | RAG Trajectory | Correct | Gold Need | RAG Need | Correct | Trajectory Confidence |
|---|---|---|---|---:|---|---|---:|---:|
| C01 | Positive Planning after Project Feedback | `planning_progress` | `stable` | False | `action_planning` | `action_planning` | True | 0.93 |
| C02 | Breakup and Emotional Escalation | `persistent_negative_escalating` | `persistent_negative_escalating` | True | `emotional_regulation` | `emotional_regulation` | True | 0.96 |
| C03 | Ambiguous Study Planning | `planning_progress` | `planning_progress` | True | `action_planning` | `action_planning` | True | 0.96 |
| C04 | Sarcastic Product Complaint | `persistent_negative` | `persistent_negative` | True | `practical_guidance` | `practical_guidance` | True | 0.94 |
| C05 | Interview Anxiety and Reassurance | `persistent_negative` | `persistent_negative` | True | `reassurance` | `reassurance` | True | 0.96 |
| C06 | Overwhelmed but Starting to Plan | `planning_progress` | `planning_progress` | True | `action_planning` | `action_planning` | True | 0.94 |
| C07 | Low Motivation and Boredom | `persistent_negative` | `persistent_negative` | True | `emotional_validation` | `emotional_validation` | True | 0.94 |
| C08 | Recovery after Initial Anxiety | `improving` | `improving` | True | `practical_guidance` | `action_planning` | False | 0.94 |

---

## C01: Positive Planning after Project Feedback

### Multi-turn Conversation

1. User: I feel proud because I finally finished my project.
2. User: My professor gave me some positive feedback.
3. User: I feel more motivated to keep improving it.
4. User: I want to plan my next steps and make it even better.

### Gold Answers

- Gold emotional trajectory: `planning_progress`
- Gold user need: `action_planning`

### RAG System Annotation

| Annotator | Trajectory Label | User Need Label | Confidence | Emotional Trajectory Label Meaning |
|---|---|---|---:|---|
| RAG_System | `stable` | `action_planning` | 0.93 | Emotion remains mostly unchanged across turns. |

### RAG Need Transition

`emotional_validation -> emotional_validation -> emotional_validation -> action_planning`

### RAG Summaries

**Trajectory summary:** The user's recent emotional trajectory appears to be stable. The current emotion is positive, the current intensity is 0.45, and the current support need is action_planning.

**Need summary:** The user's support need changes across turns as: emotional_validation → emotional_validation → emotional_validation → action_planning. The overall current support need appears to be action_planning.

---

## C02: Breakup and Emotional Escalation

### Multi-turn Conversation

1. User: I just broke up with my girlfriend. I feel terrible.
2. User: I don't know what to do now.
3. User: My life feels so empty.
4. User: I can't calm down. Everything feels too much.

### Gold Answers

- Gold emotional trajectory: `persistent_negative_escalating`
- Gold user need: `emotional_regulation`

### RAG System Annotation

| Annotator | Trajectory Label | User Need Label | Confidence | Emotional Trajectory Label Meaning |
|---|---|---|---:|---|
| RAG_System | `persistent_negative_escalating` | `emotional_regulation` | 0.96 | Negative emotion remains and becomes stronger over time. |

### RAG Need Transition

`emotional_validation -> practical_guidance -> emotional_validation -> emotional_regulation`

### RAG Summaries

**Trajectory summary:** The user has shown repeated negative emotions, and the latest turn suggests stronger emotional intensity or a higher need for emotional regulation. The assistant should slow down, validate the feeling, and provide calm grounding support.

**Need summary:** The user's support need changes across turns as: emotional_validation → practical_guidance → emotional_validation → emotional_regulation. The overall current support need appears to be emotional_regulation.

---

## C03: Ambiguous Study Planning

### Multi-turn Conversation

1. User: I want to plan my next steps and make it better.
2. User: Maybe on study.
3. User: I want to focus more and improve my learning habits.
4. User: Can you help me make a simple plan for tomorrow?

### Gold Answers

- Gold emotional trajectory: `planning_progress`
- Gold user need: `action_planning`

### RAG System Annotation

| Annotator | Trajectory Label | User Need Label | Confidence | Emotional Trajectory Label Meaning |
|---|---|---|---:|---|
| RAG_System | `planning_progress` | `action_planning` | 0.96 | User moves toward concrete goals, planning, or next steps. |

### RAG Need Transition

`action_planning -> action_planning -> action_planning -> action_planning`

### RAG Summaries

**Trajectory summary:** The user appears to be refining a plan rather than becoming more negative. The assistant should treat this as planning progress and help turn the idea into concrete next steps.

**Need summary:** The user's support need changes across turns as: action_planning → action_planning → action_planning → action_planning. The overall current support need appears to be action_planning.

---

## C04: Sarcastic Product Complaint

### Multi-turn Conversation

1. User: Great, another update that breaks everything.
2. User: I just love waiting forever for this app to load.
3. User: Now I cannot even finish my work on time.
4. User: What should I do now?

### Gold Answers

- Gold emotional trajectory: `persistent_negative`
- Gold user need: `practical_guidance`

### RAG System Annotation

| Annotator | Trajectory Label | User Need Label | Confidence | Emotional Trajectory Label Meaning |
|---|---|---|---:|---|
| RAG_System | `persistent_negative` | `practical_guidance` | 0.94 | Negative emotion remains across multiple turns. |

### RAG Need Transition

`emotional_validation -> emotional_validation -> emotional_validation -> practical_guidance`

### RAG Summaries

**Trajectory summary:** The user has shown negative emotion across multiple recent turns. The emotional state is not clearly improving yet, so the assistant should continue offering validation and gentle support.

**Need summary:** The user's support need changes across turns as: emotional_validation → emotional_validation → emotional_validation → practical_guidance. The overall current support need appears to be practical_guidance.

---

## C05: Interview Anxiety and Reassurance

### Multi-turn Conversation

1. User: I am worried about my interview.
2. User: I keep thinking I might fail.
3. User: I prepared, but I still feel nervous.
4. User: Can you reassure me a little?

### Gold Answers

- Gold emotional trajectory: `persistent_negative`
- Gold user need: `reassurance`

### RAG System Annotation

| Annotator | Trajectory Label | User Need Label | Confidence | Emotional Trajectory Label Meaning |
|---|---|---|---:|---|
| RAG_System | `persistent_negative` | `reassurance` | 0.96 | Negative emotion remains across multiple turns. |

### RAG Need Transition

`reassurance -> reassurance -> reassurance -> reassurance`

### RAG Summaries

**Trajectory summary:** The user has shown negative emotion across multiple recent turns. The emotional state is not clearly improving yet, so the assistant should continue offering validation and gentle support.

**Need summary:** The user's support need changes across turns as: reassurance → reassurance → reassurance → reassurance. The overall current support need appears to be reassurance.

---

## C06: Overwhelmed but Starting to Plan

### Multi-turn Conversation

1. User: I feel really overwhelmed by all my deadlines.
2. User: I don't know where to start.
3. User: Maybe I should make a list first.
4. User: Can you help me choose the first step?

### Gold Answers

- Gold emotional trajectory: `planning_progress`
- Gold user need: `action_planning`

### RAG System Annotation

| Annotator | Trajectory Label | User Need Label | Confidence | Emotional Trajectory Label Meaning |
|---|---|---|---:|---|
| RAG_System | `planning_progress` | `action_planning` | 0.94 | User moves toward concrete goals, planning, or next steps. |

### RAG Need Transition

`emotional_regulation -> action_planning -> action_planning -> action_planning`

### RAG Summaries

**Trajectory summary:** The user appears to be refining a plan rather than becoming more negative. The assistant should treat this as planning progress and help turn the idea into concrete next steps.

**Need summary:** The user's support need changes across turns as: emotional_regulation → action_planning → action_planning → action_planning. The overall current support need appears to be action_planning.

---

## C07: Low Motivation and Boredom

### Multi-turn Conversation

1. User: My life feels so boring lately.
2. User: I don't feel excited about anything.
3. User: I just scroll on my phone for hours.
4. User: I wish I could find something meaningful to do.

### Gold Answers

- Gold emotional trajectory: `persistent_negative`
- Gold user need: `emotional_validation`

### RAG System Annotation

| Annotator | Trajectory Label | User Need Label | Confidence | Emotional Trajectory Label Meaning |
|---|---|---|---:|---|
| RAG_System | `persistent_negative` | `emotional_validation` | 0.94 | Negative emotion remains across multiple turns. |

### RAG Need Transition

`emotional_validation -> emotional_validation -> emotional_validation -> emotional_validation`

### RAG Summaries

**Trajectory summary:** The user has shown negative emotion across multiple recent turns. The emotional state is not clearly improving yet, so the assistant should continue offering validation and gentle support.

**Need summary:** The user's support need changes across turns as: emotional_validation → emotional_validation → emotional_validation → emotional_validation. The overall current support need appears to be emotional_validation.

---

## C08: Recovery after Initial Anxiety

### Multi-turn Conversation

1. User: I felt really anxious about my presentation yesterday.
2. User: I practiced again and now I feel a little better.
3. User: I still worry that I might forget something.
4. User: Can you help me review the key points one more time?

### Gold Answers

- Gold emotional trajectory: `improving`
- Gold user need: `practical_guidance`

### RAG System Annotation

| Annotator | Trajectory Label | User Need Label | Confidence | Emotional Trajectory Label Meaning |
|---|---|---|---:|---|
| RAG_System | `improving` | `action_planning` | 0.94 | Emotion becomes more positive or more constructive across turns. |

### RAG Need Transition

`emotional_validation -> emotional_validation -> reassurance -> action_planning`

### RAG Summaries

**Trajectory summary:** The user's emotional state appears to be improving. The assistant can acknowledge the progress and help the user continue moving forward.

**Need summary:** The user's support need changes across turns as: emotional_validation → emotional_validation → reassurance → action_planning. The overall current support need appears to be action_planning.
