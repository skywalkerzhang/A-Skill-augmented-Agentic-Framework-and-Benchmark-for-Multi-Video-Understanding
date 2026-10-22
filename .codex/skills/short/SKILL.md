---
name: short
description: Lightweight tool-selection policy for short prompt mode.
---

# Short Skill

Use this skill in short mode to choose tools quickly and avoid unnecessary calls.

## Goal
- Minimize turns and token usage.
- Prefer one focused tool call per turn.
- Answer as soon as evidence is sufficient.
- Resolve instruction conflicts consistently.

## Tool Selection Policy
1. Identify task family from the question text.
2. Default policy: prefer `video_reader` first.
3. Similarity-style tasks (`similar/style/look/appearance`) are the exception:
   - Primary signal: `visual_similarity`.
   - Start with pairwise comparisons against `ref` (for example `ref` vs `A/B/C/D`) on short aligned windows.
   - Use `video_reader` only for targeted verification when similarity scores are close, contradictory, or a clip is empty.
   - If another instruction explicitly forces "read ref first", do one short `video_reader` call on `ref`, then return to `visual_similarity`.
4. Use extra tools only for unresolved ambiguity:
   - `scene_graph` / `object_tracker`: object presence or counting ambiguity
   - `spatial_relation`: left/right/relative-position ambiguity
5. Do not repeat the same tool call on the same range with the same question.

## Conflict Resolution
- Priority order:
  1) Output-format and safety constraints
  2) Explicit hard task constraints (for example required first read)
  3) This skill's default policy
- When constraints conflict, satisfy higher priority with the minimum extra call, then return to the default policy.

## Output Discipline
- Each turn output either:
  - one tool call + `[Pause]`, or
  - final `<answer>X</answer>`
- No long analysis text.

## Useful Question Templates
- `What is the main action and key object?`
- `Which object appears/disappears in this segment?`
- `Where is OBJ1 relative to OBJ2?`

