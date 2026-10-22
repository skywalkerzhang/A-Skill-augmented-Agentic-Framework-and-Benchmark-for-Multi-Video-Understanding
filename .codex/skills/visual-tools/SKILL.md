---
name: visual-tools
description: Lightweight visual tools (scene/ocr/track/similarity).
---

# Visual Tools Skill

Use for ranking and disambiguation in visual comparison.

## Rules
- `visual_similarity` is a CLIP-score tool. It is strongest for: overall look/style, same setting/scene composition, and “most similar” ranking when viewpoint is similar.
- For prompts like "same event / same incident", start from `video_reader` to align event evidence (actors, key actions, key objects, temporal order). If needed, use `visual_similarity` only as supporting evidence or a tie-break on **aligned short windows** around the key moment.
- For appearance/style/look questions, run at least one `visual_similarity` call before answering, then use `video_reader` only for targeted verification if scores are close or contradictory.
- Avoid full-sweep `video_reader` unless necessary.
- Keep outputs concise; no long reasoning text.

## Scene graph for counting (practical tuning)
Use `scene_graph` when you need **grounded** object presence/counting and `video_reader` is vague.

Guidelines:
- Treat `"not visible"` as **unknown**, not zero — adjust your tool call instead of finalizing.
- Prefer **shorter windows** (split long ranges) to reduce missed detections.
- Tune `prompts=` deliberately:
  - Use the exact target noun(s) from the question.
  - Add close synonyms/variants (comma-separated), e.g. `prompts=bag,grocery bag,shopping bag`.
  - If the target is a person, prefer `object_tracker target=person` for peak counts across frames.
- If counts are unstable, cross-check:
  - `scene_graph` for presence/layout in a tight window
  - `object_tracker` for peak/temporal consistency

## Tools
- `<scene_detector id="VIDEO_ID">fps=1;threshold=0.6</scene_detector>`
- `<object_tracker id="VIDEO_ID">start:end;target=person;fps=2;conf=0.25</object_tracker>`
- `<spatial_relation id="VIDEO_ID">start:end;targets=person,table;fps=2;conf=0.25</spatial_relation>`
- `<scene_graph id="VIDEO_ID">start:end;prompts=person,table;model=groundingdino</scene_graph>`
- `<visual_similarity id="A,B">a=0:10;b=20:30;fps=2;model=clip</visual_similarity>`

## Suggested Similarity Workflow
1. Run `visual_similarity` to get initial ranking.
2. If top-1 is clearly above top-2, answer.
3. Exception only: if ranking is close, run one focused `video_reader` check on tied candidates.
   Do not use `video_reader` as a default step for similarity questions.
4. Then output final answer.

## Scope Clarification
- Only similarity-style tasks can skip `video_reader` as the first step.
- Non-similarity tasks should still start with `video_reader`.
