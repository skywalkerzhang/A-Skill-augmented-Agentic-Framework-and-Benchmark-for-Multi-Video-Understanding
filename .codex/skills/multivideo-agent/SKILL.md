---
name: multivideo-agent
description: General orchestration guardrails for multi-video tasks. Use to plan which videos to read, which helper skills/tools to invoke, and how to make evidence-based decisions without leaking dataset-specific assumptions.
---

# Multivideo Agent

Use this skill to plan multi-video reading and evidence-based decisions.

## Core rules
- Identify decision type first (similarity, count, spatial, speech).
- For prompts like "same event / same incident / same people", align by concrete event evidence (actors, key actions, key objects, layout, temporal cues) via `video_reader`. Similarity scores can be influenced by viewpoint/lighting, so treat `visual_similarity` as supporting evidence or a tie-break.
- **Counting**: never treat "not visible" in a short window as count=0. If the target is not clearly present, treat it as **unknown** and scan forward in additional windows. Prefer `scene_graph` (and tune prompts / split windows) or `object_tracker` to confirm counts. Before answering, align `ref/A/B/C/D` by **confirmed** counts; do not finalize with unknowns.
- **Counting (cost control)**: avoid chopping the video into many small windows. Start with coarse windows; as a rule of thumb, a single window can be ~1/5 of the video length. In early turns, keep the number of segments small (e.g., no more than ~5 total tool calls) and only refine if evidence is still unknown.
- **Nested / per-video sub-questions**: if the task embeds a separate sub-question for each video with two alternatives (e.g., "Option 1" vs "Option 2"), first decide which alternative is correct **for each video** from its observed outcome, then choose the candidate whose alternative index matches `ref`. Do not choose by scene similarity.
- Read only required videos/tools; avoid redundant calls.
- Keep questions consistent across candidates.
- Respect view↔video mapping; never assume A/B/C order.
- Similarity tasks: `visual_similarity` first, `video_reader` only for targeted verification.
- Output concise final answer; avoid long explanations.
