---
name: multi-video-compare
description: Compare ref vs candidates by reading all videos consistently and deciding after evidence.
---

# Multi-Video Compare Skill

Use for ref vs candidates matching.

## Rules
- Read `ref` first, then candidates.
- Ask comparable questions across candidates.
- If the question explicitly asks about look/style/appearance or “most similar”, call `visual_similarity` before final answer (CLIP-score works best when viewpoint is similar). For “same event” style prompts, prioritize `video_reader` event alignment and use `visual_similarity` only as a tie-break if needed.
- Do not finalize from weak or partial evidence.
- Keep responses short and tool-driven.

## Similarity Default Pattern
1. Run `visual_similarity` for `ref` vs candidates.
2. If top scores are close, do targeted `video_reader` checks.
3. Output final `<answer>X</answer>` with no extra text.
