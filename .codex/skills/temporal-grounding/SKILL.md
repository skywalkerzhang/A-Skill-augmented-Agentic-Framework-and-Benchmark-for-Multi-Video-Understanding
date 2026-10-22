---
name: temporal-grounding
description: Locate relevant timestamps with temporal_grounding_agent and validate with video_reader before answering.
---

# Temporal Grounding Skill

Use to propose candidate timestamps; always verify with `video_reader`.

## Rules
- Query = entities + actions (+ options if needed).
- Treat timestamps as hints, not evidence.
- If no hits, scan sequential chunks.
