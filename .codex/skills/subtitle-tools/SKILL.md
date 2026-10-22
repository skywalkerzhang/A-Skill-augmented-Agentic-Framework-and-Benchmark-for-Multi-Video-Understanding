---
name: subtitle-tools
description: Use subtitle_retriever and subtitle_extractor when questions involve spoken content or text.
---

# Subtitle Tools Skill

Use only when text/speech is central.

## Rules
- Search first, then extract exact lines.
- Validate visually if ambiguous.

## Tools
- `<subtitle_retriever id="VIDEO_ID">query</subtitle_retriever>`
- `<subtitle_extractor id="VIDEO_ID">start:end</subtitle_extractor>`
