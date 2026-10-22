import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    skill_dir: Path
    skill_md: Path


def _parse_front_matter(md_text: str) -> dict:
    lines = md_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta


def _resolve_base_dirs(base_dirs: Optional[Iterable[str]] = None) -> List[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    if base_dirs is None:
        env = os.getenv("SKILLS_DIRS")
        if env:
            base_dirs = [p for p in env.split(os.pathsep) if p]
        else:
            base_dirs = [
                ".codex/skills",
            ]
    resolved = []
    for p in base_dirs:
        path = Path(p)
        if not path.is_absolute():
            path = repo_root / path
        resolved.append(path)
    return resolved


def discover_skills(base_dirs: Optional[Iterable[str]] = None) -> List[SkillInfo]:
    skills = []
    for base in _resolve_base_dirs(base_dirs):
        if not base.exists():
            continue
        for md in list(base.rglob("SKILL.md")) + list(base.rglob("Skill.md")):
            try:
                text = md.read_text(encoding="utf-8")
            except OSError:
                continue
            meta = _parse_front_matter(text)
            name = meta.get("name") or md.parent.name
            description = meta.get("description", "").strip()
            skills.append(
                SkillInfo(
                    name=name,
                    description=description,
                    skill_dir=md.parent,
                    skill_md=md,
                )
            )
    skills.sort(key=lambda s: s.name.lower())
    return skills


def load_skill_markdown(base_dirs: Optional[Iterable[str]] = None) -> List[dict]:
    skills = discover_skills(base_dirs)
    results = []
    for s in skills:
        try:
            content = s.skill_md.read_text(encoding="utf-8")
        except OSError:
            continue
        results.append(
            {
                "name": s.name,
                "description": s.description,
                "skill_md": str(s.skill_md),
                "content": content,
            }
        )
    return results


def format_skills_for_prompt(base_dirs: Optional[Iterable[str]] = None) -> str:
    items = load_skill_markdown(base_dirs)
    parts = []
    for item in items:
        parts.append(f"## Skill: {item['name']}\n{item['content']}")
    return "\n\n".join(parts)


def skills_to_json(base_dirs: Optional[Iterable[str]] = None) -> str:
    skills = discover_skills(base_dirs)
    payload = [
        {
            "name": s.name,
            "description": s.description,
            "skill_dir": str(s.skill_dir),
            "skill_md": str(s.skill_md),
        }
        for s in skills
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print(skills_to_json())
