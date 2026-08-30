import re
from dataclasses import dataclass


FENCED_CODE_RE = re.compile(r"```(?P<language>[\w+#.-]*)\s*\n(?P<code>.*?)```", re.DOTALL)
CODE_START_RE = re.compile(
    r"^\s*(?:#include\b|using\s+namespace\b|class\s+\w+|public:|private:|protected:|"
    r"def\s+\w+\s*\(|async\s+def\s+\w+\s*\(|fn\s+\w+\s*\(|function\s+\w+\s*\(|"
    r"(?:const|let|var)\s+\w+\s*=|package\s+[\w.]+\s*;|import\s+[\w.]+\s*;)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CodeArtifact:
    code: str
    programming_language: str | None
    instruction: str
    is_code_only: bool

    def prompt_descriptor(self) -> dict[str, object]:
        return {
            "present": True,
            "programming_language": self.programming_language,
            "line_count": len(self.code.splitlines()),
            "character_count": len(self.code),
            "content_policy": "原始代码由服务端注入 TaskSpec，模型不得复制代码内容",
        }


def extract_code_artifact(message: str) -> CodeArtifact | None:
    fenced_matches = list(FENCED_CODE_RE.finditer(message))
    if fenced_matches:
        largest = max(fenced_matches, key=lambda item: len(item.group("code")))
        code = largest.group("code").strip()
        instruction = FENCED_CODE_RE.sub(" ", message).strip()
        language = normalize_language(largest.group("language")) or detect_language(code)
        return CodeArtifact(code, language, instruction, not instruction)

    lines = message.strip().splitlines()
    if not lines:
        return None

    start_index = next((index for index, line in enumerate(lines) if CODE_START_RE.match(line)), None)
    if start_index is None:
        return None

    candidate = "\n".join(lines[start_index:]).strip()
    if not looks_like_code(candidate):
        return None

    instruction = "\n".join(lines[:start_index]).strip()
    return CodeArtifact(candidate, detect_language(candidate), instruction, not instruction)


def looks_like_code(value: str) -> bool:
    lines = value.splitlines()
    if re.search(r"(?m)^\s*(?:async\s+)?def\s+\w+\s*\(", value):
        indented_lines = sum(bool(re.match(r"^\s{2,}\S", line)) for line in lines[1:])
        if indented_lines >= 2:
            return True
    signals = 0
    signals += min(value.count("{"), 2)
    signals += min(value.count("}"), 2)
    signals += min(value.count(";"), 2)
    signals += int(bool(re.search(r"\b(?:for|while|if|return|class|def|fn|function)\b", value)))
    signals += int(bool(re.search(r"(?:\+\+|--|==|->|::|\[\w*\])", value)))
    return signals >= 4


def detect_language(code: str) -> str | None:
    checks = (
        ("C++", r"#include\s*<|\bstd::|\busing\s+namespace\s+std\b|(?m:^\s*(?:public|private|protected):\s*$)|\bvector\s*<|\bstring\s+\w+\s*\("),
        ("Java", r"\bpublic\s+(?:static\s+)?class\b|\bSystem\.out\.|\bimport\s+java\."),
        ("Python", r"(?m:^\s*(?:async\s+)?def\s+\w+\s*\()|(?m:^\s*(?:from\s+\w+\s+)?import\s+)"),
        ("Rust", r"\bfn\s+\w+\s*\(|\blet\s+mut\b|\bimpl\s+\w+"),
        ("TypeScript", r"\binterface\s+\w+|:\s*(?:string|number|boolean)\b|\btype\s+\w+\s*="),
        ("JavaScript", r"\b(?:const|let|var)\s+\w+\s*=|\bfunction\s+\w+\s*\(|=>"),
        ("Go", r"\bpackage\s+main\b|\bfunc\s+\w+\s*\("),
    )
    for language, pattern in checks:
        if re.search(pattern, code):
            return language
    return None


def normalize_language(value: str) -> str | None:
    normalized = value.strip().casefold()
    aliases = {
        "cpp": "C++",
        "c++": "C++",
        "cc": "C++",
        "java": "Java",
        "py": "Python",
        "python": "Python",
        "rs": "Rust",
        "rust": "Rust",
        "ts": "TypeScript",
        "typescript": "TypeScript",
        "js": "JavaScript",
        "javascript": "JavaScript",
        "go": "Go",
        "golang": "Go",
    }
    return aliases.get(normalized, value.strip() or None)
