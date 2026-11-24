from pathlib import Path

def _scalar(text):
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return [_scalar(x) for x in text[1:-1].split(",") if x.strip()]
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text

def load_config(path):
    """Parse the intentionally small YAML subset used by this isolated audit."""
    root = {}
    stack = [(-1, root)]
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, value = line.strip().split(":", 1)
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key] = _scalar(value)
        else:
            parent[key] = {}
            stack.append((indent, parent[key]))
    return root
