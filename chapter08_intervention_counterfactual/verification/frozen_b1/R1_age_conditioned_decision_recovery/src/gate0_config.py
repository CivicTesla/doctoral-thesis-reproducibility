"""Dependency-free loader for the small Gate 0/1 YAML subset."""

from pathlib import Path


def _scalar(text):
    text = text.strip()
    if not text:
        return None
    if text.startswith("{") and text.endswith("}"):
        output = {}
        for item in text[1:-1].split(","):
            key, value = item.split(":", 1)
            output[key.strip()] = _scalar(value)
        return output
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [] if not inner else [
            _scalar(item) for item in inner.split(",")]
    if ((text.startswith('"') and text.endswith('"'))
            or (text.startswith("'") and text.endswith("'"))):
        return text[1:-1]
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _minimal_yaml(text):
    lines = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            lines.append((len(line) - len(line.lstrip()), line.strip()))

    def parse(index, indent):
        is_list = lines[index][1].startswith("- ")
        output = [] if is_list else {}
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent != indent:
                raise ValueError("invalid YAML indentation")
            if is_list:
                if not content.startswith("- "):
                    break
                item = content[2:].strip()
                if item:
                    output.append(_scalar(item))
                    index += 1
                else:
                    index += 1
                    value, index = parse(index, lines[index][0])
                    output.append(value)
            else:
                if content.startswith("- "):
                    break
                if ":" not in content:
                    raise ValueError(f"invalid YAML mapping line: {content}")
                key, value = content.split(":", 1)
                index += 1
                if value.strip():
                    output[key.strip()] = _scalar(value)
                elif index < len(lines) and lines[index][0] > indent:
                    nested, index = parse(index, lines[index][0])
                    output[key.strip()] = nested
                else:
                    output[key.strip()] = {}
        return output, index

    if not lines:
        return {}
    value, consumed = parse(0, lines[0][0])
    if consumed != len(lines):
        raise ValueError("unparsed YAML content")
    return value


def load_gate0_config(path):
    value = _minimal_yaml(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Gate 0 config must be a mapping")
    return value
