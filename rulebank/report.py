"""Report text: split into sentences keeping every other character, and render a revised text from sentence-level changes."""
import re

HEADER = re.compile(r"(?:(?<=\s)|^)(findings|impression)\s*:|^[ \t]*[A-Z][A-Z &/]+:[ \t]*$", re.I | re.M)


SENT = re.compile(r"(?:\d+\.[ \t]+)?\S.*?(?:\.(?=\s|$)|(?=\n)|$)", re.S)   # a list number "1. " opens a sentence


LABEL = re.compile(r"[A-Z][A-Za-z ,&/()-]{0,40}:(?=[ \t])")              # 'Pleura:' at the start of a line


def segment(text):
    """-> list of {"kind": "glue" | "sent" | "header" | "label", "text", "section": "findings" | "impression"}
    (+ "id" on sentences)"""
    pieces, pos, section = [], 0, "findings"

    def body(chunk, offset):
        p = 0
        for m in SENT.finditer(chunk):
            if m.start() > p:
                pieces.append({"kind": "glue", "text": chunk[p:m.start()], "section": section})
            s, a = m.group(0), offset + m.start()
            lab = LABEL.match(s) if a == 0 or text[a - 1] == "\n" else None
            if lab:
                rest = s[lab.end():]
                pieces.append({"kind": "label", "text": lab.group(0), "section": section})
                pieces.append({"kind": "glue", "text": rest[:len(rest) - len(rest.lstrip())], "section": section})
                s = rest.lstrip()
            pieces.append({"kind": "sent", "text": s, "section": section})
            p = m.end()
        if p < len(chunk):
            pieces.append({"kind": "glue", "text": chunk[p:], "section": section})

    for m in HEADER.finditer(text):
        body(text[pos:m.start()], pos)
        if re.match(r"\s*impression", m.group(0), re.I):
            section = "impression"
        pieces.append({"kind": "header", "text": m.group(0), "section": section})
        pos = m.end()
    body(text[pos:], pos)
    assert "".join(p["text"] for p in pieces) == text
    n = 0
    for p in pieces:
        if p["kind"] == "sent":
            n += 1
            p["id"] = n
    return pieces


LEAKED = re.compile(r"^\s*(?:\[\d+\]\s*)?(?:\((?:findings|impression)[^)]*\)\s*)?(?:\[\d+\]\s*)?", re.I)


TRAILING = re.compile(r"\s*\((?:findings|impression)[^)]*\)\s*$", re.I)    # "Atelectasis is present. (findings)"


def clean(sentence):
    """a new sentence without a leaked '[3]' / '(findings)' / '(findings, Pleura)' at its start or a section tag at its end (the editor copies the
    numbered() format although told not to: 126 / 185 of the new sentences in round 0 of SelfCue whole / focused,
    RJ 2026-10-23: strip in code, keep counting)"""
    return TRAILING.sub("", LEAKED.sub("", sentence, count=1)).strip()


def render(pieces, changes, adds):
    """changes: {id: ("remove", None) | ("rewrite", new sentence)}; adds: [sentence] -> revised text"""
    S = {p["id"]: p for p in pieces if p["kind"] == "sent"}
    assert set(changes) <= set(S), set(changes) - set(S)
    text = "".join(p["text"] for p in pieces)
    sep = "\n" if "\n" in text.strip() else " "
    out = []
    imp = next((i for i, p in enumerate(pieces) if p["kind"] == "header" and p["section"] == "impression"), None)
    for i, p in enumerate(pieces):
        if i == imp and adds:
            out.append(sep.join(adds) + sep)
        if p["kind"] == "sent" and p["id"] in changes:
            action, new = changes[p["id"]]
            if action == "remove":
                continue
            assert action == "rewrite" and new.strip(), (action, new)
            out.append(new.strip())
            continue
        out.append(p["text"])
    if imp is None and adds:
        out.append(sep + sep.join(adds))
    t = "".join(out)                                    # a removed sentence leaves its glue: tidy spaces, keep newlines
    t = re.sub(r"[ \t]+(?=\n)", "", re.sub(r"(?<=\n)[ \t]+", "", t))
    return re.sub(r"[ \t]{2,}", " ", t).strip()


def sents(text):
    return [p for p in segment(text) if p["kind"] == "sent"]


def numbered(pieces):
    """the sentences as the LLM sees them: '[3] (findings) Pleural effusion-thickening was not detected.', with the line's sub-section label when there is one"""
    L, label = [], None
    for p in pieces:
        if p["kind"] == "label":
            label = p["text"].rstrip(":")
        elif p["kind"] == "header" or (p["kind"] == "glue" and "\n" in p["text"]):
            label = None
        elif p["kind"] == "sent":
            L.append(f"[{p['id']}] ({p['section']}{', ' + label if label else ''}) {p['text'].strip()}")
    return "\n".join(L)


def delete_sentence(sid):
    return {"id": sid, "action": "remove", "new_sentence": ""}


def rewrite_sentence(sid, new):
    return {"id": sid, "action": "rewrite", "new_sentence": new}


def revise(draft, changes, adds):
    """changes: [delete_sentence(id) | rewrite_sentence(id, sentence)]; adds: [sentence] appended to the findings -> revised text"""
    ch = {x["id"]: (x["action"], clean(x["new_sentence"]) if x["action"] == "rewrite" else None) for x in changes}
    return render(segment(draft), ch, [clean(s) for s in adds])
