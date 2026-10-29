def split_words(text: str) -> list[str]:
    units: list[str] = []
    start = 0
    for word in text.split():
        end = text.index(word, start) + len(word)
        units.append(text[start:end])
        start = end
    return [*units, text[start:]] if start < len(text) else units


def cap_split(unit: str, max_bytes: int) -> list[str]:
    pieces: list[str] = []
    piece, size = "", 0
    for ch in unit:
        width = len(ch.encode())
        if piece and size + width > max_bytes:
            pieces.append(piece)
            piece, size = "", 0
        piece += ch
        size += width
    return [*pieces, piece] if piece else pieces


def to_units(text: str, max_bytes: int) -> list[str]:
    units = [
        piece for word in split_words(text) for piece in cap_split(word, max_bytes)
    ]
    if "".join(units) != text:
        raise ValueError("segmentation is not lossless")
    return units
