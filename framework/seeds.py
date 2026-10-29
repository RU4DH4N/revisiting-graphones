import hashlib


def seed_for(root_seed: int, *role: object) -> int:
    digest = hashlib.sha256("/".join(map(str, (root_seed, *role))).encode()).digest()
    return int.from_bytes(digest[:4], "big") >> 1
