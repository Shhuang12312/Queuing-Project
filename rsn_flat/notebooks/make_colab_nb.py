"""Regenerate the %%writefile cells of rsn_colab_flat.ipynb from the current rsn_flat/*.py.

    python rsn_flat/notebooks/make_colab_nb.py

Each module cell is `%%writefile <name>.py` followed by the file's exact text, so the
notebook always carries the current code. Run after any change to
rsn.py, offline.py, je_stats.py, policy.py, simulation.py, je.py or check.py.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT = os.path.dirname(HERE)
NB = os.path.join(HERE, "rsn_colab_flat.ipynb")
FILES = ["rsn.py", "offline.py", "je_stats.py", "policy.py", "simulation.py", "je.py", "check.py"]


def to_source(text):
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines[:-1] + [lines[-1].rstrip("\n")] if lines else []


def main():
    nb = json.load(open(NB))
    done = set()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        first = src.splitlines()[0] if src else ""
        if first.startswith("%%writefile "):
            name = first.split()[1]
            if name in FILES:
                body = open(os.path.join(FLAT, name)).read()
                cell["source"] = to_source(f"%%writefile {name}\n" + body)
                cell["outputs"], cell["execution_count"] = [], None
                done.add(name)
    missing = [f for f in FILES if f not in done]
    if missing:
        raise SystemExit(f"no %%writefile cell found for {missing}")
    json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
    open(NB, "a").write("\n")
    print("regenerated", sorted(done))


if __name__ == "__main__":
    main()
