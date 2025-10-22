import json
from pathlib import Path


NB_PATH = Path("predicting_demand_for_perishable_goodipynb.ipynb")


def load_nb(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and "cells" in data, "Invalid notebook file"
    return data


def save_nb(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def replace_content_paths(cell):
    if cell.get("cell_type") != "code":
        return False
    src = "".join(cell.get("source", []))
    if "/content/" in src:
        src = src.replace("/content/", "data/")
        cell["source"] = [line + ("\n" if not line.endswith("\n") else "") for line in src.splitlines()]
        return True
    return False


def append_modeling_cells(nb: dict):
    # Check if a modeling cell already exists
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", [])).lower()
            if "baseline_model" in src:
                return False
    md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Baseline Modeling\n",
            "Run the baseline pipeline directly from this notebook.\n",
        ],
    }
    code = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Execute the baseline script or import and call main()\n",
            "import sys\n",
            "sys.path.append('src')\n",
            "import baseline_model as bm\n",
            "bm.main()\n",
        ],
    }
    # Viewer cell to show saved predictions head
    viewer_exists = any(
        cell.get("cell_type") == "code" and "validation_predictions.csv" in "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
    )
    viewer = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# View the first rows of saved predictions\n",
            "import pandas as pd, os\n",
            "p = 'outputs/validation_predictions.csv'\n",
            "if os.path.exists(p):\n",
            "    display(pd.read_csv(p).head())\n",
            "else:\n",
            "    print('File not found:', p)\n",
        ],
    }
    nb["cells"].append(md)
    nb["cells"].append(code)
    if not viewer_exists:
        nb["cells"].append(viewer)
    return True


def append_merged_loader(nb: dict):
    # Avoid duplicate insertion
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "data/merged_dataset.csv" in src:
                return False

    md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Load Merged Dataset\n",
            "This cell loads `data/merged_dataset.csv` produced by `src/merge_all_datasets.py`.\n",
        ],
    }
    code = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "merged_path = 'data/merged_dataset.csv'\n",
            "df = pd.read_csv(merged_path)\n",
            "if 'WeekDate' in df.columns:\n",
            "    df['WeekDate'] = pd.to_datetime(df['WeekDate'], errors='coerce')\n",
            "display(df.head())\n",
        ],
    }

    # Insert after the first pandas import cell if present; else at start
    insert_idx = 0
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code" and "import pandas as pd" in "".join(cell.get("source", [])):
            insert_idx = i + 1
            break
    nb["cells"][insert_idx:insert_idx] = [md, code]
    return True


def main():
    if not NB_PATH.exists():
        print("Notebook not found:", NB_PATH)
        return 1
    nb = load_nb(NB_PATH)
    changes = 0
    for cell in nb.get("cells", []):
        if replace_content_paths(cell):
            changes += 1
    appended_model = append_modeling_cells(nb)
    appended_merged = append_merged_loader(nb)
    if changes or appended_model or appended_merged:
        save_nb(NB_PATH, nb)
        print(
            f"Updated notebook: {changes} cell(s) modified, modeling cell appended: {appended_model}, merged loader appended: {appended_merged}"
        )
    else:
        print("No changes needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
