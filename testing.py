import json

def safe_load_cells(ipynb_path):
    with open(ipynb_path, "r", encoding="utf-8", errors="ignore") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print("❌ Still not valid JSON — structure is likely broken.")
            return None

    good_cells = []
    for i, cell in enumerate(data.get("cells", [])):
        try:
            json.dumps(cell)
            good_cells.append(cell)
        except (TypeError, ValueError):
            print(f"⚠️ Skipped corrupted cell #{i}")

    data["cells"] = good_cells
    return data

recovered = safe_load_cells("proj1_clean.ipynb")

if recovered:
    with open("proj1.ipynb", "w", encoding="utf-8") as f:
        json.dump(recovered, f, indent=2)
    print("✅ Cleaned notebook saved as proj1_recovered.ipynb")

