"""Copy the assessment files into data/ using the file names the README/scorer expect.
Usage: python -m src.prepare_data <folder-or-zip-with-the-original-files>
"""
import shutil, sys, tempfile, zipfile
from pathlib import Path

NAMES = {"train-test.csv": "train_test.csv", "validation.csv": "validation.csv",
         "validation-predictions-template.csv": "validation_predictions_template.csv",
         "december-chart-inputs.csv": "december_chart_inputs.csv"}


def main(src: str) -> None:
    src_path = Path(src)
    if src_path.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp()); zipfile.ZipFile(src_path).extractall(tmp); src_path = tmp
    out = Path("data"); out.mkdir(exist_ok=True); found = 0
    for old, new in NAMES.items():
        hits = [h for h in list(src_path.rglob(old)) + list(src_path.rglob(new)) if "outputs" not in h.parts]
        if not hits:
            print(f"  missing: {old}"); continue
        found += 1
        if hits[0].resolve() == (out / new).resolve():
            print(f"  data/{new} already in place"); continue
        shutil.copy(hits[0], out / new); print(f"  {hits[0].name} -> data/{new}")
    sys.exit(0 if found == len(NAMES) else "some files were not found")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
