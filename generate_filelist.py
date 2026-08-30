"""Generate a CSV listing all filenames under data/ — used for DVC versioning in Q3."""
import os
import csv

def main():
    rows = []
    for root, _, files in os.walk("data"):
        for f in sorted(files):
            rows.append(os.path.relpath(os.path.join(root, f), "data"))

    with open("filenames.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"])
        for r in sorted(rows):
            writer.writerow([r])

    print(f"Wrote {len(rows)} rows to filenames.csv")

if __name__ == "__main__":
    main()
