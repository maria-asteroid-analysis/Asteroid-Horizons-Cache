from pathlib import Path

folder = Path(__file__).parent
output_file = folder / "csv_names.txt"

names = sorted(
    csv_file.stem.replace("_", " ")
    for csv_file in folder.glob("*.csv")
)

output_file.write_text("\n".join(names) + "\n", encoding="utf-8")
print(f"Created {output_file} with {len(names)} names.")
