import json
import csv

# Path to your JSON file
input_file = "baltic-sea-grid-precise-medium.json"

# Path to output CSV file
output_file = "baltic_sea_cells.csv"

# Load JSON
with open(input_file, "r") as f:
    data = json.load(f)

cells = data["cells"]

# Write CSV
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "lat", "lon"])  # header

    for cell in cells:
        cell_id = cell["id"]
        lat = cell["center"]["lat"]
        lon = cell["center"]["lon"]
        writer.writerow([cell_id, lat, lon])

print(f"CSV file written: {output_file}")
