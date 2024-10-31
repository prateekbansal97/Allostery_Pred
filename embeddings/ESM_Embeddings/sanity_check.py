from collections import defaultdict

fasta_file = "GPCRs.fasta"
labels = defaultdict(int)

with open(fasta_file, "r") as f:
    for line in f:
        if line.startswith(">"):
            label = line.strip()
            labels[label] += 1

# Print duplicate labels
duplicates = {label: count for label, count in labels.items() if count > 1}
if duplicates:
    print("Duplicate sequence labels found:")
    for label, count in duplicates.items():
        print(f"{label}: {count} times")
else:
    print("No duplicate sequence labels found.")

