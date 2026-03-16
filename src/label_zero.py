import os

# ========================= # PROCESS FILES # =========================


def set_lebal_zero(labels_folder):
    """
    all classes to one class
    """

    for filename in os.listdir(labels_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_folder, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            updated_lines = []

            for line in lines:
                parts = line.strip().split()

                if len(parts) >= 5:
                    parts[0] = "0"  # Replace class id
                    updated_lines.append(" ".join(parts) + "\n")
                else:
                    updated_lines.append(line)

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(updated_lines)

            print(f"Updated: {filename}")

    print("✅ All class IDs replaced with 0")
