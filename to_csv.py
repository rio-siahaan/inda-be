import os
import json
import csv

def extract_ids(key, vervar_keys, var_keys, turvar_keys, tahun_keys):
    for vervar in vervar_keys:
        vervar_str = str(vervar)
        if key.startswith(vervar_str):
            key_remaining = key[len(vervar_str):]

            for var in var_keys:
                var_str = str(var)
                if key_remaining.startswith(var_str):
                    key_remaining = key_remaining[len(var_str):]

                    for turvar in turvar_keys:
                        turvar_str = str(turvar)
                        if key_remaining.startswith(turvar_str):
                            key_remaining = key_remaining[len(turvar_str):]

                            for tahun in tahun_keys:
                                tahun_str = str(tahun)
                                if key_remaining.startswith(tahun_str) and key_remaining.endswith("0"):
                                    return vervar, var, turvar, tahun
    return None, None, None, None


def process_single_json(json_path):
    output_folder = "csv_generated"
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.basename(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "datacontent" not in data:
        raise ValueError(f"Tidak ada 'datacontent' dalam file: {filename}")

    subject_label = data.get('subject', [{}])[0].get('label', "-")
    var_label = data['var'][0]['label']
    turvar_labels = {item['val']: item['label'] for item in data.get('turvar', [])}
    vervar_labels = {item['val']: item['label'] for item in data.get('vervar', [])}
    tahun_labels = {item['val']: item['label'] for item in data.get('tahun', [])}
    datacontent = data['datacontent']

    vervar_keys = sorted(vervar_labels.keys(), key=lambda x: len(str(x)), reverse=True)
    var_keys = sorted(set(item['val'] for item in data.get('var', [])), key=lambda x: len(str(x)), reverse=True)
    turvar_keys = sorted(turvar_labels.keys(), key=lambda x: len(str(x)), reverse=True)
    tahun_keys = sorted(tahun_labels.keys(), key=lambda x: len(str(x)), reverse=True)

    csv_data = []
    for key, value in datacontent.items():
        if value is not None:
            vervar_val, var_val, turvar_val, tahun_val = extract_ids(key, vervar_keys, var_keys, turvar_keys, tahun_keys)

            if None in (vervar_val, var_val, turvar_val, tahun_val):
                continue

            vervar_label = vervar_labels.get(vervar_val, "-").replace("Tidak ada", "-")
            turvar_label = turvar_labels.get(turvar_val, "-").replace("Tidak ada", "-")
            tahun_label = tahun_labels.get(tahun_val, "-").replace("Tidak ada", "-")

            csv_data.append([subject_label, vervar_label, var_label.replace("Tidak ada", "-"), turvar_label, tahun_label, value])

    # Nama file CSV sama seperti JSON tapi .csv
    csv_filename = os.path.splitext(filename)[0] + ".csv"
    csv_file_path = os.path.join(output_folder, csv_filename)

    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label_subject", "label_vervar", "label_var", "label_turvar", "label_tahun", "nilai"])
        writer.writerows(csv_data)

    return csv_file_path  # Untuk dikirim balik via FastAPI
