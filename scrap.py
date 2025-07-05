import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Folder tempat menyimpan file JSON
output_folder = 'JSON'
os.makedirs(output_folder, exist_ok=True)

# Domain dan Key sebagai variabel
domain = os.getenv("DOMAIN_BPS")
key = os.getenv("API_KEY_BPS")

# Base URL tanpa varID
base_url = f'https://webapi.bps.go.id/v1/api/list/model/data/domain/{domain}/var/{{}}/key/{key}/'

# Fungsi untuk membersihkan nama file dari karakter yang tidak valid
def sanitize_filename(filename):
    # Gantikan karakter tidak valid dengan underscore atau karakter lain yang valid
    return filename.replace('/', '-').replace('\\', '-').replace(':', '-')

# Loop untuk varID dari 1 hingga 50 (dapat diubah sesuai kebutuhan)
for var_id in range(30, 50): # isi sesuai banyaknya varID (ID tabel) dari Web API BPS, contohnya sampai varID 700 
    # Buat URL dengan varID saat ini
    url = base_url.format(var_id)
    
    # Ambil respons dari API
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        
        # Cek jika data tersedia
        if data.get('data-availability') == 'available' and 'var' in data:
            # Ambil nama file dari label varID
            label = data['var'][0]['label']
            
            # Membersihkan nama file
            sanitized_label = sanitize_filename(label)
            filename = f"{sanitized_label}.json"
            
            # Path lengkap untuk menyimpan file
            file_path = os.path.join(output_folder, filename)
            
            try:
                # Simpan data sebagai file JSON
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                
                print(f"Data untuk varID {var_id} disimpan di {file_path}")
            except FileNotFoundError as e:
                print(f"Gagal menyimpan file untuk varID {var_id}: {e}")
        else:
            print(f"Data untuk varID {var_id} tidak tersedia, melewati...")
    else:
        print(f"Gagal mengambil data untuk varID {var_id}, status code: {response.status_code}")
