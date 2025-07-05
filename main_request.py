import requests
import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from datetime import datetime

app = FastAPI()

logging.basicConfig(level=logging.DEBUG)

class InputRequest(BaseModel):
    batasBawah: int
    batasAtas: int

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


@app.post("/process_request")
async def process_request(data: InputRequest):
    batasBawah = data.batasBawah
    batasAtas = data.batasAtas
    if not batasBawah:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No response text provided"})
    if not batasAtas:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No response text provided"})

    hasil = request_to_api(batasBawah, batasAtas)

    return JSONResponse(status_code=200, content={
        "status": "success",
        "message": f"Data berhasil diproses dari varID {batasBawah} sampai {batasAtas - 1}",
        "detail": hasil
    })
    
def request_to_api(batasBawah: int, batasAtas: int):
    hasil = []
    waktu_request = datetime.now().isoformat()
    for var_id in range(batasBawah, batasAtas):
        url = base_url.format(var_id)
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()

            if data.get('data-availability') == 'available' and 'var' in data:
                label = data['var'][0]['label']
                sanitized_label = sanitize_filename(label)
                filename = f"{sanitized_label}.json"
                file_path = os.path.join(output_folder, filename)

                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                    hasil.append({"var_id": var_id, "filename": filename, "status": "tersimpan", 'waktu_request': waktu_request})
                except FileNotFoundError as e:
                    hasil.append({"var_id": var_id, "status": "gagal simpan", "error": str(e)})
            else:
                hasil.append({"var_id": var_id, "status": "tidak tersedia"})
        else:
            hasil.append({"var_id": var_id, "status": "gagal fetch", "code": response.status_code})
    return hasil
