from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import requests
import logging
import os
import shutil
from fastapi.responses import StreamingResponse
from load_qdrant import load_csv_files_with_metadata, create_or_update_vector_store
import io
from main_model import generate_response 

app = FastAPI()

# Logging config
logging.basicConfig(level=logging.DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model untuk validasi request
class InputData(BaseModel):
    response_text: str
    id: str
    selectedModel: str

class CSVIngestRequest(BaseModel):
    filename: str
    content: str
    
@app.post("/ingest")
async def ingest_csv(data: CSVIngestRequest):
    try:
        csv_bytes = data.content.encode("utf-8")
        file_like = io.BytesIO(csv_bytes)
        file_like.name = data.filename

        documents = load_csv_files_with_metadata([file_like])
        if not documents:
            return {"status": "error", "message": "Gagal parsing dokumen"}

        create_or_update_vector_store(documents, batch_size=25)

        return {"status": "success", "message": f"{len(documents)} dokumen diproses"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def send_to_main(response_text: str, session_id: str, selectedModel: str, persona: str, name: str) -> Optional[str]:
    main_url = 'http://localhost:8000/process_text'
    payload = {'response_text': response_text, 'id_chat': session_id, 'selectedModel': selectedModel, 'persona': persona, 'name': name}

    try:
        logging.debug(f"Sending to main.py. Payload: {payload}")
        response = requests.post(main_url, json=payload)
        logging.debug(f"Main.py response status: {response.status_code}, Response: {response.text}")

        if response.status_code == 200:
            response_json = response.json()
            processed_text = response_json.get('processed_text', '')
            usage_metadata = response_json.get('usage_metadata', {}) 
            return processed_text, usage_metadata
        else:
            logging.error(f"Failed to get processed response from main.py. Status code: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Error sending data to main.py: {str(e)}")
        return None

@app.get("/ping")
def ping():
    return {"message": "pong"}

# @app.post("/get_response")
# async def get_response(data: InputData):
#     response_text = data.response_text
#     session_id = data.id
#     selectedModel = data.selectedModel
#     persona = data.persona
#     name = data.name

#     if response_text and session_id:
#         logging.debug(f"Received response_text: {response_text} with id: {session_id} and model : {selectedModel}")
#         processed_text, usage_metadata = send_to_main(response_text, session_id, selectedModel, persona, name)
#         return {"status": "success", "processed_text": processed_text, "usage_metadata": usage_metadata}
#     else:
#         logging.error("No response_text or session_id provided")
#         return {"status": "error", "message": "No response_text or session_id provided"}

@app.post("/get_response")
async def get_response(data: InputData):
    if not data.response_text or not data.id:
        return {"status": "error", "message": "Field tidak lengkap"}

    result, usage = generate_response(
        data.response_text,
        data.id,
        data.selectedModel,
        data.persona,
        data.name
    )
    return {"status": "success", "processed_text": result, "usage_metadata": usage}

@app.post("/convert-json-to-csv")
async def convert_json_to_csv(file: UploadFile = File(...)):
    # buat folder sementara
    temp_dir = "temp_upload"
    os.makedirs(temp_dir, exist_ok = True)
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    from to_csv import process_single_json
    csv_path = process_single_json(file_path)
    
    return StreamingResponse(open(csv_path), "rb", media_type = "text/csv", headers={
        "Content-Disposition": f"attachment; filename={os.path.basename(csv_path)}"
    })
    