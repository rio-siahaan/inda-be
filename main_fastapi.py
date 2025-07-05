from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import requests
import logging
import os
import shutil
from fastapi.responses import StreamingResponse

app = FastAPI()

# Logging config
logging.basicConfig(level=logging.DEBUG)

# Pydantic model untuk validasi request
class InputData(BaseModel):
    response_text: str
    id: str
    selectedModel: str

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

@app.post("/get_response")
async def get_response(data: InputData):
    response_text = data.response_text
    session_id = data.id
    selectedModel = data.selectedModel
    persona = data.persona
    name = data.name

    if response_text and session_id:
        logging.debug(f"Received response_text: {response_text} with id: {session_id} and model : {selectedModel}")
        processed_text, usage_metadata = send_to_main(response_text, session_id, selectedModel, persona, name)
        return {"status": "success", "processed_text": processed_text, "usage_metadata": usage_metadata}
    else:
        logging.error("No response_text or session_id provided")
        return {"status": "error", "message": "No response_text or session_id provided"}

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
    