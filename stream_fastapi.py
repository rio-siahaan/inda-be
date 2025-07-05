from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import requests
import logging
import httpx

app = FastAPI()

# Logging config
logging.basicConfig(level=logging.DEBUG)

# Pydantic model untuk validasi request
class InputData(BaseModel):
    response_text: str
    id: str
    selectedModel: str

def send_to_main(response_text: str, session_id: str, selectedModel: str) -> Optional[str]:
    main_url = 'http://localhost:8000/process_text'
    payload = {'response_text': response_text, 'id': session_id, 'selectedModel': selectedModel}

    try:
        logging.debug(f"Sending to main.py. Payload: {payload}")
        response = requests.post(main_url, json=payload)
        logging.debug(f"Main.py response status: {response.status_code}, Response: {response.text}")

        if response.status_code == 200:
            return response.json().get('processed_text', '')
        else:
            logging.error(f"Failed to get processed response from main.py. Status code: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Error sending data to main.py: {str(e)}")
        return None

@app.post("/get_response_stream")
async def get_response_stream(data: InputData):
    async def event_generator():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                "http://localhost:8000/process_text_stream",
                json={
                    "response_text": data.response_text,
                    "id": data.id,
                    "selectedModel": data.selectedModel
                },
            ) as response:
                async for line in response.aiter_lines():
                    if line.strip():
                        yield f"data: {line}\n\n"  # Format SSE

    return StreamingResponse(event_generator(), media_type="text/event-stream")