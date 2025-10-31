import io
import pandas as pd
from flask import json
import httpx
import asyncio
from typing import Optional, Dict, Any, List
from models import TipoAsistenteEnum

class AzureOpenAIAssistant:
    def __init__(self, api_key: str, endpoint: str, assistant_id: str, api_version: str = "2024-05-01-preview"):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.assistant_id = assistant_id
        self.api_version = api_version
        self.base_url = f"{self.endpoint}/openai/assistants"
        self.headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }

    async def create_thread(self) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoint}/openai/threads?api-version={self.api_version}",  # <-- URL corregida
                headers=self.headers
            )
            response.raise_for_status()
            thread_id = response.json()["id"]
            print(f"[AzureOpenAI] Thread creado: {thread_id}")
            return thread_id

    async def create_message(self, thread_id: str, content: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoint}/openai/threads/{thread_id}/messages?api-version={self.api_version}",
                headers=self.headers,
                json={"role": "user", "content": content}
            )
            response.raise_for_status()
            message_id = response.json()["id"]
            print(f"[AzureOpenAI] Mensaje creado en thread {thread_id}: {message_id}")
            return message_id

    async def create_message_with_files(self, thread_id: str, content: str, file_ids: Optional[List[str]]) -> Optional[str]:
        try:
            if not file_ids:
                return await self.create_message(thread_id, content)
            message_ids = []
            for i in range(0, len(file_ids), 5):
                batch = file_ids[i:i+5]
                attachments = [
                    {
                        "file_id": fid,
                        "tools": [
                            {"type": "file_search"},
                            {"type": "code_interpreter"}
                        ]
                    }
                    for fid in batch
                ]
                message_payload = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{content} (Archivos {i+1}-{i+len(batch)})"
                        }
                    ],
                    "attachments": attachments
                }
                print("Payload enviado a Azure OpenAI:", message_payload)
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.endpoint}/openai/threads/{thread_id}/messages?api-version={self.api_version}",
                        headers=self.headers,
                        json=message_payload
                    )
                    response.raise_for_status()
                    message_id = response.json()["id"]
                    print(f"[AzureOpenAI] Mensaje con archivos creado en thread {thread_id}: {message_id} (Archivos {i+1}-{i+len(batch)})")
                    message_ids.append(message_id)
            return message_ids[-1] if message_ids else None
        except Exception as e:
            print(f"[AzureOpenAI][ERROR] create_message_with_files Unexpected error: {str(e)}")
            return None

    async def create_run(self, thread_id: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoint}/openai/threads/{thread_id}/runs?api-version={self.api_version}",
                headers=self.headers,
                json={
                    "assistant_id": self.assistant_id,
                    "temperature": 0.5,
                    "top_p": 0.8,
                    "max_prompt_tokens": 100000,
                    "max_completion_tokens": 100000,
                    "parallel_tool_calls": False,
                    "truncation_strategy": {
                        "type": "auto"
                    }
                }
            )
            response.raise_for_status()
            run_id = response.json()["id"]
            print(f"[AzureOpenAI] Run creado en thread {thread_id}: {run_id}")
            return run_id

    async def get_run_status(self, thread_id: str, run_id: str, max_retries: int = 10, retry_interval: float = 2.0) -> Dict[str, Any]:
        attempt = 0
        while attempt < max_retries:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.endpoint}/openai/threads/{thread_id}/runs/{run_id}?api-version={self.api_version}",
                        headers=self.headers
                    )
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as e:
                print(f"[AzureOpenAI][ERROR] get_run_status intento {attempt+1}: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                print(f"[AzureOpenAI][ERROR] get_run_status intento {attempt+1}: {str(e)}")
            attempt += 1
            await asyncio.sleep(retry_interval)
        print(f"[AzureOpenAI][ERROR] get_run_status falló tras {max_retries} intentos para run {run_id}")
        return {}

    async def wait_for_required_action(
        self,
        thread_id: str,
        run_id: str,
        tipo_asistente: TipoAsistenteEnum,
        interval: float = 10.0,
        timeout: float = 10000.0
    ) -> Optional[Dict[str, Any]]:
        elapsed = 0
        required_action_detected = False
        required_action_response = None
        failed_retries = 0  # contador de reintentos en estado failed
        while elapsed < timeout:
            run_status = await self.get_run_status(thread_id, run_id)
            status = run_status.get("status")
            print(f"[AzureOpenAI] status {status}")
            if status == "requires_action" and run_status.get("required_action"):
                required_action_detected = True
                required_action = run_status["required_action"]
                required_action_response = required_action
                tool_calls = required_action.get("submit_tool_outputs", {}).get("tool_calls", [])
                tool_outputs = [
                    {
                        "tool_call_id": call["id"],
                        "output": "Ok, función ejecutada correctamente."
                    }
                    for call in tool_calls
                ]
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.endpoint}/openai/threads/{thread_id}/runs/{run_id}/submit_tool_outputs?api-version={self.api_version}",
                        headers=self.headers,
                        json={"tool_outputs": tool_outputs}
                    )
                    response.raise_for_status()
                print(f"[AzureOpenAI] Acción requerida completada en run {run_id} ({tipo_asistente.value})")
            if status == "completed":
                #if not required_action_detected:
                #    retry_message = (
                #        "Por favor, ejecuta la función configurada en el assistant y entrega el resultado de la revisión."
                #    )
                #    await self.create_message(thread_id, retry_message)
                #    new_run_id = await self.create_run(thread_id)
                #    print(f"[AzureOpenAI] Run adicional creado en thread {thread_id}: {new_run_id} (no hubo required_action inicial)")
                #    return await self.wait_for_required_action(thread_id, new_run_id, tipo_asistente, interval, timeout)
                assistant_response = await self.get_completed_run_response(thread_id, run_id)
                print(f"[AzureOpenAI] Run completado en thread {thread_id}: {run_id}")
                return {
                    "required_action": required_action_response,
                    "assistant_response": assistant_response,
                    "last_run_status": run_status
                }
            if status == "failed":
                failed_retries += 1
                print(f"[AzureOpenAI] Run {run_id} estado (failed), reintento {failed_retries}/4")
                if failed_retries < 4:
                    await asyncio.sleep(interval)
                    elapsed += interval
                    continue
                else:
                    return {
                        "required_action": required_action_response,
                        "assistant_response": status,
                        "last_run_status": run_status
                    }
            if status in ["cancelling", "cancelled", "incomplete", "expired"]:
                print(f"[AzureOpenAI] Run {run_id} estado ({status})")
                return {
                    "required_action": required_action_response,
                    "assistant_response": status,
                    "last_run_status": run_status
                }
            await asyncio.sleep(interval)
            elapsed += interval
        print(f"[AzureOpenAI] wait_for_required_action Timeout esperando required_action o completion en run {run_id}")
        raise TimeoutError("wait_for_required_action Run did not reach required_action or completed state in time.")

    async def get_completed_run_response(self, thread_id: str, run_id: str, max_retries: int = 5, retry_interval: float = 2.0) -> Optional[str]:
        attempt = 0
        while attempt < max_retries:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.endpoint}/openai/threads/{thread_id}/messages?api-version={self.api_version}",
                        headers=self.headers
                    )
                    response.raise_for_status()
                    messages = response.json().get("data", [])
                    assistant_texts = []
                    for msg in messages:
                        if msg.get("role") == "assistant":
                            content = msg.get("content")
                            if isinstance(content, list):
                                for c in content:
                                    if c.get("type") == "text":
                                        text_obj = c.get("text")
                                        if isinstance(text_obj, dict):
                                            assistant_texts.append(text_obj.get("value", ""))
                                        elif isinstance(text_obj, str):
                                            assistant_texts.append(text_obj)
                            elif isinstance(content, str):
                                assistant_texts.append(content)
                    return "\n".join(assistant_texts) if assistant_texts else None
            except httpx.HTTPStatusError as e:
                print(f"[AzureOpenAI][ERROR] get_completed_run_response intento {attempt+1}: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                print(f"[AzureOpenAI][ERROR] get_completed_run_response intento {attempt+1}: {str(e)}")
            attempt += 1
            await asyncio.sleep(retry_interval)
        print(f"[AzureOpenAI][ERROR] get_completed_run_response falló tras {max_retries} intentos para thread {thread_id}, run {run_id}")
        return None

    async def run_assistant_flow(
        self,
        user_message: str,
        tipo_asistente: TipoAsistenteEnum,
        file_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            thread_id = await self.create_thread()
            if file_ids:
                await self.create_message_with_files(thread_id, "Estos son los archivos que debes revisar", file_ids)
            await self.create_message(thread_id, user_message)
            run_id = await self.create_run(thread_id)
            result = await self.wait_for_required_action(thread_id, run_id, tipo_asistente=tipo_asistente)
            return result
        except httpx.HTTPStatusError as e:
            print(f"[AzureOpenAI][ERROR] run_assistant_flow {tipo_asistente.value} {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"[AzureOpenAI][ERROR] run_assistant_flow Unexpected error: {str(e)}")
            return None

    async def upload_file_from_formdata_v2(self, file, filename: str, purpose: str = "assistants") -> Optional[Dict[str, Any]]:
        try:
            file_bytes = await file.read()
            if filename.lower().endswith((".xlsx", ".xls")):
                excel_io = io.BytesIO(file_bytes)
                df = pd.read_excel(excel_io)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                file_bytes = csv_buffer.getvalue().encode('utf-8')
                filename = filename.rsplit('.', 1)[0] + ".txt"
                mime_type = "text"
            else:
                mime_type = "application/octet-stream"
            async with httpx.AsyncClient() as client:
                files = {"file": (filename, file_bytes, mime_type)}
                data = {"purpose": purpose}
                response = await client.post(
                    f"{self.endpoint}/openai/files?api-version={self.api_version}",
                    headers={
                        "api-key": self.api_key
                    },
                    data=data,
                    files=files
                )
                response.raise_for_status()
                file_id = response.json().get("id")
                print(f"[AzureOpenAI] Archivo subido: {file_id} ({filename})")
                return response.json()
        except httpx.HTTPStatusError as e:
            print(f"[AzureOpenAI][ERROR] Upload file:{filename} {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"[AzureOpenAI][ERROR] {filename} Unexpected error al subir archivo: {str(e)}")
            return None

    async def depureFilesV2(self, current_file_ids: list):
        """
        Elimina solo los archivos cuyos IDs estén en current_file_ids.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.endpoint}/openai/files?api-version={self.api_version}",
                    headers={"api-key": self.api_key}
                )
                response.raise_for_status()
                files = response.json().get("data", [])
                print(f"[AzureOpenAI] Archivos encontrados: {len(files)})")
                for file in files:
                    file_id = file.get("id")
                    if file_id and file_id in current_file_ids:
                        del_response = await client.delete(
                            f"{self.endpoint}/openai/files/{file_id}?api-version={self.api_version}",
                            headers={"api-key": self.api_key}
                        )
                        if del_response.status_code == 200:
                            print(f"[AzureOpenAI] Archivo eliminado: {file_id}")
                        else:
                            print(f"[AzureOpenAI][ERROR] No se pudo eliminar archivo: {file_id} - {del_response.status_code}")
        except Exception as e:
            print(f"[AzureOpenAI][ERROR] depureFiles Unexpected error: {str(e)}")