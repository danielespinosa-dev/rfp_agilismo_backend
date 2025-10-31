import io
import pandas as pd
from flask import json
import httpx
import asyncio
from typing import Optional, Dict, Any, List
from models import TipoAsistenteEnum

class OpenAIAssistant:
    def __init__(self, api_key: str, assistant_id: str):
        self.api_key = api_key
        self.assistant_id = assistant_id
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2"
        }

    async def create_thread(self) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/threads",
                headers=self.headers
            )
            response.raise_for_status()
            thread_id = response.json()["id"]
            print(f"[OpenAI] Thread creado: {thread_id}")
            return thread_id

    async def create_message(self, thread_id: str, content: str) -> Optional[str]:
        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/threads/{thread_id}/messages",
                        headers=self.headers,
                        json={"role": "user", "content": content}
                    )
                    response.raise_for_status()
                    message_id = response.json()["id"]
                    print(f"[OpenAI] Mensaje creado en thread {thread_id}: {message_id}")
                    return message_id
            except Exception as e:
                attempt += 1
                print(f"[OpenAI][ERROR] create_message intento {attempt} en thread {thread_id}: {str(e)}")
                if attempt < max_attempts:
                    await asyncio.sleep(2)  # Espera antes de reintentar
        print(f"[OpenAI][ERROR] create_message falló tras {max_attempts} intentos en thread {thread_id}")
        return None

    async def create_message_with_files(self, thread_id: str, content: str, file_ids: Optional[List[str]]) -> Optional[str]:
        """
        Crea un mensaje en el hilo del asistente incluyendo archivos adjuntos.
        Si hay más de 5 archivos, los envía en lotes de 5 por mensaje.
        Captura y loguea cualquier excepción por lote, reintentando hasta 3 veces.
        Devuelve None si todos los lotes fallan.
        """
        try:
            if not file_ids:
                # Si no hay archivos, solo envía el mensaje normal
                return await self.create_message(thread_id, content)
            message_ids = []
            # Procesar archivos en lotes de 5
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
                success = False
                attempts = 0
                while not success and attempts < 3:
                    try:
                        async with httpx.AsyncClient() as client:
                            response = await client.post(
                                f"{self.base_url}/threads/{thread_id}/messages",
                                headers=self.headers,
                                json=message_payload
                            )
                            print("message_payload enviado a OpenAI:", message_payload)
                            print("response recibido de OpenAI:", response.json())
                            response.raise_for_status()
                            message_id = response.json()["id"]
                            print(f"[OpenAI] Mensaje con archivos creado en thread {thread_id}: {message_id} (Archivos {i+1}-{i+len(batch)})")
                            message_ids.append(message_id)
                            success = True
                    except Exception as e:
                        attempts += 1
                        print(f"[OpenAI][ERROR] create_message_with_files intento {attempts} (Archivos {i+1}-{i+len(batch)}): {str(e)}")
                        if attempts < 3:
                            await asyncio.sleep(2)  # Espera antes de reintentar
                await asyncio.sleep(5)  # Delay de 5 segundos entre lotes
            # Retorna el último message_id (o lista si prefieres)
            return message_ids[-1] if message_ids else None
        except Exception as e:
            print(f"[OpenAI][ERROR] create_message_with_files Unexpected error fuera del ciclo: {str(e)}")
            return None

    async def create_run(self, thread_id: str) -> Optional[str]:
        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/threads/{thread_id}/runs",
                        headers=self.headers,
                        json={"assistant_id": self.assistant_id}
                    )
                    response.raise_for_status()
                    run_id = response.json()["id"]
                    print(f"[OpenAI] Run creado en thread {thread_id}: {run_id}")
                    return run_id
            except Exception as e:
                attempt += 1
                print(f"[OpenAI][ERROR] create_run intento {attempt} en thread {thread_id}: {str(e)}")
                if attempt < max_attempts:
                    await asyncio.sleep(2)  # Espera antes de reintentar
        print(f"[OpenAI][ERROR] create_run falló tras {max_attempts} intentos en thread {thread_id}")
        return None

    async def get_run_status(self, thread_id: str, run_id: str, max_retries: int = 10, retry_interval: float = 2.0) -> Dict[str, Any]:
        """
        Consulta el estado de un run en OpenAI. Si la petición falla, reintenta hasta max_retries veces.
        """
        attempt = 0
        while attempt < max_retries:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.base_url}/threads/{thread_id}/runs/{run_id}",
                        headers=self.headers
                    )
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as e:
                print(f"[OpenAI][ERROR] get_run_status intento {attempt+1}: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                print(f"[OpenAI][ERROR] get_run_status intento {attempt+1}: {str(e)}")
            attempt += 1
            await asyncio.sleep(retry_interval)
        print(f"[OpenAI][ERROR] get_run_status falló tras {max_retries} intentos para run {run_id}")
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
        while elapsed < timeout:
            run_status = await self.get_run_status(thread_id, run_id)
            status = run_status.get("status")
            print(f"[OpenAI] status ({tipo_asistente.value}) {status}")
            if status == "requires_action" and run_status.get("required_action"):
                try:
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
                            f"{self.base_url}/threads/{thread_id}/runs/{run_id}/submit_tool_outputs",
                            headers=self.headers,
                            json={"tool_outputs": tool_outputs}
                        )
                        response.raise_for_status()
                    print(f"[OpenAI] Acción requerida completada en run {run_id} ({tipo_asistente.value})")
                except Exception as e:
                    print(f"[OpenAI][ERROR] wait_for_required_action al procesar required_action en run {run_id}: {str(e)}")
                    # Continúa esperando el siguiente estado
            if status == "completed":
                if not required_action_detected:
                    retry_message = (
                        "Por favor, ejecuta la función configurada en el assistant y entrega el resultado de la revisión."
                    )
                    await self.create_message(thread_id, retry_message)
                    new_run_id = await self.create_run(thread_id)
                    print(f"[OpenAI] Run adicional creado en thread {thread_id}: {new_run_id} (no hubo required_action inicial)")
                    return await self.wait_for_required_action(thread_id, new_run_id, tipo_asistente, interval, timeout)
                assistant_response = await self.get_completed_run_response(thread_id, run_id)
                print(f"[OpenAI] Run completado en thread {thread_id}: {run_id}")
                return {
                    "required_action": required_action_response,
                    "assistant_response": assistant_response,
                    "last_run_status": run_status
                }
            if status in ["cancelling","failed", "cancelled","incomplete","expired"]:
                print(f"[OpenAI] Run {run_id} estado ({status})")
                return {
                    "required_action": required_action_response,
                    "assistant_response": status,
                    "last_run_status": run_status
                }
            await asyncio.sleep(interval)
            elapsed += interval
        
        print(f"[OpenAI] wait_for_required_action Timeout esperando required_action o completion en run {run_id}")
        raise TimeoutError("wait_for_required_action Run did not reach required_action or completed state in time.")

    async def get_completed_run_response(self, thread_id: str, run_id: str, max_retries: int = 5, retry_interval: float = 2.0) -> Optional[str]:
        """
        Consulta la respuesta del asistente cuando el run está completado.
        Retorna el contenido de texto de todos los mensajes del asistente, separados por salto de línea.
        Si la petición falla, reintenta hasta max_retries veces.
        """
        attempt = 0
        while attempt < max_retries:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.base_url}/threads/{thread_id}/messages",
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
                print(f"[OpenAI][ERROR] get_completed_run_response intento {attempt+1}: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                print(f"[OpenAI][ERROR] get_completed_run_response intento {attempt+1}: {str(e)}")
            attempt += 1
            await asyncio.sleep(retry_interval)
        print(f"[OpenAI][ERROR] get_completed_run_response falló tras {max_retries} intentos para thread {thread_id}, run {run_id}")
        return None

    async def run_assistant_flow(
        self,
        user_message: str,
        tipo_asistente: TipoAsistenteEnum,
        file_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Ejecuta el flujo completo: crea hilo, mensaje (con archivos si hay), run y espera el llamado a función.
        Retorna el required_action si se dispara, None si termina sin requerir acción.
        """
        try:
            thread_id = await self.create_thread()
            #if file_ids:
            #    await self.create_message_with_files(thread_id, "Estos son los archivos que debes revisar", file_ids)
            await self.create_message(thread_id, user_message)
            run_id = await self.create_run(thread_id)
            result = await self.wait_for_required_action(thread_id, run_id, tipo_asistente=tipo_asistente)
            return result
        except httpx.HTTPStatusError as e:
            print(f"[OpenAI][ERROR] run_assistant_flow {tipo_asistente.value} {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"[OpenAI][ERROR] run_assistant_flow Unexpected error: {str(e)}")
            return None

    async def upload_file_from_formdata(self, file, filename: str, purpose: str = "assistants") -> Optional[Dict[str, Any]]:
        """
        Sube un archivo recibido como FormData (por ejemplo, desde FastAPI) al API de OpenAI.
        """
        try:
            async with httpx.AsyncClient() as client:
                files = {"file": (filename, await file.read(), "application/octet-stream")}
                data = {"purpose": purpose}
                response = await client.post(
                    f"{self.base_url}/files",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "OpenAI-Beta": "assistants=v2"
                    },
                    data=data,
                    files=files
                )
                response.raise_for_status()
                file_id = response.json().get("id")
                print(f"[OpenAI] Archivo subido: {file_id} ({filename})")
                return response.json()
        except httpx.HTTPStatusError as e:
            print(f"[OpenAI][ERROR] Upload file:{filename} {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"[OpenAI][ERROR] {filename} Unexpected error al subir archivo: {str(e)}")
            return None     

    async def upload_file_from_formdata_v2(self, file, filename: str, purpose: str = "assistants") -> Optional[Dict[str, Any]]:
        """
        Sube un archivo recibido como FormData (por ejemplo, desde FastAPI) al API de OpenAI.
        Si el archivo es Excel, lo convierte a CSV antes de subirlo.
        Excluye archivos de imagen.
        """
        try:
            image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
            if filename.lower().endswith(image_extensions):
                print(f"[OpenAI][INFO] Archivo excluido por ser imagen: {filename}")
                return None
            file_bytes = await file.read()
            # Detecta si es un archivo Excel por la extensión
            if filename.lower().endswith(('.xlsx', '.xls')):
                # Convierte el Excel a CSV
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
                    f"{self.base_url}/files",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "OpenAI-Beta": "assistants=v2"
                    },
                    data=data,
                    files=files
                )
                response.raise_for_status()
                file_id = response.json().get("id")
                print(f"[OpenAI] Archivo subido: {file_id} ({filename})")
                return response.json()
        except httpx.HTTPStatusError as e:
            print(f"[OpenAI][ERROR] Upload file:{filename} {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"[OpenAI][ERROR] {filename} Unexpected error al subir archivo: {str(e)}")
            return None

    async def depureFiles(self):
        """
        Consulta todos los archivos en OpenAI y los elimina uno por uno.
        """
        try:
            async with httpx.AsyncClient() as client:
                # Obtener la lista de archivos
                response = await client.get(
                    f"{self.base_url}/files",
                    headers=self.headers
                )
                response.raise_for_status()
                files = response.json().get("data", [])
                print(f"[OpenAI] Archivos encontrados: {len(files)}")
                # Eliminar cada archivo
                for file in files:
                    file_id = file.get("id")
                    if file_id:
                        del_response = await client.delete(
                            f"{self.base_url}/files/{file_id}",
                            headers=self.headers
                        )
                        if del_response.status_code == 200:
                            print(f"[OpenAI] Archivo eliminado: {file_id}")
                        else:
                            print(f"[OpenAI][ERROR] No se pudo eliminar archivo: {file_id} - {del_response.status_code}")
        except Exception as e:
            print(f"[OpenAI][ERROR] depureFiles Unexpected error: {str(e)}")

    async def add_files_to_vector_store(self, vector_store_id: str, file_ids: List[str]) -> Optional[List[Dict[str, Any]]]:
        """
        Agrega archivos a un vector store existente en OpenAI, uno por uno.
        Retorna la lista de respuestas del API.
        """
        results = []
        try:
            async with httpx.AsyncClient() as client:
                for file_id in file_ids:
                    payload = {"file_id": file_id}
                    response = await client.post(
                        f"{self.base_url}/vector_stores/{vector_store_id}/files",
                        headers=self.headers,
                        json=payload
                    )
                    if response.status_code == 200:
                        print(f"[OpenAI] Archivo {file_id} agregado al vector store {vector_store_id}")
                        results.append(response.json())
                    else:
                        print(f"[OpenAI][ERROR] add_files_to_vector_store {response.status_code} - {response.text}")
            return results
        except Exception as e:
            print(f"[OpenAI][ERROR] add_files_to_vector_store Unexpected error: {str(e)}")
            return None

    async def delete_all_files_from_vector_store(self, vector_store_id: str) -> bool:
        """
        Elimina todos los archivos de un vector store en OpenAI.
        """
        try:
            # Obtener la lista de archivos en el vector store
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/vector_stores/{vector_store_id}/files",
                    headers=self.headers
                )
                response.raise_for_status()
                files = response.json().get("data", [])
                file_ids = [f.get("id") for f in files if f.get("id")]
                # Eliminar cada archivo
                for file_id in file_ids:
                    del_response = await client.delete(
                        f"{self.base_url}/vector_stores/{vector_store_id}/files/{file_id}",
                        headers=self.headers
                    )
                    if del_response.status_code == 200:
                        print(f"[OpenAI] Archivo {file_id} eliminado del vector store {vector_store_id}")
                    else:
                        print(f"[OpenAI][ERROR] No se pudo eliminar archivo {file_id} del vector store {vector_store_id} - {del_response.status_code}")
            return True
        except Exception as e:
            print(f"[OpenAI][ERROR] delete_all_files_from_vector_store Unexpected error: {str(e)}")
            return False