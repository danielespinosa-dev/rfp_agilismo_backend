from openai import AzureOpenAI
import pandas as pd
import io
from typing import Optional, Dict, Any, List
from models import TipoAsistenteEnum

class AzureOpenAISDKAssistant:
    def __init__(self, api_key: str, endpoint: str, assistant_id: str, api_version: str = "2024-05-01-preview"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.assistant_id = assistant_id
        self.api_version = api_version
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )

    async def create_thread(self) -> str:
        thread = self.client.beta.threads.create()
        thread_id = thread.id
        print(f"[AzureOpenAI SDK] Thread creado: {thread_id}")
        return thread_id

    async def create_message(self, thread_id: str, content: str) -> str:
        message = self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content
        )
        print(f"[AzureOpenAI SDK] Mensaje creado en thread {thread_id}: {message.id}")
        return message.id

    async def create_message_with_files(self, thread_id: str, content: str, file_ids: Optional[List[str]]) -> Optional[str]:
        if not file_ids:
            return self.create_message(thread_id, content)
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
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=[{"type": "text", "text": f"{content} (Archivos {i+1}-{i+len(batch)})"}],
                attachments=attachments
            )
            print(f"[AzureOpenAI SDK] Mensaje con archivos creado en thread {thread_id}: {message.id}")
            message_ids.append(message.id)
        return message_ids[-1] if message_ids else None

    async def create_run(self, thread_id: str) -> str:
        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self.assistant_id,
            temperature=0.2,
            top_p=1,
            max_prompt_tokens=12000,
            max_completion_tokens=4000,
            parallel_tool_calls=False,
            truncation_strategy={"type": "auto"}
        )
        print(f"[AzureOpenAI SDK] Run creado en thread {thread_id}: {run.id}")
        return run.id

    async def get_run_status(self, thread_id: str, run_id: str, max_retries: int = 10, retry_interval: float = 2.0) -> Dict[str, Any]:
        import asyncio
        attempt = 0
        while attempt < max_retries:
            try:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run_id
                )
                return run_status.dict()
            except Exception as e:
                print(f"[AzureOpenAI SDK][ERROR] get_run_status intento {attempt+1}: {str(e)}")
            attempt += 1
            await asyncio.sleep(retry_interval)
        print(f"[AzureOpenAI SDK][ERROR] get_run_status falló tras {max_retries} intentos para run {run_id}")
        return {}

    async def wait_for_required_action(
        self,
        thread_id: str,
        run_id: str,
        tipo_asistente: TipoAsistenteEnum,
        interval: float = 10.0,
        timeout: float = 10000.0
    ) -> Optional[Dict[str, Any]]:
        import asyncio
        elapsed = 0
        required_action_detected = False
        required_action_response = None
        failed_retries = 0
        while elapsed < timeout:
            run_status = self.get_run_status(thread_id, run_id)
            status = run_status.get("status")
            print(f"[AzureOpenAI SDK] status ({tipo_asistente.value}) {status}")
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
                self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run_id,
                    tool_outputs=tool_outputs
                )
                print(f"[AzureOpenAI SDK] Acción requerida completada en run {run_id} ({tipo_asistente.value})")
            if status == "completed":
                if not required_action_detected:
                    retry_message = (
                        "Por favor, ejecuta la función configurada en el assistant y entrega el resultado de la revisión."
                    )
                    self.create_message(thread_id, retry_message)
                    new_run_id = self.create_run(thread_id)
                    print(f"[AzureOpenAI SDK] Run adicional creado en thread {thread_id}: {new_run_id} (no hubo required_action inicial)")
                    return await self.wait_for_required_action(thread_id, new_run_id, tipo_asistente, interval, timeout)
                assistant_response = self.get_completed_run_response(thread_id, run_id)
                print(f"[AzureOpenAI SDK] Run completado en thread {thread_id}: {run_id}")
                return {
                    "required_action": required_action_response,
                    "assistant_response": assistant_response,
                    "last_run_status": run_status
                }
            if status == "failed":
                failed_retries += 1
                print(f"[AzureOpenAI SDK] Run {run_id} estado (failed), reintento {failed_retries}/4")
                if failed_retries < 4:
                    await asyncio.sleep(interval*10)
                    elapsed += interval
                    continue
                else:
                    return {
                        "required_action": required_action_response,
                        "assistant_response": status,
                        "last_run_status": run_status
                    }
            if status in ["cancelling", "cancelled", "incomplete", "expired"]:
                print(f"[AzureOpenAI SDK] Run {run_id} estado ({status})")
                return {
                    "required_action": required_action_response,
                    "assistant_response": status,
                    "last_run_status": run_status
                }
            await asyncio.sleep(interval)
            elapsed += interval
        print(f"[AzureOpenAI SDK] wait_for_required_action Timeout esperando required_action o completion en run {run_id}")
        raise TimeoutError("wait_for_required_action Run did not reach required_action or completed state in time.")

    async def get_completed_run_response(self, thread_id: str, run_id: str, max_retries: int = 5, retry_interval: float = 2.0) -> Optional[str]:
        import asyncio
        attempt = 0
        while attempt < max_retries:
            try:
                messages = self.client.beta.threads.messages.list(thread_id=thread_id)
                assistant_texts = []
                for msg in messages.data:
                    if msg.role == "assistant":
                        for c in msg.content:
                            if c.type == "text":
                                text_obj = c.text
                                if isinstance(text_obj, dict):
                                    assistant_texts.append(text_obj.get("value", ""))
                                elif isinstance(text_obj, str):
                                    assistant_texts.append(text_obj)
                return "\n".join(assistant_texts) if assistant_texts else None
            except Exception as e:
                print(f"[AzureOpenAI SDK][ERROR] get_completed_run_response intento {attempt+1}: {str(e)}")
            attempt += 1
            await asyncio.sleep(retry_interval)
        print(f"[AzureOpenAI SDK][ERROR] get_completed_run_response falló tras {max_retries} intentos para thread {thread_id}, run {run_id}")
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
        except Exception as e:
            print(f"[AzureOpenAI SDK][ERROR] run_assistant_flow Unexpected error: {str(e)}")
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
            file_obj = io.BytesIO(file_bytes)
            file_response = self.client.files.create(
                file=file_obj,
                purpose=purpose
            )
            print(f"[AzureOpenAI SDK] Archivo subido: {file_response.id} ({filename})")
            return file_response.dict()
        except Exception as e:
            print(f"[AzureOpenAI SDK][ERROR] {filename} Unexpected error al subir archivo: {str(e)}")
            return None

    async def depureFiles(self):
        try:
            files = self.client.files.list()
            print(f"[AzureOpenAI SDK] Archivos encontrados: {len(files.data)})")
            for file in files.data:
                file_id = file.id
                if file_id:
                    self.client.files.delete(file_id)
                    print(f"[AzureOpenAI SDK] Archivo eliminado: {file_id}")
        except Exception as e:
            print(f"[AzureOpenAI SDK][ERROR] depureFiles Unexpected error: {str(e)}")

    async def depureFilesV2(self, current_file_ids: list):
        try:
            files = self.client.files.list()
            print(f"[AzureOpenAI SDK] Archivos encontrados: {len(files.data)})")
            for file in files.data:
                file_id = file.id
                if file_id and file_id in current_file_ids:
                    self.client.files.delete(file_id)
                    print(f"[AzureOpenAI SDK] Archivo eliminado: {file_id}")
        except Exception as e:
            print(f"[AzureOpenAI SDK][ERROR] depureFilesV2 Unexpected error: {str(e)}")
    
    SUPPORTED_EXTENSIONS = [
    ".c", ".cpp", ".cs", ".css", ".doc", ".docx", ".go", ".html", ".java", ".js", ".json",
    ".md", ".pdf", ".php", ".pptx", ".py", ".rb", ".sh", ".tex", ".ts", ".txt"
    ]

    def ensure_supported_extension(filename: str) -> str:
        ext = filename.lower().rsplit('.', 1)[-1]
        if not any(filename.lower().endswith(e) for e in SUPPORTED_EXTENSIONS):
            # Default to .txt if unsupported
            filename = filename.rsplit('.', 1)[0] + ".txt"
        return filename