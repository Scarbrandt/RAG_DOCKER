# app/main_http.py
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import db
import sberrag
from fastapi.responses import FileResponse, StreamingResponse
import io
import zipfile

SUPERADMIN = 123

def is_super_admin(user_id: int) -> bool:
    try:
        return user_id == int(SUPERADMIN)
    except (TypeError, ValueError):
        return False

app = FastAPI(title="RAG Service HTTP API")

# Pydantic models
class StartRequest(BaseModel):
    user_id: int
    full_name: str

class AdminModifyRequest(BaseModel):
    user_id: int
    target_user_id: int

class ChatIDRequest(BaseModel):
    chat_id: int

class QueryRequest(BaseModel):
    user_id: int
    chat_id: int
    message_id: int
    text: Optional[str] = None

class FeedbackRequest(BaseModel):
    user_id: int
    chat_id: int
    message_id: int
    feedback: int

class DownloadRequest(BaseModel):
    orig_msg_id: int
    cht_ref_id: int

class FillIndexRequest(BaseModel):
    user_id: int
    folder_path: str

# Endpoints
@app.post("/start")
async def start(req: StartRequest):
    if is_super_admin(req.user_id):
        role = "Супер администратор"
    elif db.is_admin(req.user_id):
        role = "Администратор"
    else:
        role = "Пользователь"
    message = (
        f"Здравствуйте, {req.full_name}! Вы {role}.\n"
        "Добро пожаловать в RAG-сервис. Пожалуйста, введите ваш запрос."
    )
    return {"message": message, "role": role}

@app.on_event("startup")
async def on_startup():
    # инициализация FAISS-индексов
    db.migrate_postgres(SUPERADMIN)
    await sberrag.initialize_faiss()
    await sberrag.setup_retrievers()

@app.post("/admin/add")
async def add_admin(req: AdminModifyRequest):
    if not is_super_admin(req.user_id):
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    db.add_admin(req.target_user_id)
    return {"detail": f"Пользователь {req.target_user_id} добавлен в администраторы"}

@app.post("/admin/delete")
async def delete_admin(req: AdminModifyRequest):
    if not is_super_admin(req.user_id):
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    db.delete_admin(req.target_user_id)
    return {"detail": f"Пользователь {req.target_user_id} удалён из администраторы"}

@app.post("/admin/list")
async def list_admins():
    admins = db.get_all_admins()
    return {"admins": admins}

@app.post("/article/add")
async def add_article(user_id: int = Form(...), file: UploadFile = File(...)):
    if not db.is_admin(user_id):
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    await sberrag.read_pdf(temp_path)
    return {"detail": "PDF получен и добавлен для обработки"}

@app.post("/cache/save")
async def save_faiss_cache(req: ChatIDRequest):
    if not db.is_admin(req.chat_id):
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    await sberrag.save_faiss()
    return {"detail": "Кэш успешно сохранён"}

@app.post("/chat/id")
async def get_chat_id(req: ChatIDRequest):
    return {"chat_id": req.chat_id}

@app.post("/express")
async def express(req: QueryRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Поле 'text' обязательно")
    answer = await sberrag.express_answer(req.text)
    link = db.get_ref(answer['message_id'], answer['chat_id'])
    return {"answer": answer, "link": link}

@app.post("/sbert")
async def sbert(req: QueryRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Поле 'text' обязательно")
    answer = await sberrag.answer_sbert(req.text)
    db.add_message(message_id=req.message_id, chat_id=req.chat_id, text=req.text, context=answer['context'], answer_text=answer['answer'])
    return {"answer": answer}

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    if req.feedback == 1:
        message_text = db.get_answer(req.message_id, req.chat_id)
        await sberrag.save_answer(req.message_id, req.chat_id, message_text)
        await sberrag.save_faiss_answer()
        return {"detail": "Спасибо за вашу оценку! Ответ сохранён для улучшения."}
    else:
        return {"detail": "Жаль, что вы остались недовольны."}

@app.post("/download")
async def download(req: DownloadRequest):
    refs = db.get_all_ref(req.orig_msg_id, req.cht_ref_id)
    if not refs:
        raise HTTPException(status_code=404, detail="Файлы не найдены")

    # если ровно один файл — отдадим его напрямую
    if len(refs) == 1:
        path = refs[0]
        path = path.strip("/tmp")
        if not os.path.isfile(f"/app/docs/{path}"):
            raise HTTPException(status_code=404, detail=f"Файл не найден: /app/docs/{path}")
        return FileResponse(
            f"/app/docs/{path}",
            media_type="application/octet-stream",
            filename=os.path.basename(path)
        )

    # если файлов несколько — упакуем в ZIP и стримим на лету
    zip_stream = io.BytesIO()
    with zipfile.ZipFile(zip_stream, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for row in refs:
            file_path = row[0]
            file_path = file_path.strip("/tmp")
            if os.path.isfile(f"/app/docs/{file_path}"):
                zf.write(f"/app/docs/{file_path}", arcname=os.path.basename(file_path))
    zip_stream.seek(0)

    return StreamingResponse(
        zip_stream,
        media_type="application/x-zip-compressed",
        headers={
            "Content-Disposition": "attachment; filename=files.zip"
        }
    )

@app.post("/index/fill/dynamic")
async def fill_index_dynamic(req: FillIndexRequest, background_tasks: BackgroundTasks):
    if not is_super_admin(req.user_id) and not db.is_admin(req.user_id):
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    background_tasks.add_task(sberrag.fill_index_folder, req.folder_path)
    return {"detail": f"Запущено индексирование PDF из папки: {req.folder_path}"}
