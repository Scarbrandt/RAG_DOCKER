# Импорт необходимых библиотек 
import torch
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
import re
from langchain.docstore.document import Document
import pickle
import logging
import os
import json
from typing import List
import time
from os import getenv
import urllib
from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
import shutil  # Добавлен импорт для работы с файловой системой
import pdfplumber
import os
import pandas as pd
from transformers import AutoModel
import asyncio
from langchain_core.runnables.config import run_in_executor
import ast
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

torch.backends.cudnn.benchmark = True

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
logging.basicConfig(level=logging.INFO)
logging.info("Запуск загрузки векторного хранилища")

DOCS_FOLDER     = os.getenv("DOCS_FOLDER", "./docs")
# Куда сохранять FAISS-индекс
FAISS_INDEX_DIR  = Path(os.getenv("FAISS_INDEX_DIR", "./faiss_index"))

FAISS_INDEX_DIR_ANSWER = Path(os.getenv("FAISS_INDEX_DIR_ANSWER", "./faiss_index_answer"))
# Кеш моделей (HF, jina и т.п.)
MODEL_CACHE     = os.getenv("MODEL_CACHE", "./model_cache")

os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
# (если надо) Jina-кеш
os.environ["JINA_CACHE_DIR"]  = MODEL_CACHE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Используемое устройство: {device}")


eb = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True, cache_dir = MODEL_CACHE, device_map = 'cuda')


class EmbeddingsCustom(Embeddings):
    def __init__(self, model):
        self.model = model
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        start_time = time.time()
        embeddings = self.model.encode(texts, task="retrieval.passage", truncate_dim = 1024)
        end_time = time.time()
        logging.info(f"Embeddings shape (documents): {embeddings.shape}")
        logging.info(f"Время эмбеддинга документов: {end_time - start_time:.2f} секунд")
        if len(embeddings.shape) != 2:
            raise ValueError(f"Expected 2D array for embeddings, got {embeddings.shape}")
        if embeddings.shape[1] != 1024:
            raise ValueError(f"Embedding dimension mismatch: expected {1024}, got {embeddings.shape[1]}")
        return embeddings.tolist()
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await run_in_executor(None, self.embed_documents, texts)

    def embed_query(self, text: str) -> List[float]:
        start_time = time.time()
        embedding = self.model.encode([text], task="retrieval.query", truncate_dim = 1024)
        end_time = time.time()
        logging.info(f"Embedding shape (query): {embedding.shape}") 
        logging.info(f"Embedding values (query): {embedding[:1, :5]}")  
        logging.info(f"Время эмбеддинга запроса: {end_time - start_time:.2f} секунд")
        if len(embedding.shape) != 2 or embedding.shape[0] != 1:
            raise ValueError(f"Expected embedding shape (1, d), got {embedding.shape}")
        if embedding.shape[1] != 1024:
            raise ValueError(f"Embedding dimension mismatch: expected {1024}, got {embedding.shape[1]}")
        return embedding[0].tolist()
    
    async def aembed_query(self, texts: List[str]) -> List[float]:
        return await run_in_executor(None, self.embed_query, texts)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.aembed_query(texts)

embeddings = EmbeddingsCustom(eb)

async def save_faiss():
    store.save_local(FAISS_INDEX_DIR)

async def save_faiss_answer():
    store_answer.save_local(FAISS_INDEX_DIR_ANSWER)

async def initialize_faiss():
    global store, store_answer, docs_list, bm_retriever

    index_file = FAISS_INDEX_DIR/"index.faiss"
    if index_file.exists():
        try:
            logging.info(f"Loading FAISS index from {FAISS_INDEX_DIR}")
            store = FAISS.load_local(
                str(FAISS_INDEX_DIR),
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logging.error(f"Error loading FAISS index, rebuilding: {e}")
            shutil.rmtree(FAISS_INDEX_DIR)
            FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
            init_doc = Document(page_content="init", metadata={"link": "init"})
            store = await FAISS.afrom_documents([init_doc], embeddings)
            store.save_local(str(FAISS_INDEX_DIR))
            docs_list = [init_doc]
    else:
        logging.info(f"No existing FAISS index at {FAISS_INDEX_DIR}, creating new.")
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        init_doc = Document(page_content="init", metadata={"link": "init"})
        store = await FAISS.afrom_documents([init_doc], embeddings)
        store.save_local(str(FAISS_INDEX_DIR))
        docs_list = [init_doc]

    answer_file = FAISS_INDEX_DIR_ANSWER/"index.faiss"
    if answer_file.exists():
        try:
            logging.info(f"Loading FAISS-answer index from {FAISS_INDEX_DIR_ANSWER}")
            store_answer = FAISS.load_local(
                str(FAISS_INDEX_DIR_ANSWER),
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logging.error(f"Error loading FAISS-answer index, rebuilding: {e}")
            shutil.rmtree(FAISS_INDEX_DIR_ANSWER)
            FAISS_INDEX_DIR_ANSWER.mkdir(parents=True, exist_ok=True)
            store_answer = await FAISS.afrom_documents([Document(page_content="init", metadata={"link":"init"})], embeddings)
            store_answer.save_local(str(FAISS_INDEX_DIR_ANSWER))
    else:
        logging.info(f"No existing FAISS-answer index at {FAISS_INDEX_DIR_ANSWER}, creating new.")
        FAISS_INDEX_DIR_ANSWER.mkdir(parents=True, exist_ok=True)
        store_answer = await FAISS.afrom_documents([Document(page_content="init", metadata={"link":"init"})], embeddings)
        store_answer.save_local(str(FAISS_INDEX_DIR_ANSWER))

async def setup_retrievers():
    global retriever, retriever_answer
    # store и store_answer уже созданы после initialize_faiss()
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 9}
    )
    retriever_answer = store_answer.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.6}
    )



logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

async def clean_text(full_text: str) -> str:
    FOOTER_PATTERNS = [
        r'РТУ МИРЭА',
        r'Система менеджмента качества обучения',
        r'СМКО МИРЭА 8\.1/03\.ДП\.01-21',
        r'стр\.\s*\d+\s*из\s*\d+',
        r'Минобрнауки России',
    ]
    lines = full_text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not any(re.search(pattern, line) for pattern in FOOTER_PATTERNS):
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

async def group_chars_by_line(chars, y_tolerance=3):
    lines_dict = {}
    for c in chars:
        y = round(c['top'] / y_tolerance) * y_tolerance
        lines_dict.setdefault(y, []).append(c)
    lines_info = []
    for y, clist in lines_dict.items():
        clist.sort(key=lambda c: c['x0'])
        line_text = "".join(c['text'] for c in clist).strip()
        lines_info.append((y, clist, line_text))
    lines_info.sort(key=lambda x: x[0])
    return lines_info

async def is_line_bold(chars_of_line, threshold=0.5):
    if not chars_of_line:
        return False
    bold_count = sum(1 for c in chars_of_line if 'bold' in c.get('fontname', '').lower())
    return (bold_count / len(chars_of_line)) >= threshold

async def is_valid_header(line: str) -> bool:
    line = line.strip()
    # Тип 1: без точки после числа
    m1 = re.match(r'^(\d+)\s+([A-ZА-ЯЁ\s,]+)[\s,.:;!?]*$', line)
    if m1:
        text_part = m1.group(2)
        stripped = re.sub(r'[\s,]+', '', text_part)
        if stripped.upper() == stripped:
            return True
    # Тип 2: со знаком точки после числа
    m2 = re.match(r'^(\d+(?:\.\d+)*)\.\s*(.+)$', line)
    if not m2:
        return False
    number_part = m2.group(1)
    text_part = m2.group(2)
    if number_part.count('.') > 1:
        return text_part.strip().upper() == text_part.strip()
    elif '.' in number_part:
        return text_part.strip().upper() == text_part.strip() or text_part.strip()[0].isupper()
    else:
        return text_part.strip()[0].isupper() or text_part.strip().upper() == text_part.strip()

async def extract_blocks(pdf_path: str) -> dict:
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        headings_type1 = []
        headings_dot = []   
        for page in pdf.pages:
            txt = page.extract_text() or ""
            full_text += txt + "\n"
        full_text = await clean_text(full_text)
        
        for page in pdf.pages:
            chars = page.chars
            lines_info = await group_chars_by_line(chars, y_tolerance=3)
            for (_, clist, line_text) in lines_info:
                if await is_valid_header(line_text) and await is_line_bold(clist, threshold=0.8):
                    if re.match(r'^\d+\s+[A-ZА-ЯЁ\s,]+[\s,.:;!?]*$', line_text):
                        headings_type1.append(line_text.strip())
                    else:
                        headings_dot.append(line_text.strip())
        if pdf_path == f'{MODEL_CACHE}/STO_Rukovodstvo-po-kachestvu-obrazovaniya_2024.pdf':
            if headings_type1:
                headings = headings_type1
                logging.info(f"Используем заголовки первого типа (без точки): {headings}")
            else:
                headings = headings_dot
                logging.info(f"Используем заголовки с точкой: {headings}")
        else:
            headings = headings_type1 + headings_dot
            logging.info(f"Найдены заголовки: {headings}")
        
    
    if not headings:
        logging.warning("Не удалось найти заголовки (по шаблону и жирности).")
        return {}
    
    blocks = {}
    for i, heading in enumerate(headings):
        start = full_text.find(heading)
        if start == -1:
            logging.info(f"Заголовок '{heading}' не найден в полном тексте, пропускаем.")
            continue
        if i + 1 < len(headings):
            end = full_text.find(headings[i+1], start)
            if end == -1:
                end = len(full_text)
        else:
            end = len(full_text)
        block_text = full_text[start:end].strip()
        blocks[heading] = block_text
    blocks['Заголовки'] = 'Заголовки документа: '
    blocks['Заголовки']+=",".join(headings)
    return blocks

async def format_table(df: pd.DataFrame) -> str:
    table_str = ""
    headers = " | ".join(str(col).strip() if col else "" for col in df.columns.tolist())
    table_str += f"{headers} |\n"
    separator = " | ".join(['---'] * len(df.columns))
    table_str += f"{separator} |\n"
    for _, row in df.iterrows():
        row_data = " | ".join(str(item).strip() if item else "" for item in row)
        table_str += f"{row_data} |\n"
    return table_str.strip()

async def parse_tables(pdf_path: str) -> list:
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                if table:
                    try:
                        df = pd.DataFrame(table[1:], columns=table[0])
                    except Exception as e:
                        logging.error(f"Ошибка при создании DataFrame на странице {page_num}: {e}")
                        continue
                    table_formatted = await format_table(df)
                    tables.append({'page': page_num, 'table': table_formatted})
    return tables

async def associate_tables_to_blocks(blocks: dict, tables: list) -> dict:
    blocks_with_tables = {}
    block_keys = list(blocks.keys())
    for i, key in enumerate(block_keys):
        blocks_with_tables[key] = blocks[key]
        if i < len(tables):
            table_text = tables[i]['table']
            if table_text:
                blocks_with_tables[key] += "\n\n" + table_text
    return blocks_with_tables


async def create_json(file_name: str, blocks: dict):
    json_structure = {file_name: {}}
    for i, (heading, text) in enumerate(blocks.items(), 1):
        match = re.match(r'^\d+\s+(.*)$', heading)
        section_title = match.group(1) if match else heading
        json_key = f"{i}. {section_title}"
        text_cleaned = ' '.join(text.split())
        json_structure[file_name][json_key] = text_cleaned
    out_name = f"{file_name}.json"
    with open(out_name, 'w', encoding='utf-8') as f:
        json.dump(json_structure, f, ensure_ascii=False, indent=4)
    logging.info(f"JSON-файл '{out_name}' успешно создан.")


# 7. Основная функция обработки PDF

async def process_pdf(pdf_path: str) -> List[str]:
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    blocks= await extract_blocks(pdf_path)
    if not blocks:
        logging.info("Заголовки не найдены. Сохраняем весь документ как один блок.")
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += (page.extract_text() or "") + "\n"
        full_text = await clean_text(full_text)
        blocks = {"Документ": full_text}
    tables = await parse_tables(pdf_path)
    blocks_with_tables = await associate_tables_to_blocks(blocks, tables)
    try:
        await create_json(file_name, blocks_with_tables)
    except Exception as e:
        logging.error(f"Ошибка при создании JSON-файла: {e}")
    logging.info(f"Обработка файла {pdf_path} завершена.")
    return list(blocks_with_tables.values())

async def read_pdf(file_name: str):
        global bm_retriever
        global docs_list
        global ensemble_retriever
        pages = await process_pdf(file_name)
        for i in range(len(pages)):
            tmp = pages[i]
            tmp = tmp.replace(' -\n', '')
            tmp = tmp.replace('-\n', '')
            tmp = tmp.replace('Т\n', '')
            tmp = tmp.replace('____________', '')
            tmp = tmp.replace('___', '')
            tmp = tmp.replace('_______________', '')
            tmp = tmp.replace('_______________________________________________________________', '')
            tmp = tmp.replace('___________________', '')
            #tmp = pattern.sub('', tmp)
            pages[i] = tmp
        chunks_docs = [
            Document(page_content=pages[i], metadata={"link": file_name})
            for i in range(len(pages))
        ]

        start_add = time.time()
        await store.aadd_texts(
            [doc.page_content for doc in chunks_docs],
            metadatas=[doc.metadata for doc in chunks_docs],
            embeddings=embeddings
        )
        end_add = time.time()
        logging.info(f"Время добавления документов в FAISS: {end_add - start_add:.2f} секунд")

        docs_list.extend(chunks_docs)
        logging.info(f"Документ {file_name} успешно обработан и добавлен в индекс.")

model_name = "Qwen/Qwen3-4B"
cache_dir = MODEL_CACHE

tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map="cuda",
        use_fast=True,
        trust_remote_code=True
    )
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        device_map="cuda",
        cache_dir=cache_dir,
        trust_remote_code=True,
        use_safetensors=True

    )
def generate_ranking(tokenizer, model, question, context, type_):
    if type_ == '0':
        messages = [ {"role": "system", "content": """Вы — ранжировщик текстовых фрагментов в системе RAG (Retrieval-Augmented Generation).\n
        Пожалуйста, предоставьте результат в формате словаря Python, где ключ — номер блока, а значение — оценка релевантности блока. Также добавь ключ short_answer. Если предоставленного контекста достаточно для ответа, в качестве значения помести ответ на основе контекста. Сами блоки (Блок_1, Блок_2 и т.п.) указывать в ответе не нужно - это метаинформация, чтобы тебе было удобнее ориентироваться в массиве текстов, а пользователю она ничего не скажет. Ответ должен быть в соответствии с правилами русского языка и в официальном стиле. Иначе, если в представленном контексте информации нет, то значение ключа будет 0.\n
        Вам будет предоставлен запрос и несколько извлеченных текстовых блоков в обрезанном виде, связанных с этим запросом.\n
        Ваша задача — оценить и присвоить каждому блоку оценку на основе его предполагаемой релевантности предоставленному запросу.\n
        Думайте шаг за шагом по-русски\n
        Объясните свои доводы в нескольких предложениях, ссылаясь на конкретные пункты из контекста, чтобы обосновать свою оценку.\n
        Пожалуйста, выполни следующие шаги:\n
        1. **Анализ**: Проанализируй приведенные текстовые блоки, выявив ключевую информацию и её отношение к запросу.\n
        Определите, предоставляет ли отдельный блок прямой ответ, частичную информацию или фоновый контекст, связанный с запросом.\n
        Избегайте предположений — основывайтесь исключительно на предоставленном содержимом.\n
        2. **Оценка релевантности** (от 0 до 1 с шагом 0.1):\n
        - 0 = Полностью нерелевантно: Блок не имеет никакого отношения к запросу.\n
        - 0.1 = Практически нерелевантно: Очень слабая или неясная связь с запросом.\n
        - 0.2 = Очень слабо релевантно: Минимальная или косвенная связь.\n
        - 0.3 = Слабо релевантно: Затрагивает незначительный аспект запроса без существенных деталей.\n
        - 0.4 = Отчасти релевантно: Содержит частичную информацию, несколько связанную с запросом, но не полную.\n
        - 0.5 = Умеренно релевантно: Отвечает на запрос, но с ограниченной или частичной релевантностью.\n
        - 0.6 = Достаточно релевантно: Предоставляет полезную информацию, хотя и без глубины или специфики.\n
        - 0.7 = Релевантно: Явно связано с запросом, предлагая существенную, но не полностью исчерпывающую информацию.\n
        - 0.8 = Очень релевантно: Сильно связано с запросом и предоставляет значительную информацию.\n
        - 0.9 = Высоко релевантно: Почти полностью отвечает на запрос с детальной и конкретной информацией.\n
        - 1 = Абсолютно релевантно: Прямо и полностью отвечает на запрос со всей необходимой специфической информацией.\n\n
        3. **Дополнительные рекомендации**:\n
        - **Объективность**: Оценивайте блоки, основываясь только на их содержимом относительно запроса.\n
        - **Ясность**: Будьте четкими и лаконичными в ваших обоснованиях.\n
        - **Без предположений**: Не делайте выводов, выходящих за рамки явно указанного в блоке."""}, 

                {"role": "user", "content": str({'Question':question, 'context': context})}
            ]
    elif type_ == '1':
        messages = [ {"role": "system", "content": """Вы — оценщик ответов на вопросы в системе RAG (Retrieval-Augmented Generation).\n
                      На вход вам подается ответ и вопрос, на который этот ответ теориетически отвечает.\n
                      Думайте шаг за шагом по-русски\n
                      Пожалуйста, предоставьте оценку релевантности предоставленного ответа, отражающая его объяснительную часть. Выведи только число без лишних слов.\n
                      Думайте шаг за шагом для определения оценки.\n
                      Оценка релевантности равна 1, если ответ полностью отвечает на вопрос и не содержит ничего лишнего. 
                      Оценка релевантности равно 0 в остальных случаях, даже если ответ частично отвечает на вопрос."""}, 

                {"role": "user", "content": str({'Question':question, 'context': context})}
            ]
    elif type_ == '2':
        messages = [ {"role": "system", "content": """Ты - помощник по поиску информации, который использует данные из контекста для полного, но ёмкого ответа на вопросы или запросы. Словосочетание 'Не знаю' можно применить только если исходя из контекста информации вообще нет, если какая-то информация есть, то ты не имеешь права его применять. В тех случаях, когда ты можешь его применить, то отвечай только 'Не знаю' без лишних дополнений, рассуждений и возвращения контекста. Ответь четко и кратко на поставленный вопрос или запрос, основываясь на контексте, без лишней информации и словосочетаний 'Ответ' и 'Не знаю'.Если тебе написали 'Привет' или попросили рассказать о возможностях, то коротко расскажи о себе.\n"""}, 
                {"role": "user", "content": str({'Question':question, 'context': context})}
            ]
    if type_=='0' or type_=='1':
        text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
                enable_thinking=True,
                use_trition=True,
            )
    elif type_=='2':
        text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
                enable_thinking=False,
                use_trition=False,
            )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    if type_=='0' or type_=='1':
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,
            do_sample=True, temperature=0.6, top_k=20, min_p=0, top_p=0.95, num_beams = 1, bos_token_id=151643,
    eos_token_id=[
        151645,
        151643
    ],
    pad_token_id= 151643,
        )
    elif type_=='2':
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=True, temperature=0.7, top_k=20, min_p=0, top_p=0.8, num_beams = 1,bos_token_id=151643,
    eos_token_id=[
        151645,
        151643
    ],
    pad_token_id= 151643
        )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print(thinking_content)
    return content

async def answer_sbert(question: str) -> dict:
        start_retriever = time.time()
        context_docs = await retriever.ainvoke(question)
        torch.cuda.empty_cache()
        end_retriever = time.time()
        print(f"Время выполнения retriever: {end_retriever - start_retriever:.2f} секунд")
        
        filtered_docs = [doc for doc in context_docs if doc.metadata.get("link") != "init"]

        print(f"Количество найденных документов: {len(filtered_docs)}")
        for idx, doc in enumerate(filtered_docs):
            print(f"\nДокумент {idx + 1}:")
            print(f"Источник: {doc.metadata.get('link', 'Неизвестно')}")
            print(f"Содержание (первые 500 символов): {doc.page_content[:500]}...")
        context = [{f"""Блок_{idx+1}""": str(doc.page_content)} for idx, doc in enumerate(filtered_docs)]
        docs = [Document(page_content=doc.page_content, metadata={**doc.metadata, "orig_block": idx + 1}) for idx, doc in enumerate(filtered_docs)]
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " "],
            chunk_size=3000,
            chunk_overlap=500
        )
        chunked_docs = splitter.split_documents(docs)
        temp_store = FAISS.from_documents(chunked_docs, embeddings, normalize_L2=True)

        temp_retriever = temp_store.as_retriever(search_type="similarity", search_kwargs={"k": 7, "include_scores": True} )
        sub_chunks = await temp_retriever.ainvoke(question)
        print(sub_chunks)
        context_sub_chunks = []
        for chunk_idx, chunk in enumerate(sub_chunks, start=1):
            orig = chunk.metadata.get("orig_block", None)
            context_sub_chunks.append({
                f"Блок_{orig}": chunk.page_content
            })

        logging.info(f'Извлеченные фрагменты:\n{context_sub_chunks}')

        
        content = await asyncio.to_thread(generate_ranking, tokenizer, model,question, context_sub_chunks, '0')
        dict_scores = ast.literal_eval(content[content.find('{'):content.find('}')+1])
        dict_scores_rank = dict_scores.copy()
        dict_scores_rank['short_answer'] = 0.0
        logging.info(f'Ранги: {dict_scores_rank}')
        dict_scores_rank= dict(sorted(dict_scores_rank.items(), key=lambda v: v[1], reverse=True))
        
        list_for_top_answer = [k if v>=0.8 else None for k,v in dict_scores_rank.items()][:4]
        list_for_top_answer = set([x for x in list_for_top_answer if x is not None])
        if len(list_for_top_answer) == 0:
            list_for_top_answer = set([k if v>=0.6 else None for k,v in dict_scores_rank.items()][:4])
        logging.info(f'Список источников для ответа: {list_for_top_answer}')

        content_for_fast_answer = set([name.get(good_name) for good_name in list_for_top_answer for name in context_sub_chunks ])
        content_for_fast_answer = set([x for x in content_for_fast_answer if x is not None])

        content_for_answer = set([name.get(good_name) for good_name in list_for_top_answer for name in context ])
        content_for_answer = set([x for x in content_for_answer if x is not None])

        if (dict_scores.get('short_answer')) != (float('0') or int('0')):
            filtered_docs = [doc for doc in filtered_docs if doc.page_content in content_for_answer]
            print({"query": question, "answer": dict_scores['short_answer'].replace('<|begin_of_text|>', '').strip(), "context": filtered_docs})
            return {"query": question, "answer": dict_scores['short_answer'].replace('<|begin_of_text|>', '').strip(), "context": filtered_docs}

        
        else:
            # 3. Очищаем кеш CUDA

            answer = await asyncio.to_thread(generate_ranking, tokenizer, model,question, context, '0')

            filtered_docs = [doc for doc in filtered_docs if doc.page_content in context]
            return {"answer": answer.replace('<|begin_of_text|>', '').strip(), "context": filtered_docs}

async def save_answer(message_id, chat_id, message_text):
        answer = Document(page_content=str(message_text), metadata={"message_id": message_id, "chat_id": chat_id})
        start_add = time.time()
        await store_answer.aadd_texts(
            [answer.page_content],
            metadatas=[answer.metadata],
            embeddings=embeddings
        )
        end_add = time.time()
        print(f"Время добавления документов в FAISS: {end_add - start_add:.2f} секунд")
        await save_faiss_answer()
        logging.info(f"Ответ {message_id} успешно обработан и добавлен в индекс.")

async def express_answer(question: str) -> dict:

        start_retriever = time.time()
        answer = await retriever_answer.ainvoke(question)
        logging.info(answer)
        if len(answer) == 0:
            return None
        torch.cuda.empty_cache()
        end_retriever = time.time()
        logging.info(f"Время выполнения retriever: {end_retriever - start_retriever:.2f} секунд")

        filtered_docs = [doc for doc in answer if doc.metadata.get("link") != "init"]

        logging.info(f"Количество найденных документов: {len(filtered_docs)}")
        for idx, doc in enumerate(filtered_docs):
            logging.info(f"\nДокумент {idx + 1}:")
            logging.info(f"Источник: {doc.metadata.get('link', 'Неизвестно')}")
            logging.info(f"Содержание (первые 500 символов): {doc.page_content[:500]}...")

        context = {f"""Ответ: """: str(doc.page_content) for doc in filtered_docs}
        
        content = await asyncio.to_thread(generate_ranking, tokenizer, model,question, context, '1')
        logging.info(f'Content: {content}')

        if int(content.replace('\n', '')) ==1:
            return {'answer': [doc.page_content for doc in answer if doc.metadata.get("link") != "init"][0], 'message_id': [doc.metadata.get("message_id") for doc in answer][0], 'chat_id': [doc.metadata.get("chat_id") for doc in answer][0]}
        else: 
            return None

async def fill_index_folder(folder: str, batch_size: int = 1):
    if not os.path.isdir(folder):
        raise ValueError(f"Папка не найдена: {folder}")

    pdf_files = [fn for fn in os.listdir(folder) if fn.lower().endswith(".pdf")]
    logging.info(f"Найдено {len(pdf_files)} PDF в {folder} для индексирования.")
    for file_name in pdf_files:
        full_path = os.path.join(folder, file_name)
        logging.info(f"Обрабатываем {full_path}...")
        # Предполагается, что read_pdf умеет принимать полный путь
        await read_pdf(full_path)

    # Сохранить индекс
    store.save_local(FAISS_INDEX_DIR)
    logging.info(f"Индекс обновлён и сохранён в {FAISS_INDEX_DIR}")
