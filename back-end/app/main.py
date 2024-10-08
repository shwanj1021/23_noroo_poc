from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from enum import Enum
import json

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import func

app = FastAPI(redoc_url=None)
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

class Query(BaseModel):
    question : str
    
class Test(str, Enum):
    qa = "QA"
    q = "Q"
    a = "A"
    
class FtModel(str, Enum):
    m0 = "gpt35"
    m1 = "gpt4"
    m2 = "QA"
    m3 = "QA summary"

@app.get("/")
async def index():
    return "Not found"

@app.post("/ans_stream")
async def answer_stream(query: Query):       
    question = query.question
    
    def generate():
        tokens = func.num_tokens_from_string(question)
        if tokens > 8000:
            yield f"data: 질문의 최대 길이는 8000토큰 입니다. 질문에 {tokens}개의 토큰이 사용되었습니다. 질문의 길이를 줄여주세요.\n\n"
        else:
            docs = func.search_documents(question)
            context = []
            sour_doc = []
            for i, doc in enumerate(docs[0:3]):
                doc_type = doc.get('DocType')
                if doc_type == "qa_db_data": ### QA
                    context.append(f"context {i} : {doc.get('answer')}")
                    sour_doc.append({'doc_type': doc['DocType'], 'reference' : f"QA data / idx {doc['BoardIdx']}", 'content' : doc['answer']})
                elif doc_type == "qa_summary": ### QA summary
                    context.append(f"context {i} : {doc.get('QA')}")
                    sour_doc.append({'doc_type': doc['DocType'], 'content' : doc['QA']})
                else: ### pdf
                    context.append(f"context {i} : {doc.get('page_content')}")
                    sour_doc.append({'doc_type': doc['DocType'], 'reference' : f"{doc['fileName']} / page {doc['pageNo']}", 'file_name' : doc['fileName'], 'page_no' : doc['pageNo']})
        
            messages = func.make_messages(question, context)

            tokens = func.num_tokens_from_messages(messages)
            if tokens > 16000:
                yield f"data: 답변 가능한 최대 메시지 길이는 16000토큰 입니다. 메시지에 {tokens}개의 토큰이 사용되었습니다. 메시지 길이를 줄여주세요.\n\n"
            else:
                try:
                    api_response = func.get_stream_completion_from_messages(messages)
                    
                    yield f"data: {json.dumps(sour_doc, ensure_ascii=False)}\n\n"                
                    
                    all_answer = ""
                    for line in api_response.iter_lines():
                        if line:
                            data_str = line.decode('utf-8')[6:]
                            if data_str != "[DONE]":
                                data_dict = json.loads(data_str)
                                if len(data_dict.get('choices')) != 0:
                                    if data_dict.get('choices')[0].get('delta').get('content') is not None:
                                        answer = data_dict['choices'][0]['delta']['content']
                                        all_answer += answer
                                        if answer != None:
                                            yield f"data: {answer}\n\n"
                except:
                    yield f"data: 오류가 발생했습니다.\n\n"
                            
                yield f"""data: {json.dumps({"answer": all_answer}, ensure_ascii=False)}\n\n"""
                    
    return StreamingResponse(generate(), media_type='text/event-stream')

# 노루 통합
@app.post("/ans_qa_pdf")
async def answer_qa_and_pdf(query: Query):       
    question = query.question
    
    def generate():
        tokens = func.num_tokens_from_string(question)
        if tokens > 8000:
            yield f"data: 질문의 최대 길이는 8000토큰 입니다. 질문에 {tokens}개의 토큰이 사용되었습니다. 질문의 길이를 줄여주세요.\n\n"
        else:       
            messages, sour_doc = func.search_doc_and_create_messages_from_question(question)
            
            # yield f"data: {json.dumps(messages, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps(sour_doc, ensure_ascii=False)}\n\n"

            tokens = func.num_tokens_from_messages(messages)
            # yield f"data: {tokens}\n\n"
            if tokens > 16000:
                yield f"data: 답변 가능한 최대 메시지 길이는 16000토큰 입니다. 메시지에 {tokens}개의 토큰이 사용되었습니다. 메시지 길이를 줄여주세요.\n\n"
            else:
                try:
                    api_response = func.get_stream_completion_from_messages(messages)
                    
                    all_answer = ""
                    for line in api_response.iter_lines():
                        if line:
                            data_str = line.decode('utf-8')[6:]
                            if data_str != "[DONE]":
                                data_dict = json.loads(data_str)
                                if len(data_dict.get('choices')) != 0:
                                    if data_dict.get('choices')[0].get('delta').get('content') is not None:
                                        answer = data_dict['choices'][0]['delta']['content']
                                        all_answer += answer
                                        if answer != None:
                                            yield f"data: {answer}\n\n"
                    yield f"""data: {json.dumps({"answer": all_answer}, ensure_ascii=False)}\n\n"""
                except:
                    yield f"""data: 적절한 질문을 입력하세요.\n\n"""

    return StreamingResponse(generate(), media_type='text/event-stream')

# Run with Uvicorn or another ASGI server

@app.post("/qa_documents")
async def qa_documents(question: str, mode: Test, k:int=50):
    return func.qa_documents(question, mode, k)

@app.post("/fine_tune")
async def finetune(question: str, model: FtModel = "QA"):
    messages = [{'role':'user', 'content':question}]
    result = func.get_completion_from_messages(messages, model.value) 
    
    return result.get('choices')[0].get('message').get('content')


@app.post("/ans_finetune")
async def answer_finetune(query: Query, model: FtModel="QA"):       
    question = query.question
    
    def generate():
        tokens = func.num_tokens_from_string(question)
        if tokens > 2000:
            yield f"data: 질문의 최대 길이는 2000토큰 입니다. 질문에 {tokens}개의 토큰이 사용되었습니다. 질문의 길이를 줄여주세요.\n\n"
        else:
            docs = func.search_documents(question)
            context = []
            sour_doc = []
            for i, doc in enumerate(docs[0:2]):
                context.append(f"context {i} : {doc.get('page_content')}")
                sour_doc.append({'doc_type': doc['DocType'], 'reference' : f"{doc['fileName']} / page {doc['pageNo']}", 'file_name' : doc['fileName'], 'page_no' : doc['pageNo']})
        
            messages = func.make_messages(question, context)

            tokens = func.num_tokens_from_messages(messages)
            if tokens > 4000:
                yield f"data: 답변 가능한 최대 메시지 길이는 4000토큰 입니다. 메시지에 {tokens}개의 토큰이 사용되었습니다. 메시지 길이를 줄여주세요.\n\n"
            else:
                api_response = func.get_stream_completion_from_messages(messages, model)
                
                yield f"data: {json.dumps(sour_doc, ensure_ascii=False)}\n\n"                
                
                all_answer = ""
                for line in api_response.iter_lines():
                    if line:
                        data_str = line.decode('utf-8')[6:]
                        if data_str != "[DONE]":
                            data_dict = json.loads(data_str)
                            if len(data_dict.get('choices')) != 0:
                                if data_dict.get('choices')[0].get('delta').get('content') is not None:
                                    answer = data_dict['choices'][0]['delta']['content']
                                    all_answer += answer
                                    if answer != None:
                                            yield f"data: {answer}\n\n"
                yield f"""data: {json.dumps({"answer": all_answer}, ensure_ascii=False)}\n\n"""

    return StreamingResponse(generate(), media_type='text/event-stream')

# 기술문서
@app.post("/ans_pdf")
async def answer_pdf(query: Query):       
    question = query.question
    
    def generate():
        tokens = func.num_tokens_from_string(question)
        if tokens > 8000:
            yield f"data: 질문의 최대 길이는 8000토큰 입니다. 질문에 {tokens}개의 토큰이 사용되었습니다. 질문의 길이를 줄여주세요.\n\n"
        else:       
            messages, sour_doc = func.pdf_search_and_create_messages(question)
            
            # yield f"data: {json.dumps(messages, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps(sour_doc, ensure_ascii=False)}\n\n"

            tokens = func.num_tokens_from_messages(messages)
            # yield f"data: {tokens}\n\n"
            if tokens > 16000:
                yield f"data: 답변 가능한 최대 메시지 길이는 16000토큰 입니다. 메시지에 {tokens}개의 토큰이 사용되었습니다. 메시지 길이를 줄여주세요.\n\n"
            else:
                try:
                    api_response = func.get_stream_completion_from_messages(messages)
                    
                    all_answer = ""
                    for line in api_response.iter_lines():
                        if line:
                            data_str = line.decode('utf-8')[6:]
                            if data_str != "[DONE]":
                                data_dict = json.loads(data_str)
                                if len(data_dict.get('choices')) != 0:
                                    if data_dict.get('choices')[0].get('delta').get('content') is not None:
                                        answer = data_dict['choices'][0]['delta']['content']
                                        all_answer += answer
                                        if answer != None:
                                            yield f"data: {answer}\n\n"
                    yield f"""data: {json.dumps({"answer": all_answer}, ensure_ascii=False)}\n\n"""
                except:
                    yield f"""data: 적절한 질문을 입력하세요.\n\n"""

    return StreamingResponse(generate(), media_type='text/event-stream')

#상담 챗봇
@app.post("/ans_qa")
async def answer_qa(query: Query):       
    question = query.question
    
    def generate():
        tokens = func.num_tokens_from_string(question)
        if tokens > 8000:
            yield f"data: 질문의 최대 길이는 8000토큰 입니다. 질문에 {tokens}개의 토큰이 사용되었습니다. 질문의 길이를 줄여주세요.\n\n"
        else:       
            messages, sour_doc = func.qa_search_and_create_messages(question)
            
            # yield f"data: {json.dumps(messages, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps(sour_doc, ensure_ascii=False)}\n\n"

            tokens = func.num_tokens_from_messages(messages)
            # yield f"data: {tokens}\n\n"
            if tokens > 16000:
                yield f"data: 답변 가능한 최대 메시지 길이는 16000토큰 입니다. 메시지에 {tokens}개의 토큰이 사용되었습니다. 메시지 길이를 줄여주세요.\n\n"
            else:
                try:
                    api_response = func.get_stream_completion_from_messages(messages)
                    
                    all_answer = ""
                    for line in api_response.iter_lines():
                        if line:
                            data_str = line.decode('utf-8')[6:]
                            if data_str != "[DONE]":
                                data_dict = json.loads(data_str)
                                if len(data_dict.get('choices')) != 0:
                                    if data_dict.get('choices')[0].get('delta').get('content') is not None:
                                        answer = data_dict['choices'][0]['delta']['content']
                                        all_answer += answer
                                        if answer != None:
                                            yield f"data: {answer}\n\n"
                    yield f"""data: {json.dumps({"answer": all_answer}, ensure_ascii=False)}\n\n"""
                except:
                    yield f"""data: 적절한 질문을 입력하세요.\n\n"""

    return StreamingResponse(generate(), media_type='text/event-stream')