import os
import requests
import uuid
import tiktoken

AZURE_OPENAI_KEY = "REMOVED_AZURE_OPENAI_KEY"
AZURE_OPENAI_ENDPOINT = "https://noroo-poc-openai-sweden.openai.azure.com/"
AZURE_OPENAI_API_VERSION = "2023-12-01-preview"

AZURE_SEARCH_KEY = "REMOVED_AZURE_SEARCH_KEY"
AZURE_SEARCH_ENDPOINT = "https://noroo-poc-search.search.windows.net"
# AZURE_SEARCH_KEY = "REMOVED_AZURE_SEARCH_KEY_ALT"
# AZURE_SEARCH_ENDPOINT = "https://chatgpt-abm-cs-kr.search.windows.net"
AZURE_SEARCH_API_VERSION = "2023-11-01"

DEPLOYMENT_GPT35 = "GPT35_16K_0613"
DEPLOYMENT_GPT4 = "GPT4_32K_0613"
DEPLOYMENT_EMBED = "EMBEDDING_ADA_002_2"

OPENAI_HEADERS = {'Content-Type': 'application/json','api-key': AZURE_OPENAI_KEY}
OPENAI_PARAMS = {'api-version': AZURE_OPENAI_API_VERSION}

SEARCH_HEADERS = {'Content-Type': 'application/json','api-key': AZURE_SEARCH_KEY}
SEARCH_PARAMS = {'api-version': AZURE_SEARCH_API_VERSION}


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += num_tokens_from_string(value)
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def get_embedding_vector(context):
  input = {
      "input" : context
  }
  r = requests.post(AZURE_OPENAI_ENDPOINT + "/openai/deployments/" + DEPLOYMENT_EMBED + "/embeddings",
                    json=input, headers=OPENAI_HEADERS, params=OPENAI_PARAMS)
  return r.json()['data'][0]['embedding']

def get_completion_from_messages(messages, model="gpt35", temperature=0):
    if model == "gpt4":
        deployment=DEPLOYMENT_GPT4
    elif model == "gpt35":
        deployment=DEPLOYMENT_GPT35
    elif model == "QA":
        deployment= "GPT35_FINETUNING_QA"
    elif model == "QA summary":
        deployment= "GPT35_FINETUNING_QASUMMARY"
    payloads = {
      "temperature" : temperature,
      "messages" : messages
    }
    r = requests.post(AZURE_OPENAI_ENDPOINT + "/openai/deployments/" + deployment + "/chat/completions",
                    json=payloads, headers=OPENAI_HEADERS, params=OPENAI_PARAMS)
    return r.json()

def get_stream_completion_from_messages(messages, model="gpt35", temperature=0.4):
    if model == "gpt4":
        deployment=DEPLOYMENT_GPT4
    elif model == "gpt35":
        deployment=DEPLOYMENT_GPT35
    elif model == "QA":
        deployment= "GPT35_FINETUNING_QA"
    elif model == "QA summary":
        deployment= "GPT35_FINETUNING_QASUMMARY"
    payloads = {
      "temperature" : temperature,
      "messages" : messages,
      "stream": True
    }
    r = requests.post(AZURE_OPENAI_ENDPOINT + "/openai/deployments/" +deployment + "/chat/completions",
                    json=payloads, headers=OPENAI_HEADERS, params=OPENAI_PARAMS, stream=True)
    return r

def make_messages(question, context):
    qa_prompt_template = f"""아래 context(XML tag로 구분됨)를 바탕으로 question에 대한 답변을 작성하세요. 만약 context의 내용으로 답변할 수 없다면, 제공한 문서에서 적절한 정보를 찾을 수 없습니다. 라고 말하세요. 억지로 답변을 만들어내지 마세요.
    정보를 제공하는 안내원의 말투로 답변하세요. 모든 답변은 줄 바꿈으로 구분하도록 합니다.

    <context>
    {context}
    </context>

    question: {question}
    answer:"""

    return [{'role':'user', 'content':qa_prompt_template}]



def qa_documents(question, mode='QA', k=10):
    questionVector = get_embedding_vector(question)
    index_name = 'noroo-index'
    if mode == 'Q':
        query_payloads = {
              "select": "DocType, BoardIdx, BoardSubject, question, answer",
              "vectorQueries": [
                  {
                      
                      "fields": "quesVector",
                      "kind": "vector",
                      "exhaustive": True,
                      "vector": questionVector,
                      "k" :k
                  }
              ]
          }
    elif mode == 'A':
        query_payloads = {
              "select": "DocType, BoardIdx, BoardSubject, question, answer",
              "vectorQueries": [
                  {
                      
                      "fields": "ansVector",
                      "kind": "vector",
                      "exhaustive": True,
                      "vector": questionVector,
                      "k" : k
                  }
              ]
          }
    elif mode == 'QA':
        query_payloads = {
              "select": "DocType, BoardIdx, BoardSubject, question, answer",
              "vectorQueries": [
                  {
                      
                      "fields": "qaVector",
                      "kind": "vector",
                      "exhaustive": True,
                      "vector": questionVector,
                      "k" : k
                  }
              ]
          }
    r = requests.post(AZURE_SEARCH_ENDPOINT + "/indexes/" + index_name + "/docs/search",
                    json=query_payloads, headers=SEARCH_HEADERS, params=SEARCH_PARAMS)
    search_doc = r.json()['value']
    return search_doc

def pdf_documents(question, k=10):
    questionVector = get_embedding_vector(question)
    index_name = 'noroo-pdf-index'
    query_payloads = {
          "select": "DocType, page_content, fileName, pageNo",
          "vectorQueries": [
              {
                  "fields": "contentVector",
                  "kind": "vector",
                  "exhaustive": True,
                  "vector": questionVector,
                  "k" : k
              }
          ]
      }
    r = requests.post(AZURE_SEARCH_ENDPOINT + "/indexes/" + index_name + "/docs/search",
                    json=query_payloads, headers=SEARCH_HEADERS, params=SEARCH_PARAMS)
    search_doc = r.json()['value']
    return search_doc


def qasummary_documents(question, k=10):
    questionVector = get_embedding_vector(question)
    index_name = 'noroo-qasummary-index'
    query_payloads = {
          "select": "DocType, question, answer, QA",
          "vectorQueries": [
              {
                  "fields": "qaVector",
                  "kind": "vector",
                  "exhaustive": True,
                  "vector": questionVector,
                  "k" : k
              }
          ]
      }
    r = requests.post(AZURE_SEARCH_ENDPOINT + "/indexes/" + index_name + "/docs/search",
                    json=query_payloads, headers=SEARCH_HEADERS, params=SEARCH_PARAMS)
    search_doc = r.json()['value']
    return search_doc


# 문서 검색
def search_documents(question):
    # qa_res = qa_documents(question)
    # pdf_res = pdf_documents(question)
    # data = qa_res + pdf_res
    
    # sorted_data = sorted(data, key=lambda x: x['@search.score'], reverse=True)
    sorted_data = pdf_documents(question)
    return sorted_data


def search_doc_and_create_messages_from_question(question):
    qa_contexts = []
    pdf_contexts = []
    references = []
    
    pdf_docs = pdf_documents(question, k=2)
    
    qa_docs = qasummary_documents(question, k=3)
    for i, qa_doc in enumerate(qa_docs):
        qa_contexts.append(f"상담 요약 {i+1} : {qa_doc.get('QA')}")
        pdf_docs.extend(pdf_documents(qa_doc.get('answer'), k=1)) # QA, answer, question + QA 적용 검토

    for pdf_doc in pdf_docs:
        pdf_context = f"{pdf_doc['fileName']} / page {pdf_doc['pageNo']} : \n {pdf_doc['page_content']}"
        if pdf_context not in pdf_contexts:
            pdf_contexts.append(pdf_context)
            references.append({'doc_type': pdf_doc['DocType'], 'reference' : f"{pdf_doc['fileName']} / page {pdf_doc['pageNo']}", 'file_name' : pdf_doc['fileName'], 'page_no' : pdf_doc['pageNo']})
   
    references.insert(0, {'doc_type':'qa_summary', 'reference' : '상담 요약', 'content':qa_contexts})
    
    context = f"""
    {qa_contexts}
    
    제품 정보 문서 :
    {pdf_contexts}
    """
    qa_prompt_template = f"""아래 context(XML tag로 구분됨)에는 제품에 대한 '상담 요약'과 상담과 관련된 '제품 정보 문서'가 제공됩니다. 이 내용을 검토하고, 가장 적합한 정보만을 사용하여 question에 대한 구체적인 답변을 작성하세요. 
    만약 context의 내용으로 답변할 수 없다면, 제공한 문서에서 적절한 정보를 찾을 수 없습니다. 라고 말하세요. 억지로 답변을 만들어내지 마세요.
    정보를 제공하는 안내원의 말투로 답변하세요. 모든 답변은 줄 바꿈으로 구분하도록 합니다.

    <context>
    {context}
    </context>

    question: {question}
    answer:"""

    return [{'role':'user', 'content':qa_prompt_template}], references
    
    
    
    
def pdf_documents_2(question, k=10):
    questionVector = get_embedding_vector(question)
    index_name = 'noroo-pdf-index'
    query_payloads = {
          "search": question,
          "top" : k,
          "searchFields" : "page_content",
          "select": "DocType, page_content, fileName, pageNo",
          "vectorQueries": [
              {
                  "fields": "contentVector",
                  "kind": "vector",
                  "exhaustive": True,
                  "vector": questionVector,
                  "k" : k
              }
          ]
      }
    r = requests.post(AZURE_SEARCH_ENDPOINT + "/indexes/" + index_name + "/docs/search",
                    json=query_payloads, headers=SEARCH_HEADERS, params=SEARCH_PARAMS)
    search_doc = r.json()['value']
    return search_doc


def qasummary_documents_2(question, k=10):
    questionVector = get_embedding_vector(question)
    index_name = 'noroo-qasummary-index'
    query_payloads = {
          "search": question,
          "top" : k,
          "searchFields" : "QA",
          "select": "DocType, question, answer, QA",
          "vectorQueries": [
              {
                  "fields": "qaVector",
                  "kind": "vector",
                  "exhaustive": True,
                  "vector": questionVector,
                  "k" : k
              }
          ]
      }
    r = requests.post(AZURE_SEARCH_ENDPOINT + "/indexes/" + index_name + "/docs/search",
                    json=query_payloads, headers=SEARCH_HEADERS, params=SEARCH_PARAMS)
    search_doc = r.json()['value']
    return search_doc 

def pdf_search_and_create_messages(question):
    pdf_contexts = []
    references = []
    
    pdf_docs = pdf_documents_2(question, k=3)

    for pdf_doc in pdf_docs:
        pdf_context = f"{pdf_doc['fileName']} / page {pdf_doc['pageNo']} : \n {pdf_doc['page_content']}"
        if pdf_context not in pdf_contexts:
            pdf_contexts.append(pdf_context)
            references.append({'doc_type': pdf_doc['DocType'], 'reference' : f"{pdf_doc['fileName']} / page {pdf_doc['pageNo']}", 'file_name' : pdf_doc['fileName'], 'page_no' : pdf_doc['pageNo']})
    
    prompt_template = f"""아래 context(XML tag로 구분됨)를 바탕으로 question에 대한 답변을 작성하세요. 만약 context의 내용으로 답변할 수 없다면, 제공한 문서에서 적절한 정보를 찾을 수 없습니다. 라고 말하세요. 억지로 답변을 만들어내지 마세요.
    정보를 제공하는 안내원의 말투로 답변하세요. 모든 답변은 줄 바꿈으로 구분하도록 합니다.

    <context>
    {pdf_contexts}
    </context>

    question: {question}
    answer:"""

    return [{'role':'user', 'content':prompt_template}], references


def qa_search_and_create_messages(question):
    qa_contexts = []
    references = []
    
    qa_docs = qasummary_documents_2(question, k=5)
    for i, qa_doc in enumerate(qa_docs):
        qa_contexts.append(f"상담 요약 {i+1} : {qa_doc.get('QA')}")

    references.insert(0, {'doc_type':'qa_summary', 'reference' : '상담 요약', 'content':qa_contexts})
    
    prompt_template = f"""아래 context(XML tag로 구분됨)를 바탕으로 question에 대한 답변을 작성하세요. 만약 context의 내용으로 답변할 수 없다면, 제공한 문서에서 적절한 정보를 찾을 수 없습니다. 라고 말하세요. 억지로 답변을 만들어내지 마세요.
    정보를 제공하는 안내원의 말투로 답변하세요. 모든 답변은 줄 바꿈으로 구분하도록 합니다.

    <context>
    {qa_contexts}
    </context>

    question: {question}
    answer:"""

    return [{'role':'user', 'content':prompt_template}], references