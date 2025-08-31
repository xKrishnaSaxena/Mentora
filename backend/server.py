import cv2
import numpy as np
import base64
import asyncio
import google.generativeai as genai
from dotenv import load_dotenv
import os
from gtts import gTTS
import io
import json
import re
import logging
from aiohttp import web, WSMsgType
import jwt
import bcrypt
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone, timedelta
import aiohttp_cors
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ZILLIZ_CLOUD_URI = os.getenv("ZILLIZ_CLOUD_URI")
ZILLIZ_CLOUD_TOKEN = os.getenv("ZILLIZ_CLOUD_API_KEY")
JWT_SECRET = os.getenv("JWT_SECRET")
MONGODB_URI = os.getenv("MONGODB_URI")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env")
if not ZILLIZ_CLOUD_URI or not ZILLIZ_CLOUD_TOKEN:
    raise ValueError("❌ ZILLIZ_CLOUD_URI or ZILLIZ_CLOUD_API_KEY not found in .env")
if not JWT_SECRET:
    raise ValueError("❌ JWT_SECRET not found in .env")
if not MONGODB_URI:
    raise ValueError("❌ MONGODB_URI not found in .env")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(
    "gemini-1.5-flash",
    system_instruction=(
        "You are an AI assistant. Your task is to answer questions based on provided text or generate content as requested. "
        "For concise responses, use only spaces for formatting—no asterisks or special characters. "
        "For detailed explanations, provide responses in Markdown format and include Mermaid diagrams of flowchart or sequence diagram using ```mermaid code blocks when appropriate. "
        "If code is requested, use only JavaScript. "
        "Keep the tone human and avoid emojis. If unsure, say 'I don’t know'."
    ),
)

pdf_initialized = False
vector_store = None

def load_pdf_documents(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

async def initialize_pdf_rag():
    global pdf_initialized, vector_store
    if pdf_initialized:
        return
    collection_name = "foxlearner_ch6_embeddings"
    try:
        current_dir = os.path.dirname(__file__)
        pdf_path = os.path.join(current_dir, "lecs105.pdf")
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"❌ PDF file not found at: {pdf_path}")

        docs = load_pdf_documents(pdf_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GEMINI_API_KEY,
        )

        vector_store = Milvus.from_documents(
            documents=split_docs,
            collection_name=collection_name,
            embedding=embeddings,
            connection_args={
                "uri": ZILLIZ_CLOUD_URI,
                "token": ZILLIZ_CLOUD_TOKEN,
                "secure": True,
            },
        )
        pdf_initialized = True
        logger.info("PDF RAG initialized successfully")
        print(f"✅ Embedding complete. Stored in Milvus collection: {collection_name}")
    except Exception as e:
        logger.error(f"PDF RAG initialization failed: {str(e)}")

def clean_text(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text)
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    return text.strip()

async def text_to_speech(text: str):
    try:
        tts = gTTS(text=text, lang="en")
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_data = base64.b64encode(mp3_fp.read()).decode("utf-8")
        logger.info("Audio generated successfully")
        return audio_data
    except Exception as e:
        logger.error(f"Text-to-speech error: {str(e)}")
        return None

def process_frame(frame_data: str):
    try:
        frame_bytes = base64.b64decode(frame_data)
        frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode frame")
        return frame
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        raise

async def ask_gemini_with_image(frame, question: str):
    from PIL import Image
    try:
        logger.info(f"Processing question with image: {question}")
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        response = model.generate_content([question, pil_image])
        logger.info(f"Raw Gemini API response: {response.text}")
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return f"Error: {str(e)}"

async def handle_rag_mode(data):
    await initialize_pdf_rag()
    if not vector_store:
        return {"error": "RAG not initialized"}

    question = data["question"]
    history = data.get("history", [])
    conversation_context = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in history[-3:]])

    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an expert AI tutor helping students understand concepts from their textbook. 
The student has asked: "{question}"

### Textbook Context:
{context}

### Conversation History:
{conversation_context}

### Instructions:
1. Provide a detailed, step-by-step explanation using information from the textbook context.
2. If the concept spans multiple sections, synthesize information from different parts of the book.
3. Use clear examples and analogies.
4. For processes or systems, include a Mermaid diagram using ```mermaid code blocks.
5. Format with Markdown (headings, bullets, bold key terms).
6. If the context doesn't contain the answer, say: "This topic isn't covered in our textbook, but here's what I know..."

Remember: patient, thorough, encouraging.
"""
    response = model.generate_content(prompt)
    response_text = response.text or ""
    response_text_md = response_text

    return {
        "question": question,
        "text": clean_text(response_text_md),
        "detailed": response_text_md,
        "responseType": "rag",
        "mode": "rag",
        "context": context,
    }

async def handle_teach_mode(data):
    category = data.get("category", "")
    question = data["question"]
    history = data.get("history", [])
    prompt = f"""You are an expert tutor on {category}. Explain step-by-step and check for understanding.

Here is the conversation history:
"""
    for msg in history:
        prompt += f'{msg["role"]}: {msg["content"]}\n'

    prompt += """
Provide the next response.

If the user is asking a new question, start in small parts and ask if they understand after each part.
If they say 'yes', proceed; if 'no', go deeper and include a ```mermaid diagram when helpful.

Respond in JSON with:
- "concise": short (<= 2 sentences, plain text)
- "detailed": Markdown explanation

Return only the JSON object.
"""
    response = model.generate_content(prompt)
    response_text = (response.text or "").strip()
    try:
        response_json = json.loads(response_text)
        concise = response_json["concise"]
        detailed = response_json["detailed"]
    except json.JSONDecodeError:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                response_json = json.loads(response_text[start:end])
                concise = response_json["concise"]
                detailed = response_json["detailed"]
            except json.JSONDecodeError:
                logger.error("Failed to parse extracted JSON")
                concise = clean_text(response_text)
                detailed = "Error: Unable to generate detailed explanation."
        else:
            logger.error("No JSON object found in response")
            concise = clean_text(response_text)
            detailed = "Error: Unable to generate detailed explanation."
    return {
        "question": question,
        "text": concise,
        "detailed": detailed,
        "responseType": "answer",
        "mode": "teach",
    }

async def handle_learn_mode(data):
    logger.info(f"Handling learn mode with data keys: {list(data.keys())}")
    include_code = "code" in data.get("question", "").lower()
    # Allow learn mode without a frame (fallback to text-only)
    frame_b64 = data.get("frame")
    if frame_b64:
        frame = process_frame(frame_b64)
        prompt = f"""Analyze the provided image and answer this question in JSON with:
- "concise": short (<= 2 sentences, plain text).
- "detailed": Markdown{' with a code snippet only if requested' if not include_code else ''}, and end with a ```mermaid diagram.

Question: {data['question']}
Return only the JSON object, without extra text or backticks."""
        response_text = await ask_gemini_with_image(frame, prompt)
    else:
        prompt = f"""Answer the question in JSON with "concise" and "detailed" fields as above.
Include a ```mermaid diagram at the end if it helps.
Question: {data['question']}"""
        response = model.generate_content(prompt)
        response_text = response.text or ""

    try:
        response_json = json.loads(response_text)
        concise = response_json.get("concise", clean_text(response_text))
        detailed = response_json.get("detailed", response_text)
    except json.JSONDecodeError:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                response_json = json.loads(response_text[start:end])
                concise = response_json.get("concise", clean_text(response_text))
                detailed = response_json.get("detailed", response_text[start:end])
            except json.JSONDecodeError:
                logger.error("Failed to parse extracted JSON")
                concise = clean_text(response_text)
                detailed = "Error: Unable to generate detailed explanation."
        else:
            logger.error("No JSON object found in response")
            concise = clean_text(response_text)
            detailed = "Error: Unable to generate detailed explanation."

    return {
        "question": data["question"],
        "text": concise,
        "detailed": detailed,
        "responseType": "answer",
        "mode": "learn",
    }

async def handle_quiz_mode(data):
    topic = data["question"]
    prompt = (
        f"Generate a 5-question MCQ quiz on: {topic}. Each question has 4 options A-D, indicate the correct letter and a brief explanation.\n"
        f"Format for each question:\n"
        f"Q: [question]\nA: [A]\nB: [B]\nC: [C]\nD: [D]\nCorrect: [A|B|C|D]\nExplanation: [text]"
    )
    response = model.generate_content(prompt)
    quiz_text = clean_text(response.text or "")
    quiz_markdown = f"### Quiz on {topic}\n\n"
    for i, block in enumerate(quiz_text.split("\n\n"), 1):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if (
            len(lines) >= 7
            and lines[0].startswith("Q:")
            and lines[5].startswith("Correct:")
            and lines[6].startswith("Explanation:")
        ):
            question = lines[0][2:].strip()
            options = [line[2:].strip() for line in lines[1:5]]
            correct = lines[5][8:].strip()
            explanation = lines[6][12:].strip()
            quiz_markdown += f"**Question {i}:** {question}\n\n"
            for opt in options:
                quiz_markdown += f"- {opt}\n"
            quiz_markdown += f"\n**Correct Answer:** {correct}\n\n**Explanation:** {explanation}\n\n"
    return {
        "question": data["question"],
        "text": f"Here is your quiz on {topic}",
        "detailed": quiz_markdown,
        "responseType": "quiz",
        "mode": "quiz",
    }

# --------- HTTP handlers ---------

async def register_handler(request):
    data = await request.json()
    name = data.get("name")
    email = data.get("email")
    studentRegNumber = data.get("studentRegNumber")
    dob = data.get("dob")
    password = data.get("password")

    if not name or not email or not password or not studentRegNumber or not dob:
        return web.Response(status=400, text="Missing fields")

    db = request.app["db"]
    existing_user = await db.users2.find_one({"email": email})
    if existing_user:
        return web.Response(status=409, text="User already exists")

    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    result = await db.users2.insert_one(
        {
            "name": name,
            "email": email,
            "studentRegNumber": studentRegNumber,
            "dob": dob,
            "password": hashed_password,
            "createdAt": datetime.now(timezone.utc),
        }
    )
    return web.json_response({"message": "User created", "user_id": str(result.inserted_id)})

async def login_handler(request):
    data = await request.json()
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return web.Response(status=400, text="Missing email or password")
    db = request.app["db"]
    user = await db.users2.find_one({"email": email})
    if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        payload = {"user_id": str(user["_id"]), "exp": datetime.now(timezone.utc) + timedelta(hours=6)}
        token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
        return web.json_response({"token": token})
    else:
        return web.Response(status=401, text="Invalid credentials")

async def websocket_handler(request):
    token = request.query.get("token")
    if token:
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            user_id = payload["user_id"]
        except jwt.InvalidTokenError:
            return web.Response(status=401, text="Invalid token")
    else:
        return web.Response(status=401, text="Missing token")

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    ws.user_id = user_id

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    logger.info(f"Received message: {data}")
                    mode = data.get("mode")
                    if mode == "learn":
                        response_data = await handle_learn_mode(data)
                    elif mode == "teach":
                        response_data = await handle_teach_mode(data)
                    elif mode == "quiz":
                        response_data = await handle_quiz_mode(data)
                    elif mode == "rag":
                        response_data = await handle_rag_mode(data)
                    else:
                        response_data = {"error": "Invalid mode"}

                    if data.get("type") == "voice_query" and not response_data.get("error"):
                        audio_base64 = await text_to_speech(response_data["text"])
                        if audio_base64:
                            response_data["audio"] = audio_base64
                            logger.info("Audio generated for voice response")
                        else:
                            response_data["error"] = "Failed to generate audio"
                            logger.error("Audio generation failed")

                    await ws.send_json(response_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    await ws.send_json({"error": "Invalid JSON"})
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await ws.send_json({"error": str(e)})
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket handler: {e}")
    finally:
        logger.info("Closing WebSocket connection")
        await ws.close()
    return ws

async def health_check(request):
    return web.Response(text="OK", status=200)

async def root_handler(request):
    return web.Response(text="OK", status=200)

async def start_server():
    asyncio.create_task(initialize_pdf_rag())
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client["mydatabase"]
    app = web.Application()
    app["db"] = db

    cors = aiohttp_cors.setup(app)
    route_configs = [
        {"path": "/health", "handler": health_check, "method": "GET", "allow_methods": ["GET"]},
        {"path": "/", "handler": root_handler, "method": "GET", "allow_methods": ["GET"]},
        {"path": "/register", "handler": register_handler, "method": "POST", "allow_methods": ["POST"]},
        {"path": "/login", "handler": login_handler, "method": "POST", "allow_methods": ["POST"]},
        {"path": "/ws", "handler": websocket_handler, "method": "GET", "allow_methods": ["GET"]},
    ]
    for config in route_configs:
        if config["method"] == "GET":
            route = app.router.add_get(config["path"], config["handler"])
        elif config["method"] == "POST":
            route = app.router.add_post(config["path"], config["handler"])
        cors.add(
            route,
            {
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods=config["allow_methods"],
                )
            },
        )

    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv("PORT", 8765))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info(f"Server started on http://0.0.0.0:{port} and ws://0.0.0.0:{port}/ws")
    return site

async def main():
    await start_server()
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
