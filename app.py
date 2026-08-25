import os
import sys
import asyncio
import tempfile
from dotenv import load_dotenv
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from crewai import Agent, LLM, Task, Crew
from crewai.tools import tool
from langsmith.integrations.otel import OtelSpanProcessor
from opentelemetry import trace
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(OtelSpanProcessor())
trace.set_tracer_provider(tracer_provider)
CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)

try:
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
except (ImportError, AttributeError):
    pass

# API Keys
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
MISTRAL_API_KEY = (os.getenv("MISTRAL_API_KEY") or os.getenv("Mistral_Api_key") or "").strip()
MISTRAL_API_KEY_2 = (os.getenv("mistral_api_key_2") or "").strip()

# LLM Configurations
groq_llm = LLM(
    model="groq/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    verbose=True,
)

mistral_llm = LLM(
    model="mistral/mistral-large-latest",
    api_key=MISTRAL_API_KEY,
    verbose=True,
)

mistral_llm_2 = LLM(
    model="mistral/mistral-large-latest",
    api_key=MISTRAL_API_KEY_2,
    verbose=True,
)

vector_store = None

ALLOWED_TYPES = {
    "application/pdf": "PDF Document",
    "text/csv": "CSV Spreadsheet",
}

@tool
def retrive_content(query: str, k: int = 2) -> str:
    """By using the query string search the pdf and gives matching documents"""
    global vector_store
    retrieved_docs = vector_store.similarity_search(query, k)
    docs_content = ""
    for doc in retrieved_docs:
        docs_content += f"Source: {doc.metadata}\n"
        docs_content += f"Content: {doc.page_content}\n\n"
    return docs_content

methodology_reviewer = Agent(
    role="Methodology Reviewer",
    goal="Experimental Design Specialist",
    backstory=(
        "You are a Computational Statistics expert."
        "Strictly evaluate experimental design, data leakage, std deviations, "
        "baselines, and reproducibility from what is written."
    ),
    llm=groq_llm,
    tools=[retrive_content],
    verbose=True
)

novelty_reviewer = Agent(
    role="Novelty Reviewer",
    goal="Research Originality Analyst",
    backstory=(
        "You are an AI researcher."
        "Focus on originality vs incremental tweaks, buried related work, "
        "and dishonest claims. Determine what research would miss without this paper."
    ),
    llm=mistral_llm,
    tools=[retrive_content],
    verbose=True
)

clarity_reviewer = Agent(
    role="Clarity Reviewer",
    goal="Scientific Communication Expert",
    backstory=(
        "You are a Scientific Communication Expert."
        "Flag unclear writing, undefined terms, unlabeled figure axes, and broken flow."
    ),
    llm=mistral_llm_2,
    tools=[retrive_content],
    verbose=True
)

limitations_reviewer = Agent(
    role="Limitations Reviewer",
    goal="Research Ethics & Critical Analysis Specialist",
    backstory=(
        "You are an AI ethics researcher. "
        "Uncover unstated limitations, overclaims, cherry-picked results, and ethical risks."
    ),
    llm=mistral_llm,
    tools=[retrive_content],
    verbose=True
)

judge_agent = Agent(
    role="Judge Agent",
    goal="Program Chair & Meta-Reviewer",
    backstory=(
        "You are an AI Program Chair. Synthesize specialist reviews. "
        "Prioritize genuine novelty over pure methodology."
    ),
    llm=groq_llm,
    verbose=True
)

methodology_task = Task(
    description=(
        "Use retrieve_content tool (max 1 query). Briefly evaluate: "
        "experimental_design, dataset_quality, statistical_rigor, reproducibility, fairness_of_comparison. "
        "Check fair baselines, std devs, ablations, and leak-free splits. Cite section numbers. "
        "CRITICAL: Keep all text entries short and impact-focused (max 1 sentence per field)."
    ),
    expected_output=(
        "Raw valid JSON matching exact schema: "
        '{"dimension_scores":{"experimental_design":int,"dataset_quality":int,"statistical_rigor":int,'
        '"reproducibility":int,"fairness_of_comparison":int},"overall_methodology_score":int,'
        '"strengths":["short string"],"weaknesses":["short string"],"improvement_suggestions":["short string"],'
        '"verdict":"1-2 sentence summary"}'
    ),
    agent=methodology_reviewer
)

novelty_task = Task(
    description=(
        "Use retrieve_content tool (max 1 query). Briefly evaluate: "
        "originality, technical_contribution, literature_positioning, significance_of_results, potential_impact. "
        "Separate claimed vs actual novelty. Flag missing citations or biased benchmarks. "
        "If overall_novelty_score < 4, set verdict to REJECT. Cite section numbers. "
        "CRITICAL: Keep all text entries short and impact-focused (max 1 sentence per field)."
    ),
    expected_output=(
        "Raw valid JSON matching exact schema: "
        '{"dimension_scores":{"originality":int,"technical_contribution":int,"literature_positioning":int,'
        '"significance_of_results":int,"potential_impact":int},"overall_novelty_score":int,'
        '"claimed_contributions":["short string"],"actual_contributions":["short string"],'
        '"novelty_concerns":["short string"],"missing_citations":["short string"],'
        '"improvement_suggestions":["short string"],"verdict":"REJECT or short 1-sentence decision"}'
    ),
    agent=novelty_reviewer
)

clarity_task = Task(
    description=(
        "Use retrieve_content tool (max 1 query). Briefly evaluate: abstract_quality, "
        "structure_organization, technical_writing, figures_and_tables, language_grammar. "
        "Flag undefined terms, unlabeled figures, and grammar errors. "
        "RULES: max 2 well_written_sections, max 2 poorly_written_sections, max 2 specific_line_edits, "
        "max 2 figure_issues. Keep all text strings under 10 words. Cite section numbers."
    ),
    expected_output=(
        "Raw valid JSON matching exact schema: "
        '{"dimension_scores":{"abstract_quality":int,"structure_organization":int,"technical_writing":int,'
        '"figures_and_tables":int,"language_grammar":int},"overall_clarity_score":int,'
        '"well_written_sections":["short string"],"poorly_written_sections":["short string"],'
        '"specific_line_edits":[{"original":"short string","issue":"short string","fix":"short string"}],'
        '"figure_issues":["short string"],"improvement_suggestions":["short string"],"verdict":"1-sentence summary"}'
    ),
    agent=clarity_reviewer
)

limitations_task = Task(
    description=(
        "Use retrieve_content tool (max 1 query). Briefly evaluate: acknowledged_limitations, "
        "scope_of_claims, ethical_considerations, future_work_quality, negative_results_honesty. "
        "Identify missed limits and overclaims. Cite section numbers. "
        "RULES: max 2 missing_limitations, max 2 overclaimed_results, max 2 ethical_flags, "
        "max 2 improvement_suggestions. Keep all text strings under 10 words."
    ),
    expected_output=(
        "Raw valid JSON matching exact schema: "
        '{"dimension_scores":{"acknowledged_limitations":int,"scope_of_claims":int,'
        '"ethical_considerations":int,"future_work_quality":int,"negative_results_honesty":int},'
        '"overall_limitations_score":int,"missing_limitations":["short string"],'
        '"overclaimed_results":[{"claim":"short string","reality":"short string"}],'
        '"ethical_flags":["short string"],"improvement_suggestions":["short string"],"verdict":"1-sentence summary"}'
    ),
    agent=limitations_reviewer
)

judge_task = Task(
    description=(
        "Synthesize ONLY the 4 reviews passed in context (do NOT use tools). "
        "Calculate score: novelty*0.35 + methodology*0.30 + clarity*0.20 + limitations*0.15. "
        "Hard reject ONLY if novelty<3 or methodology<2. Landmark papers score >= WEAK_ACCEPT. "
        "Write a brief 2-sentence author summary and suggest a venue. "
        "CRITICAL: Return valid JSON object only."
    ),
    expected_output=(
        'A valid JSON object: '
        '{\n'
        '  "paper_title": "string",\n'
        '  "individual_scores": {"novelty": float, "methodology": float, "clarity": float, "limitations": float},\n'
        '  "weighted_final_score": float,\n'
        '  "hard_rules_triggered": bool,\n'
        '  "final_verdict": "ACCEPT",\n'
        '  "confidence": 90,\n'
        '  "top_strengths": ["string"],\n'
        '  "top_weaknesses": ["string"],\n'
        '  "mandatory_revisions": ["string"],\n'
        '  "summary_for_authors": "string",\n'
        '  "suggested_venue": "string"\n'
        '}'
    ),
    agent=judge_agent,
    context=[methodology_task, novelty_task, clarity_task, limitations_task]
)

_active_event_queue = None
_active_event_loop = None

def make_task_completed_cb(code_name):
    def _cb(output):
        if _active_event_queue and _active_event_loop:
            msg = f"✔ Evaluation section finished for {code_name}"
            _active_event_loop.call_soon_threadsafe(_active_event_queue.put_nowait, {
                "event": "log",
                "status": "completed",
                "agent": code_name,
                "message": msg
            })
    return _cb

methodology_task.callback = make_task_completed_cb("METHODOLOGY")
novelty_task.callback = make_task_completed_cb("NOVELTY")
clarity_task.callback = make_task_completed_cb("CLARITY")
limitations_task.callback = make_task_completed_cb("LIMITATIONS")
judge_task.callback = make_task_completed_cb("JUDGE")

crew = Crew(
    agents=[
        methodology_reviewer,
        novelty_reviewer,
        clarity_reviewer,
        limitations_reviewer,
        judge_agent
    ],
    tasks=[
        methodology_task,
        novelty_task,
        clarity_task,
        limitations_task,
        judge_task
    ],
    verbose=True,
)

def load_document_from_upload(upload_file: UploadFile, content_type: str):
    file_extension = ".pdf" if content_type == "application/pdf" else ".csv"

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(upload_file.file.read())
        temp_file_path = temp_file.name

    try:
        if content_type == "application/pdf":
            loader = PyPDFLoader(temp_file_path)
        else:
            loader = CSVLoader(temp_file_path)
        
        documents = loader.load()
        return documents
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/upload-single/")
async def upload_single_file(file: UploadFile = File(...)):
    global vector_store

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Allowed types: {list(ALLOWED_TYPES.keys())}"
        )

    documents = load_document_from_upload(file, file.content_type)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    doc_chunks = text_splitter.split_documents(documents)

    embedding_model = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=MISTRAL_API_KEY,
    )
    
    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents=doc_chunks)

    file_label = ALLOWED_TYPES[file.content_type]
    return {
        "status": "success",
        "message": f"{file_label} uploaded and indexed successfully!",
        "filename": file.filename,
        "chunks_created": len(doc_chunks)
    }

def map_agent_code(name_str):
    s = str(name_str).upper()
    if "METHOD" in s:
        return "METHODOLOGY"
    elif "NOVEL" in s:
        return "NOVELTY"
    elif "CLARIT" in s:
        return "CLARITY"
    elif "LIMIT" in s:
        return "LIMITATIONS"
    elif "JUDGE" in s or "CHAIR" in s:
        return "JUDGE"
    return "AGENT"

async def event_generator():
    global _active_event_queue, _active_event_loop
    try:
        event_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        _active_event_queue = event_queue
        _active_event_loop = loop

        def step_callback(step_output):
            try:
                agent_name = "AGENT"
                if hasattr(step_output, 'agent'):
                    agent_name = getattr(step_output.agent, 'role', 'AGENT')
                msg = f"Analyzing document content... {str(step_output)[:120]}"
                loop.call_soon_threadsafe(event_queue.put_nowait, {
                    "event": "log",
                    "status": "analyzing",
                    "agent": map_agent_code(agent_name),
                    "message": msg
                })
            except Exception:
                pass

        def task_callback(task_output):
            try:
                agent_name = "AGENT"
                if hasattr(task_output, 'agent') and task_output.agent:
                    agent_name = getattr(task_output.agent, 'role', 'AGENT')
                elif hasattr(task_output, 'name') and task_output.name:
                    agent_name = task_output.name

                summary_text = str(getattr(task_output, 'summary', '') or getattr(task_output, 'description', '') or "Evaluation complete")
                msg = f"✔ Evaluation section finished: {summary_text[:100]}"
                loop.call_soon_threadsafe(event_queue.put_nowait, {
                    "event": "log",
                    "status": "completed",
                    "agent": map_agent_code(agent_name),
                    "message": msg
                })
            except Exception:
                pass

        crew.step_callback = step_callback
        crew.task_callback = task_callback

        # 1. Notify client that processing has started
        yield f"data: {json.dumps({'event': 'started', 'message': 'Multi-agent crew initialized...'})}\n\n"
        yield f"data: {json.dumps({'event': 'log', 'agent': 'COORDINATOR', 'message': 'Dispatching manuscript to Methodology, Novelty, Clarity, and Limitations agents...'})}\n\n"

        # 2. Kick off CrewAI asynchronously
        kickoff_task = asyncio.create_task(crew.kickoff_async())

        # Stream real-time agent log events while kickoff is running
        while not kickoff_task.done():
            try:
                item = await asyncio.wait_for(event_queue.get(), timeout=0.4)
                yield f"data: {json.dumps(item)}\n\n"
            except asyncio.TimeoutError:
                pass

        # Drain any remaining queued events
        while not event_queue.empty():
            item = event_queue.get_nowait()
            yield f"data: {json.dumps(item)}\n\n"

        crew_output = await kickoff_task

        raw_text = crew_output.raw.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "", 1).rstrip("`").strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.replace("```", "", 1).rstrip("`").strip()

        try:
            parsed_json = json.loads(raw_text)
            final_payload = {"event": "completed", "review": parsed_json}
        except json.JSONDecodeError:
            final_payload = {"event": "completed", "review_raw": crew_output.raw}

        yield f"data: {json.dumps(final_payload)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_payload = {"event": "error", "detail": f"Review process failed: {str(e)}"}
        yield f"data: {json.dumps(error_payload)}\n\n"

@app.post("/review/")
async def run_review_process():
    """Triggers the multi-agent peer review crew on the uploaded document."""
    global vector_store

    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No document has been uploaded yet. Please upload a file via /upload-single/ first."
        )
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

# Serve Vite frontend static assets if they are built
frontend_dist_path = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist_path):
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="frontend")
