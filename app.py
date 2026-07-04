import gradio as gr
import os
from crewai import Agent, LLM, Task, Crew, Process
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from crewai.tools import tool

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("Mistral_Api_key")

try:
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
except (ImportError, AttributeError):
    pass

groq_llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
)
mistral_llm = LLM(
    model="mistral/mistral-large-latest",
    api_key=MISTRAL_API_KEY,
)
mistal_llm_2 = LLM(
    model="mistral/mistral-large-latest",
    api_key=os.getenv("mistral_api_key_2"),
)


vector_store = None


@tool
def retrive_content(query: str, k: int = 2) -> str:
    """By using the query string search the pdf and gives matching documents"""
    global vector_store
    if vector_store is None:
        return "Vector store not initialized. Please ensure the PDF has been successfully uploaded and processed."

    retrieved_docs = vector_store.similarity_search(query, k=k)
    docs_content = ""
    for doc in retrieved_docs:
        docs_content += f"Source: {doc.metadata}\n"
        docs_content += f"Content: {doc.page_content}\n\n"
    return docs_content


# --- Agent Definitions ---
methodology_reviewer = Agent(
    role="Methodology Reviewer",
    goal="Dr. Marcus Reid — Experimental Design Specialist",
    backstory=
    "You are Dr. Marcus Reid, MIT Professor of Computational "
    "Statistics. Spent early career at FDA setting clinical "
    "trial standards — made you permanently intolerant of "
    "weak experimental design. Reviewed 3,000+ papers for "
    "NeurIPS, ICML, ICLR, JMLR. Once blocked a 98-citation "
    "preprint because its train-test split was leaking data. "
    "You care about one thing: did they do the experiment "
    "correctly? You always ask: 'Could I reproduce this "
    "from what is written here?' If no — that is your "
    "first comment. You hate: missing std deviations, "
    "undertrained baselines, and cherry-picked ablations.",
    llm=groq_llm,
    tools =[retrive_content],
    verbose=True
)

novelty_reviewer = Agent(
    role="Novelty Reviewer",
    goal="Dr. Aiko Tanaka — Research Originality & Contribution Analyst",
    backstory=
    "You are Dr. Aiko Tanaka, independent AI researcher. "
    "Previously at DeepMind London and Google Brain Tokyo. "
    "140+ papers, 60,000+ citations. Colleagues say you "
    "have read every ML paper ever published. You have "
    "caught 12 cases of parallel work and 3 plagiarism "
    "cases in your reviewing career. Fair but ruthless — "
    "you do not penalize honest incremental work but have "
    "zero tolerance for minor tweaks dressed as breakthroughs. "
    "You hate: buried related work, benchmarks rigged to "
    "win, and 'to the best of our knowledge' used dishonestly. "
    "You always ask: 'If this paper did not exist, what "
    "would research actually be missing?'",
    llm=mistral_llm,
    tools =[retrive_content],
    verbose=True
)

clarity_reviewer = Agent(
    role="Clarity Reviewer",
    goal="Dr. Elena Vasquez — Scientific Communication Expert",
    backstory=
    "You are Dr. Elena Vasquez, Stanford Professor of "
    "Scientific Communication and former Senior Editor "
    "at Nature Machine Intelligence. Edited 500+ published "
    "papers, rejected thousands more for incomprehensible "
    "writing. You believe unclear writing is a scientific "
    "problem — a paper nobody understands cannot be "
    "reproduced or built upon. You read every paper as "
    "a smart PhD student in a related field who should "
    "understand every claim without reading other papers. "
    "You hate: unlabeled figure axes, undefined notation, "
    "and passive voice hiding uncertainty. You always ask: "
    "'Can a smart outsider understand this completely?'",
    llm=mistal_llm_2,
    tools =[retrive_content],
    verbose=True
)

limitations_reviewer = Agent(
    role="Limitations Reviewer",
    goal="Dr. Samuel Okonkwo — Research Ethics & Critical Analysis Specialist",
    backstory=
    "You are Dr. Samuel Okonkwo, AI safety researcher "
    "and former ACM ethics board member for 8 years. "
    "Known for publicly challenging a top-lab paper being "
    "used to justify a commercial product — you proved "
    "its results did not generalize outside its benchmark. "
    "Co-authored the widely adopted 'Research Integrity "
    "Guidelines for ML Publications.' You read papers "
    "looking for what is NOT said, not what is. You "
    "hate: limitations sections that only say future work, "
    "cherry-picked examples, and abstract claims that "
    "contradict the actual experiments. You always ask: "
    "'Who gets harmed if this paper's limits are ignored?'",
    llm=mistral_llm,
    tools =[retrive_content],
    verbose=True
)

judge_agent = Agent(
    role="Judge Agent",
    goal="Prof. Diana Chen — Program Chair & Meta-Reviewer",
    backstory=
    "You are Prof. Diana Chen, the most respected Program "
    "Chair in AI research. Chaired NeurIPS twice, ICML "
    "three times, ICLR once. Made the final accept/reject "
    "decision on 40,000+ papers. First woman to win the "
    "ACM Turing Award in ML. You do not do primary "
    "reviewing — you synthesize specialist reviews and "
    "make the final call. Completely immune to prestige "
    "bias — you have rejected Turing Award winners and "
    "accepted first-year PhD students. Novelty is your "
    "top priority: a methodologically perfect paper with "
    "no novelty is still a reject. You always ask: "
    "'In 5 years, will anyone cite this for the right reason?'",
    llm=mistal_llm_2,
    tools =[retrive_content],
    verbose=True
)

# --- Task Definitions ---
methodology_task = Task(
    description=(
        "Use retrive_content tool to find required PDF docs (max 1 best queries total).\n"
        "As Dr. Marcus Reid, score 1-10 on: experimental_design, dataset_quality, "
        "statistical_rigor, reproducibility, fairness_of_comparison. Check reproducibility, "
        "fair baselines, std deviations, honest ablations, leak-free splits. "
        "Cite section numbers for strengths/weaknesses. Return strict JSON only."
    ),
    expected_output=(
        "JSON: {reviewer, dimension_scores, overall_methodology_score, strengths:[], "
        "weaknesses:[], critical_questions:[], improvement_suggestions:[], verdict}"
    ),
    agent=methodology_reviewer
)

novelty_task = Task(
    description=(
        "Use retrive_content tool to find required PDF docs (max 1 best queries total).\n"
        "As Dr. Aiko Tanaka, score 1-10 on: originality, technical_contribution, "
        "literature_positioning, significance_of_results, potential_impact. Separate claim vs "
        "actual novelty. Flag missing citations, incremental work, or biased benchmarks. "
        "If novelty score < 4, verdict must be REJECT. Return strict JSON only."
    ),
    expected_output=(
        "JSON: {reviewer, dimension_scores, overall_novelty_score, claimed_contributions:[], "
        "actual_contributions:[], novelty_concerns:[], missing_citations:[], improvement_suggestions:[], verdict}"
    ),
    agent=novelty_reviewer
)

clarity_task = Task(
    description=(
        "Use retrive_content tool to find required PDF docs (max 1 best queries total).\n"
        "You are Dr. Elena Vasquez. Score 1-10 across abstract_quality, "
        "structure_organization, technical_writing, figures_and_tables, "
        "language_grammar. Flag undefined terms, unlabeled figures, "
        "broken flow, and grammar errors with exact quotes. "
        "STRICT OUTPUT RULES: "
        "max 2 well_written_sections as plain strings, "
        "max 2 poorly_written_sections as plain strings, "
        "max 2 line_edits with original+issue+fix only, "
        "max 2 figure_issues as plain strings, "
        "No nested objects. No rationale fields. No examples arrays. "
        "Return compact JSON only — no markdown fences, no extra text."
    ),
    expected_output=(
        "JSON: {reviewer, dimension_scores, overall_clarity_score, well_written_sections:[], "
        "poorly_written_sections:[], specific_line_edits:[{original,issue,fix}], "
        "figure_issues[], improvement_suggestions:[], verdict}"
    ),
    agent=clarity_reviewer
)

limitations_task = Task(
    description=(
        "Use retrieve_content tool to find required PDF docs (max 1 query). "
        "You are Dr. Samuel Okonkwo. Score 1-10 across acknowledged_limitations, "
        "scope_of_claims, ethical_considerations, future_work_quality, negative_results_honesty. "
        "Identify missed limitations, overclaims, hidden failure cases, ethical risks. "
        "Quote exact overclaims. "
        "max 2 missing_limitations as plain strings. "
        "max 2 overclaimed_results as {claim, reality} only. "
        "max 2 ethical_flags as plain strings. "
        "max 2 improvement_suggestions as plain strings. "
        "No nested objects. No rationale fields. No harm fields. No location fields. "
        "Return compact JSON only — no markdown fences, no extra text."
    ),
    expected_output=(
        "JSON: {reviewer, dimension_scores{5 scores}, overall_limitations_score,"
        "missing_limitations:[], overclaimed_results:[{claim, reality}], ethical_flags:[], "
        "improvement_suggestions:[], verdict}"
    ),
    agent=limitations_reviewer
)

judge_task = Task(
    description=(
        "Synthesize ONLY the 4 reviews passed in context. "
        "Do NOT use any tools. Read the 4 reviewer scores directly. "
        "novelty*0.35 + methodology*0.30 + clarity*0.20 + limitations*0.15."
        "Hard reject only if novelty<3 or methodology<2. "
        "A published landmark paper should never score below WEAK_ACCEPT. "
        "Write a tough, constructive 3-sentence summary directly TO the authors. Suggest a venue. "
        "Output MUST be valid JSON wrapped inside clean Markdown blockquotes or headers for presentation."
    ),
    expected_output=(
        "Markdown containing a JSON block with: {meta_reviewer, paper_title, individual_scores:{}, "
        "weighted_final_score, hard_rules_triggered:bool, final_verdict, confidence, top_strengths:[], "
        "top_weaknesses:[], mandatory_revisions:[], summary_for_authors, suggested_venue}"
    ),
    agent=judge_agent,
    context=[methodology_task, novelty_task, clarity_task, limitations_task]
)


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
    process=Process.sequential,
    verbose=True,
)


# --- Gradio Interface Logic ---
def format_review_to_markdown(result):
    """Converts raw outputs or JSON objects into beautifully structured markdown."""
    if not result:
        return "*No review report was generated.*"
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if "error" in result:
            return f"### ⚠️ Review Interrupted\n\n**Error Details:** {result['error']}"
        md = []
        for key, value in result.items():
            heading = key.replace("_", " ").title()
            md.append(f"## {heading}")
            if isinstance(value, list):
                for item in value:
                    md.append(f"* {item}")
            elif isinstance(value, dict):
                for subkey, subval in value.items():
                    subheading = subkey.replace("_", " ").title()
                    md.append(f"### {subheading}\n\n{subval}\n")
            else:
                md.append(f"{value}")
            md.append("\n")
        return "\n".join(md)
    return str(result)


async def run_crew_review(pdf_file):
    if pdf_file is None:
        return "⚠️ *Please upload a PDF file first.*"
    if isinstance(pdf_file, str):
        pdf_path = pdf_file
    elif hasattr(pdf_file, "name"):
        pdf_path = pdf_file.name
    else:
        pdf_path = str(pdf_file)

    global vector_store
    try:
        # Load PDF document
        loader = PyPDFLoader(pdf_path)
        doc = loader.load()

        # Split text content
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        all_splits = text_splitter.split_documents(doc)

        # Initialize Embeddings & Vector Store
        embed = MistralAIEmbeddings(
            model="mistral-embed",
            api_key=MISTRAL_API_KEY,
        )

        vector_store = InMemoryVectorStore(embedding=embed)
        vector_store.add_documents(documents=all_splits)

        # Run CrewAI evaluation sequentially — the agents query the
        # retrive_content tool against the vector_store built above.
        result = await crew.kickoff_async(inputs={"pdf": pdf_path})
        return format_review_to_markdown(str(result))
    except Exception as e:
        return format_review_to_markdown({"error": f"Error during review: {str(e)}"})
    finally:
        # Clear the vector store after each run so a stale paper can't leak
        # into the next user's review on the same worker process.
        vector_store = None


async def handle_review(pdf_file):
    if pdf_file is None:
        return (
            gr.update(value="<div class='report-subtitle-text'>DRAFT STATUS: NO MANUSCRIPT DETECTED</div>"),
            gr.update(visible=True),
            gr.update(visible=False, value="")
        )
    report = await run_crew_review(pdf_file)
    if "⚠️ Review Interrupted" in report or "⚠️" in report:
        return (
            gr.update(value="<div class='report-subtitle-text'>DRAFT STATUS: ERROR ENCOUNTERED</div>"),
            gr.update(visible=False),
            gr.update(visible=True, value=report)
        )
    return (
        gr.update(value="<div class='report-subtitle-text'>DRAFT STATUS: EVALUATION COMPLETE</div>"),
        gr.update(visible=False),
        gr.update(visible=True, value=report)
    )


# Theme layout styling for both Light and Dark mode
css = """
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400..700;1,400..700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
/* Base Layout Config */
body, .gradio-container {
    background-color: #FDFDFD !important;
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    color: #1E293B !important;
}
.dark body, .dark .gradio-container {
    background-color: #0B0F19 !important;
    color: #E2E8F0 !important;
}
/* Page Header Title */
.header-banner {
    border-bottom: 1px solid #EAE6E8 !important;
    padding-bottom: 1.25rem !important;
    margin-bottom: 3.5rem !important;
    text-align: left !important;
}
.dark .header-banner {
    border-bottom: 1px solid #1E293B !important;
}
.header-title {
    font-family: 'Lora', Georgia, serif !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: #6B2D3A !important;
    margin: 0 !important;
}
.dark .header-title {
    color: #F8FAFC !important;
}
/* Submission Card (Left) */
.submission-card {
    background-color: #FFFFFF !important;
    border: 1px solid #EAE6E8 !important;
    border-left: 5px solid #6B2D3A !important;
    border-radius: 4px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02) !important;
    padding: 1.5rem !important;
}
.dark .submission-card {
    background-color: #111827 !important;
    border: 1px solid #1F2937 !important;
    border-left-color: #EF4444 !important;
}
/* Card Header Layout */
.card-header-wrapper {
    display: flex !important;
    align-items: center !important;
    gap: 0.75rem !important;
    margin-bottom: 1.25rem !important;
}
.portal-icon {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 28px !important;
    height: 28px !important;
    background-color: #F9F3F4 !important;
    border-radius: 50% !important;
}
.dark .portal-icon {
    background-color: #1F2937 !important;
}
.portal-svg {
    width: 14px !important;
    height: 14px !important;
}
.portal-title {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #6B2D3A !important;
    letter-spacing: -0.01em !important;
}
.dark .portal-title {
    color: #F8FAFC !important;
}
/* Customized File Upload Area */
.submission-card .gradio-file {
    border: 1px dashed #D1C4C6 !important;
    background-color: #FAF9FA !important;
    border-radius: 6px !important;
    padding: 2.5rem 1rem !important;
    text-align: center !important;
    transition: all 0.2s ease !important;
}
.dark .submission-card .gradio-file {
    border-color: #374151 !important;
    background-color: #1F2937 !important;
}
.submission-card .gradio-file:hover {
    border-color: #6B2D3A !important;
}
.dark .submission-card .gradio-file:hover {
    border-color: #EF4444 !important;
}
/* Submit Action Button */
.editorial-submit-btn {
    background-color: #6B2D3A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.85rem 1rem !important;
    text-align: center !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: background-color 0.2s ease !important;
}
.editorial-submit-btn:hover {
    background-color: #55232D !important;
}
.dark .editorial-submit-btn {
    background-color: #DC2626 !important;
}
.dark .editorial-submit-btn:hover {
    background-color: #B91C1C !important;
}
/* Report Card (Right) */
.report-card {
    background-color: #FFFFFF !important;
    border: 1px solid #EAE6E8 !important;
    border-radius: 4px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02) !important;
    padding: 2rem !important;
}
.dark .report-card {
    background-color: #111827 !important;
    border: 1px solid #1F2937 !important;
}
.report-header-wrapper {
    display: flex !important;
    justify-content: space-between !important;
    align-items: flex-start !important;
}
.report-title-text {
    font-family: 'Lora', Georgia, serif !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #6B2D3A !important;
    margin: 0 !important;
}
.dark .report-title-text {
    color: #F8FAFC !important;
}
.report-subtitle-text {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: #60738C !important;
    letter-spacing: 0.08em !important;
    margin-top: 0.4rem !important;
}
.dark .report-subtitle-text {
    color: #94A3B8 !important;
}
.header-action-buttons {
    display: flex !important;
    gap: 0.5rem !important;
}
.outline-btn {
    width: 32px !important;
    height: 32px !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    background-color: transparent !important;
}
.dark .outline-btn {
    border-color: #374151 !important;
}
.report-divider {
    border: none !important;
    border-top: 1px solid #EAE6E8 !important;
    margin: 1.25rem 0 2rem 0 !important;
}
.dark .report-divider {
    border-top: 1px solid #1F2937 !important;
}
/* Awaiting Manuscript Inner Panel */
.awaiting-container {
    border: 1px dashed #E2D9DC !important;
    border-radius: 6px !important;
    background-color: #FCFBFB !important;
    padding: 5rem 2rem !important;
    text-align: center !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
}
.dark .awaiting-container {
    border-color: #374151 !important;
    background-color: #1F2937 !important;
}
.awaiting-icon-wrapper {
    width: 64px !important;
    height: 64px !important;
    background-color: #F5EFF1 !important;
    border-radius: 12px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin-bottom: 1.5rem !important;
}
.dark .awaiting-icon-wrapper {
    background-color: #111827 !important;
}
.awaiting-svg {
    width: 28px !important;
    height: 28px !important;
}
.awaiting-title {
    font-family: 'Lora', Georgia, serif !important;
    font-size: 1.35rem !important;
    font-style: italic !important;
    font-weight: 500 !important;
    color: #60738C !important;
    margin-bottom: 0.75rem !important;
}
.dark .awaiting-title {
    color: #94A3B8 !important;
}
.awaiting-text {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88rem !important;
    color: #64748B !important;
    max-width: 440px !important;
    line-height: 1.6 !important;
    margin: 0 auto !important;
}
.dark .awaiting-text {
    color: #94A3B8 !important;
}
/* Markdown Evaluated Output styling */
.review-output {
    font-family: 'Lora', Georgia, serif !important;
    font-size: 1.05rem !important;
    line-height: 1.8 !important;
}
.review-output, .review-output * {
    color: #334155 !important;
}
.dark .review-output, .dark .review-output * {
    color: #CBD5E1 !important;
}
.review-output h2, .review-output h2 *,
.review-output h3, .review-output h3 * {
    color: #0F172A !important;
}
.dark .review-output h2, .dark .review-output h2 *,
.dark .review-output h3, .dark .review-output h3 * {
    color: #F8FAFC !important;
}
.review-output h2 {
    font-weight: 700 !important;
    margin-top: 2rem !important;
    margin-bottom: 0.75rem !important;
    border-bottom: 1px solid #E2E8F0 !important;
    padding-bottom: 0.25rem !important;
}
.dark .review-output h2 {
    border-bottom: 1px solid #1F2937 !important;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="slate")) as interface:
    # Page Header Title
    with gr.Column(elem_classes="header-banner"):
        gr.HTML("<h1 class='header-title'>The Research Paper Review</h1>")
        gr.Markdown(
        """
         This Project is a multi-agent research paper reviewer built with CrewAI and Retrieval-Augmented Generation (RAG).
        """)
        gr.Markdown(
        """
         Upload a scientific paper and five specialized AI agents — each 
         modeled after a distinct academic reviewer persona — independently 
        analyze different dimensions of your paper. A final judge agent 
        synthesizes all reviews into one structured report with a 
        weighted accept/reject verdict.
        """)
        gr.Markdown(
        """
         Upload papers under 8 pages  — larger papers exceed the Groq and Mistral token limits and will cause a RateLimitError.
        """)
        gr.Markdown("Research Paper link (https://arxiv.org/pdf/2606.22662)." )
        gr.Markdown("Source code link (https://huggingface.co/spaces/Avenger09/The-research-paper-review/blob/main/app.py)." )
        
    with gr.Row():
        # Submission Panel Column (Left)
        with gr.Column(scale=2):
            with gr.Column(elem_classes="submission-card"):
                gr.HTML(
                    """
                    <div class="card-header-wrapper">
                        <span class="portal-icon">
                            <svg class="portal-svg" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <circle cx="12" cy="12" r="10" stroke="#6B2D3A" stroke-width="2"/>
                                <path d="M12 8V16M12 8L8 12M12 8L16 12" stroke="#6B2D3A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </span>
                        <span class="portal-title">Submission Portal</span>
                    </div>
                    <hr class="portal-divider">
                    """
                )
                pdf_input = gr.File(
                    label="Upload Manuscript",
                    type="filepath",
                    file_types=[".pdf"]
                )
                submit_button = gr.Button(
                    "INITIATE EDITORIAL REVIEW   ➔",
                    elem_classes="editorial-submit-btn"
                )
        # Report Panel Column (Right)
        with gr.Column(scale=3):
            with gr.Column(elem_classes="report-card"):
                with gr.Row(elem_classes="report-header-wrapper"):
                    with gr.Column(scale=1):
                        gr.HTML("<h2 class='report-title-text'>Evaluation Report</h2>")
                        status_display = gr.HTML(
                            value="<div class='report-subtitle-text'>DRAFT STATUS: NO MANUSCRIPT DETECTED</div>"
                        )
                    gr.HTML(
                        """
                        <div class="header-action-buttons">
                            <div class="outline-btn"></div>
                            <div class="outline-btn"></div>
                        </div>
                        """
                    )
                gr.HTML("<hr class='report-divider'>")
                with gr.Column(visible=True) as awaiting_panel:
                    gr.HTML(
                        """
                        <div class="awaiting-container">
                            <div class="awaiting-icon-wrapper">
                                <svg class="awaiting-svg" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M14 2H6C4.9 2 4 2.9 4 4V20C4 21.1 4.9 22 6 22H18C19.1 22 20 21.1 20 20V8L14 2Z" stroke="#E2D9DC" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                    <path d="M14 2V8H20" stroke="#E2D9DC" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                    <path d="M12 18C14.2091 18 16 16.2091 16 14C16 11.7909 14.2091 10 12 10C9.79086 10 8 11.7909 8 14C8 16.2091 9.79086 18 12 18Z" stroke="#E2D9DC" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                    <path d="M12 12V14L13.5 15.5" stroke="#E2D9DC" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                </svg>
                            </div>
                            <h3 class="awaiting-title">Awaiting Manuscript Input</h3>
                            <p class="awaiting-text">
                                Submit a manuscript in the submission portal to generate the evaluation report. The multi-agent consensus system will perform a deep semantic analysis and peer-review simulation.
                            </p>
                        </div>
                        """
                    )
                review_output = gr.Markdown(
                    value="",
                    visible=False,
                    elem_classes="review-output"
                )
    submit_button.click(
        fn=handle_review,
        inputs=pdf_input,
        outputs=[status_display, awaiting_panel, review_output]
    )

if __name__ == "__main__":
    interface.launch()