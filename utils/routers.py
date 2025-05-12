# from langchain.chat_models import ChatGroq
# from langchain.prompts import ChatPromptTemplate

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


from configurations.config import ModelConfig, SchemaKeys  # assuming you store your config here

from utils.helpers import read_file,pdf_to_markdown



import numpy as np
# import faiss
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from pdf2markdown4llm import PDF2Markdown4LLM


def llm_call_fn(prompt: str) -> str:
    """
    Calls Groq LLM with the provided prompt using ChatGroq (Langchain wrapper).
    
    Parameters:
    - prompt (str): The full prompt string for summarization.
    
    Returns:
    - LLM response content (str)
    """
    groq_api_key = ModelConfig.groq_api_key
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    # Initialize LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=ModelConfig.llm_name)

    # Create prompt template dynamically
    prompt_template = ChatPromptTemplate.from_template("{input}")
    formatted_prompt = prompt_template.format(input=prompt)

    # Invoke LLM
    response = llm.invoke(formatted_prompt)

    return response.content

def summarize_resume_jd_interview(resume_path, jd_path, grouped_chunks: list) -> dict:
    """
    Generates highly concise, structured summaries of a resume, job description, and interview conversation.
    Removes all unnecessary LLM preamble or trailing notes. Returns clean, to-the-point JSON-style outputs.
    """

    resume_text = read_file(resume_path)
    jd_text = read_file(jd_path)
    flat_chunks = [" ".join(group) for group in grouped_chunks]
    interview_combined_text = "\n\n".join(flat_chunks)

    resume_prompt = f"""
Extract key structured information from the following resume.

Strict instructions:
- Output only a clean JSON-style object.
- No introduction or closing statements.
- Omit empty or irrelevant fields.
- Use compact key-value pairs with short phrases, not sentences.
- Include keys like: education, experience, skills, projects, certifications, languages, etc.

Resume:
{resume_text}
    """

    jd_prompt = f"""
Extract structured requirement fields from the following job description.

Strict instructions:
- Output only a clean JSON-style object.
- No preamble or wrap-up text.
- Use keys like: required_education, required_experience, required_skills, preferred_qualifications, tools, certifications, etc.
- Only include explicitly mentioned or clearly implied fields.
- Omit empty, null, or generic fields.

Job Description:
{jd_text}
    """

    interview_prompt = f"""
Summarize the candidate interview strictly by content only.

Strict instructions:
- Output only structured bullet points or grouped sections.
- No introductory or concluding text.
- Organize under keys like: technical_skills, communication, reasoning, attitude, etc.
- Be brief but meaningful. No filler language.

Interview Transcript:
{interview_combined_text}
    """

    try:
        parsed_resume = llm_call_fn(resume_prompt)
        parsed_jd = llm_call_fn(jd_prompt)
        interview_summary = llm_call_fn(interview_prompt)
    except Exception as e:
        raise RuntimeError(f"LLM summarization failed: {str(e)}")

    return {
        "parsed_resume": parsed_resume,
        "parsed_jd": parsed_jd,
        "interview_summary": interview_summary
    }





# # Step 1: Load chunks for Speaker 01
# def load_speaker_chunks(json_path, speaker="SPEAKER_01"):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     return [entry['text'] for entry in data if entry['speaker'] == speaker]

# Step 2: Embed chunks
def embed_chunks(chunks):
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    embeddings = embedder.embed_documents(chunks)
    print("##########in embed chunks function and type of chunks being recieved is : ", type(chunks))

    return np.array(embeddings), embedder

# Step 3: Group similar chunks (text+embedding avg)
def group_chunks(chunks, embeddings, threshold=0.2):
    """
    Groups semantically similar chunks and returns both grouped texts and their average embeddings.

    Args:
        chunks (list of str): List of text chunks.
        embeddings (ndarray): Corresponding embeddings for chunks.
        threshold (float): Cosine similarity threshold for grouping.

    Returns:
        grouped_texts (list of list of str): Each sublist contains similar text chunks.
        grouped_embeddings (list of ndarray): Mean embedding for each group.
    """
    n = len(chunks)
    visited = [False] * n
    grouped_texts = []
    grouped_embeddings = []

    for i in range(n):
        if visited[i]:
            continue
        group_text = [chunks[i]]
        group_embeds = [embeddings[i]]
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j]:
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if sim > threshold:
                    group_text.append(chunks[j])
                    group_embeds.append(embeddings[j])
                    visited[j] = True
        grouped_texts.append(group_text)
        grouped_embeddings.append(np.mean(group_embeds, axis=0))
    
    return grouped_texts, grouped_embeddings


# # Step 4: Save grouped embeddings in FAISS (no documents)
# def build_faiss_index_only_embeddings(grouped_embeddings, index_path="faiss_index.index"):
#     dim = len(grouped_embeddings[0])
#     index = faiss.IndexFlatL2(dim)
#     index.add(np.array(grouped_embeddings).astype('float32'))
#     faiss.write_index(index, index_path)
#     print(f"✅ FAISS index saved at {index_path}")
#     return index

def compute_cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def remove_redundant_chunks(chunks, embeddings, resume_vector, similarity_threshold=0.85):
    """
    Remove interview chunks that are too similar to the resume.
    """
    unique_chunks = []
    unique_embeddings = []

    for chunk, emb in zip(chunks, embeddings):
        sim = compute_cosine_similarity(emb, resume_vector)
        if sim < similarity_threshold:  # If similarity is below threshold, consider it unique
            unique_chunks.append(chunk)
            unique_embeddings.append(emb)

    return unique_chunks, unique_embeddings


# Main processing function
def process_interview_json(chunks):
    embeddings,embedder = embed_chunks(chunks)
    print("chunks : ", chunks)
    print("embeddings : ", embeddings)
    grouped_text, grouped_embs = group_chunks(chunks, embeddings)
    print("#########after embed chunks function : ", )


    # print(resume_single_vector)
    # index = build_faiss_index_only_embeddings(grouped_embs)
    # print(f"✅ FAISS DB created with {len(grouped_embs)} grouped embeddings.")
    return grouped_text
