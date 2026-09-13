import pypdf
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def load_and_chunk_pdf(file_path, chunk_size=512,overlap=50):
    "Load a PDF, extracts text, and splits it into chunks "
    
    reader = pypdf.PdfReader(file_path)
    text = ""
    
    # Extract text from each page
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Split the text into chunks
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    
    return chunks

pdf_chuncks=load_and_chunk_pdf("Food.pdf")

print(f"Loading embedding model")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def create_vector_store(chunks,model):
    """Embeds chunks and stores them in a fiass index"""
    print(f"Creating embeddings and vector store.....")
    embeddings = model.encode(chunks, convert_to_tensor=True)
    embeddings_np = embeddings.cpu().numpy()

    dimension = embeddings_np.shape[1]
    index=faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    print("Vector store created successfully")
    return index

vector_store=create_vector_store(pdf_chuncks,embedding_model)

def retrieve_similar_chunks(query , vector_store, chunks,model, top_k=3):
    """Retrieves the most similar chunks to the query from the vector store"""
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()

    distances, indices = vector_store.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

print("Loading generative LLM.....")
llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def generate_response(query, context_chunks, tokenizer, model):
    """Generates a response using the LLM based on the query and context chunks"""
    context = " ".join(context_chunks)
    prompt = f"""Context:{context}Question:{query}Answer:"""
    inputs=llm_tokenizer(prompt,return_tensors="pt",
                     max_lengths=1024,truncation=True)

    outputs=llm_model.generate(**inputs,max_length=200,
                    temperature=0.1,top_p=0.95)

    return llm_tokenizer.decode(outputs[0],skip_special_tokens=True)


def ask_chatbot(query):
    """End-to-end function to chat with the PDF."""
    retrieved_chunks=retrieve_similar_chunks(query,vector_store,pdf_chuncks,embedding_model)
    answer=generate_response(query,retrieved_chunks,llm_tokenizer,llm_model)
    return answer 

user_question="How does the paper propose to identify super wights in a data-free way?"
final_answer=ask_chatbot(user_question)

print("\n--Chatbot Response--")
print(final_answer)
