from typing import List
import os
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from lxml import etree
from PyPDF2 import PdfReader

DATA_DIR = "ICD10"
CHROMA_PERSIST_DIR = "./chroma_db"  # Directory to persist vector stores

# Global vector stores - will be populated by load_local_files()
cm_vectorstore = None
pcs_vectorstore = None

def categorize_file(filename: str) -> tuple:
    """Auto-categorize YOUR files."""
    fn_lower = filename.lower()
    
    # CM Index files
    if 'icd10cm-index' in fn_lower and filename.endswith('.xml'):
        return 'cm_index', 'xml'
    elif 'icd10cm-index' in fn_lower and filename.endswith('.pdf'):
        return 'cm_index_pdf', 'pdf'
    
    # CM Tabular files
    elif 'icd10cm-tabular' in fn_lower and filename.endswith('.xml'):
        return 'cm_tabular', 'xml'
    elif 'icd10cm-tabular' in fn_lower and filename.endswith('.pdf'):
        return 'cm_tabular_pdf', 'pdf'
    
    # CM Guidelines PDF (ICD-10-CM-October-2025-Guidelines.pdf)
    elif 'icd-10-cm' in fn_lower and 'guidelines' in fn_lower and filename.endswith('.pdf'):
        return 'cm_guidelines', 'pdf'
    
    # PCS XML files (icd10pcs_definitions_2026.xml, icd10pcs_index_2026.xml, icd10pcs_tables_2026.xml)
    elif 'icd10pcs' in fn_lower and filename.endswith('.xml'):
        return 'pcs', 'xml'
    
    # PCS PDF files (pcs_2026.pdf, pcs_guidelines_2024_APRIL 1.pdf)
    elif 'pcs' in fn_lower and filename.endswith('.pdf'):
        return 'pcs_guidelines', 'pdf'
    
    # Skip XSD schema files and other non-data files
    return None, None

def parse_xml_file(file_path: str, doc_type: str) -> List[Document]:
    """Extract all text from any XML file and chunk it"""
    print(f"  📄 Parsing XML: {os.path.basename(file_path)}")
    tree = etree.parse(file_path)
    
    # Get all text content from the XML
    all_text = ' '.join(tree.xpath('//text()')).strip()
    
    # Split into chunks of ~1000 characters
    chunk_size = 1000
    docs = []
    for i in range(0, len(all_text), chunk_size):
        chunk = all_text[i:i+chunk_size]
        if len(chunk.strip()) > 50:
            docs.append(Document(
                page_content=chunk.strip(),
                metadata={
                    'type': doc_type,
                    'source': os.path.basename(file_path),
                    'chunk': i // chunk_size
                }
            ))
    
    print(f"    ✅ Extracted {len(docs)} chunks")
    return docs

def parse_pdf_file(file_path: str, doc_type: str) -> List[Document]:
    """Parse any PDF file by extracting text from each page"""
    print(f"  📄 Parsing PDF: {os.path.basename(file_path)}")
    reader = PdfReader(file_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and len(text.strip()) > 100:
            # Split into chunks of approximately 1000 characters
            chunks = [text[j:j+1000] for j in range(0, len(text), 1000)]
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk.strip()) > 100:
                    docs.append(Document(
                        page_content=chunk.strip(),
                        metadata={
                            'page': i+1,
                            'chunk': chunk_idx,
                            'type': doc_type,
                            'source': os.path.basename(file_path)
                        }
                    ))
    print(f"    ✅ Extracted {len(docs)} chunks from {len(reader.pages)} pages")
    return docs

def parse_pcs_xml(file_path: str) -> List[Document]:
    """Extract all text from PCS XML and chunk it"""
    print(f"  📄 Parsing PCS XML: {os.path.basename(file_path)}")
    tree = etree.parse(file_path)
    
    # Get all text content from the XML
    all_text = ' '.join(tree.xpath('//text()')).strip()
    
    # Split into chunks of ~1000 characters
    chunk_size = 1000
    docs = []
    for i in range(0, len(all_text), chunk_size):
        chunk = all_text[i:i+chunk_size]
        if len(chunk.strip()) > 50:
            docs.append(Document(
                page_content=chunk.strip(),
                metadata={
                    'type': 'pcs',
                    'source': os.path.basename(file_path),
                    'chunk': i // chunk_size
                }
            ))
    
    print(f"    ✅ Extracted {len(docs)} chunks")
    return docs

def load_local_files():
    """Load ICD-10 files from DATA_DIR or use existing persistent vector stores"""
    print("\n" + "="*60)
    print("🚀 Starting ICD-10 Data Loading Process")
    print("="*60)
    global cm_vectorstore, pcs_vectorstore
    
    print("\n🤖 Initializing MiniLM embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  
    print("✅ Embeddings model loaded")
    
    # Check if persistent vector stores already exist
    cm_persist_path = os.path.join(CHROMA_PERSIST_DIR, "icd10cm")
    pcs_persist_path = os.path.join(CHROMA_PERSIST_DIR, "icd10pcs")
    
    if os.path.exists(cm_persist_path) and os.path.exists(pcs_persist_path):
        print("\n📦 Found existing vector stores on disk - loading from persistence...")
        try:
            cm_vectorstore = Chroma(
                collection_name="icd10cm",
                embedding_function=embeddings,
                persist_directory=cm_persist_path
            )
            pcs_vectorstore = Chroma(
                collection_name="icd10pcs",
                embedding_function=embeddings,
                persist_directory=pcs_persist_path
            )
            cm_count = cm_vectorstore._collection.count()
            pcs_count = pcs_vectorstore._collection.count()
            print(f"  ✅ Loaded CM vector store: {cm_count} documents")
            print(f"  ✅ Loaded PCS vector store: {pcs_count} documents")
            print("\n" + "="*60)
            print("🎉 Vector Stores Loaded from Disk!")
            print("="*60 + "\n")
            return
        except Exception as e:
            print(f"  ⚠️  Error loading persisted stores: {e}")
            print("  🔄 Will rebuild from source files...")
    
    # If no persisted stores exist, build from source files
    cm_docs, pcs_docs = [], []
    
    print(f"\n📂 Checking directory: {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        print(f"❌ ERROR: Directory '{DATA_DIR}' not found!")
        return
    
    files = os.listdir(DATA_DIR)
    print(f"📋 Found {len(files)} files in directory")
    
    print("\n📖 Processing files:")
    for file in files:
        path = os.path.join(DATA_DIR, file)
        cat, file_type = categorize_file(file)
        if not cat:
            print(f"  ⏭️  Skipping uncategorized file: {file}")
            continue
        
        # Route to appropriate parser based on file type
        if file_type == 'xml':
            if 'cm' in cat:
                cm_docs.extend(parse_xml_file(path, cat))
            else:
                pcs_docs.extend(parse_xml_file(path, cat))
        elif file_type == 'pdf':
            if 'cm' in cat:
                cm_docs.extend(parse_pdf_file(path, cat))
            else:
                pcs_docs.extend(parse_pdf_file(path, cat))
    
    print("\n" + "-"*60)
    print("💾 Creating persistent vector stores in ChromaDB...")
    
    # Create persist directory if it doesn't exist
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    
    if cm_docs:
        print(f"  🔄 Building CM vector store with {len(cm_docs)} documents...")
        cm_vectorstore = Chroma.from_documents(
            cm_docs, 
            embeddings, 
            collection_name="icd10cm",
            persist_directory=cm_persist_path
        )
        print(f"  ✅ Loaded {len(cm_docs)} CM chunks into Chroma")
        print(f"  💾 Persisted to: {cm_persist_path}")
    else:
        print("  ⚠️  No CM documents found")
    
    if pcs_docs:
        print(f"  🔄 Building PCS vector store with {len(pcs_docs)} documents...")
        pcs_vectorstore = Chroma.from_documents(
            pcs_docs, 
            embeddings, 
            collection_name="icd10pcs",
            persist_directory=pcs_persist_path
        )
        print(f"  ✅ Loaded {len(pcs_docs)} PCS chunks into Chroma")
        print(f"  💾 Persisted to: {pcs_persist_path}")
    else:
        print("  ⚠️  No PCS documents found")
    
    print("\n" + "="*60)
    print("🎉 ICD-10 Data Loading Complete!")
    print("="*60 + "\n")