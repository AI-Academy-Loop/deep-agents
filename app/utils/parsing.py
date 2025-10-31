from typing import List, Dict, Any
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from PyPDF2 import PdfReader
import xmlschema

DATA_DIR = "ICD10"
FAISS_PERSIST_DIR = "./faiss_db"  # Directory to persist vector stores

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


def parse_cm_index_xml_xsd(file_path: str) -> List[Document]:
    """Parse ICD-10-CM Index XML using XSD schema validation (like index_pipeline.py)"""
    print(f"  📄 Parsing CM Index XML with XSD: {os.path.basename(file_path)}")
    
    # Find XSD file
    xsd_path = file_path.replace('.xml', '.xsd')
    if not os.path.exists(xsd_path):
        print(f"    ⚠️  XSD not found, using fallback parser")
        return parse_xml_file(file_path, 'cm_index')
    
    try:
        schema = xmlschema.XMLSchema(xsd_path)
        data_dict = schema.to_dict(file_path)
    except Exception as e:
        print(f"    ⚠️  XSD parsing failed: {e}")
        return parse_xml_file(file_path, 'cm_index')
    
    docs = []
    root = data_dict.get("ICD10CM.index", data_dict)
    letters = root.get("letter", [])
    if not isinstance(letters, list):
        letters = [letters]
    
    for letter_node in letters:
        letter_title = letter_node.get("title") if letter_node else None
        main_terms = letter_node.get("mainTerm", []) if letter_node else []
        if not isinstance(main_terms, list):
            main_terms = [main_terms] if main_terms else []
        
        for main in main_terms:
            _process_cm_index_term(main, letter_title, None, None, 0, docs, os.path.basename(file_path))
    
    print(f"    ✅ Extracted {len(docs)} index entries with full hierarchy")
    return docs


def _process_cm_index_term(term_node: Dict, letter: str, parent_title: str, parent_code: str,
                            level: int, docs: List[Document], source: str):
    """Recursively process CM index terms and subterms"""
    if not term_node:
        return
    
    title = term_node.get("title")
    if not title:
        return
    
    # Handle nested title structures
    if isinstance(title, dict):
        title = title.get("$", "") or title.get("nemod", "")
    
    code = term_node.get("code")
    see = term_node.get("see")
    see_also = term_node.get("seeAlso")
    
    # Build structured content
    content_parts = []
    if level == 0:
        content_parts.append(f"Main Term: {title}")
    else:
        content_parts.append(f"Subterm (Level {level}): {title}")
        if parent_title:
            content_parts.append(f"Under: {parent_title}")
    
    if letter:
        content_parts.append(f"Letter: {letter}")
    
    if code:
        content_parts.append(f"ICD-10 Code: {code}")
    
    if see:
        content_parts.append(f"See: {see}")
    
    if see_also:
        content_parts.append(f"See Also: {see_also}")
    
    content = "\n".join(content_parts)
    
    if content.strip():
        docs.append(Document(
            page_content=content,
            metadata={
                'type': 'cm_index',
                'source': source,
                'term': str(title)[:100],
                'level': level,
                'letter': letter or '',
                'code': code or '',
                'parent_term': parent_title or ''
            }
        ))
    
    # Recursively process nested terms
    nested = term_node.get("term", [])
    if nested:
        if not isinstance(nested, list):
            nested = [nested]
        for sub in nested:
            _process_cm_index_term(sub, letter, title, code, level + 1, docs, source)


def parse_cm_tabular_xml_xsd(file_path: str) -> List[Document]:
    """Parse ICD-10-CM Tabular XML using XSD schema"""
    print(f"  📄 Parsing CM Tabular XML with XSD: {os.path.basename(file_path)}")
    
    xsd_path = file_path.replace('.xml', '.xsd')
    if not os.path.exists(xsd_path):
        print(f"    ⚠️  XSD not found, using fallback parser")
        return parse_xml_file(file_path, 'cm_tabular')
    
    try:
        schema = xmlschema.XMLSchema(xsd_path)
        data_dict = schema.to_dict(file_path)
    except Exception as e:
        print(f"    ⚠️  XSD parsing failed: {e}")
        return parse_xml_file(file_path, 'cm_tabular')
    
    docs = []
    root = data_dict.get("ICD10CM.tabular", data_dict)
    
    # Process chapters
    chapters = root.get("chapter", [])
    if not isinstance(chapters, list):
        chapters = [chapters] if chapters else []
    
    for chapter in chapters:
        _process_cm_chapter(chapter, docs, os.path.basename(file_path))
    
    print(f"    ✅ Extracted {len(docs)} diagnosis codes")
    return docs


def _process_cm_chapter(chapter: Dict, docs: List[Document], source: str):
    """Process a chapter and all its diagnoses"""
    if not chapter:
        return
    
    chapter_name = chapter.get("name")
    chapter_desc = chapter.get("desc")
    
    # Process sections in chapter
    sections = chapter.get("section", [])
    if sections and not isinstance(sections, list):
        sections = [sections]
    
    for section in sections:
        _process_cm_section(section, chapter_name, chapter_desc, docs, source)
    
    # Also process diagnoses directly under chapter
    diags = chapter.get("diag", [])
    if diags and not isinstance(diags, list):
        diags = [diags]
    
    for diag in diags:
        _process_cm_diag(diag, chapter_name, chapter_desc, None, docs, source)


def _process_cm_section(section: Dict, chapter_name: str, chapter_desc: str, 
                        docs: List[Document], source: str):
    """Process a section and its diagnoses"""
    if not section:
        return
    
    section_desc = section.get("desc")
    
    # Process diagnoses in section
    diags = section.get("diag", [])
    if diags and not isinstance(diags, list):
        diags = [diags]
    
    for diag in diags:
        _process_cm_diag(diag, chapter_name, chapter_desc, section_desc, docs, source)


def _process_cm_diag(diag: Dict, chapter_name: str, chapter_desc: str, section_desc: str,
                     docs: List[Document], source: str):
    """Process a diagnosis code (can be nested)"""
    if not diag:
        return
    
    code = diag.get("name")
    desc = diag.get("desc")
    
    if not code:
        return
    
    content_parts = []
    content_parts.append(f"Code: {code}")
    
    if desc:
        content_parts.append(f"Description: {desc}")
    
    if chapter_desc:
        content_parts.append(f"Chapter: {chapter_desc}")
    
    if section_desc:
        content_parts.append(f"Section: {section_desc}")
    
    # Extract various notes
    inclusion = diag.get("inclusionTerm")
    if inclusion:
        notes = _extract_notes(inclusion)
        if notes:
            content_parts.append(f"Includes: {'; '.join(notes)}")
    
    excludes1 = diag.get("excludes1")
    if excludes1:
        notes = _extract_notes(excludes1)
        if notes:
            content_parts.append(f"Excludes1 (mutually exclusive): {'; '.join(notes)}")
    
    excludes2 = diag.get("excludes2")
    if excludes2:
        notes = _extract_notes(excludes2)
        if notes:
            content_parts.append(f"Excludes2 (not included here): {'; '.join(notes)}")
    
    code_first = diag.get("codeFirst")
    if code_first:
        notes = _extract_notes(code_first)
        if notes:
            content_parts.append(f"Code First: {'; '.join(notes)}")
    
    use_additional = diag.get("useAdditionalCode")
    if use_additional:
        notes = _extract_notes(use_additional)
        if notes:
            content_parts.append(f"Use Additional Code: {'; '.join(notes)}")
    
    code_also = diag.get("codeAlso")
    if code_also:
        notes = _extract_notes(code_also)
        if notes:
            content_parts.append(f"Code Also: {'; '.join(notes)}")
    
    general_notes = diag.get("notes")
    if general_notes:
        notes = _extract_notes(general_notes)
        if notes:
            content_parts.append(f"Notes: {'; '.join(notes)}")
    
    content = "\n".join(content_parts)
    
    if content.strip():
        docs.append(Document(
            page_content=content,
            metadata={
                'type': 'cm_tabular',
                'source': source,
                'code': code,
                'description': desc[:100] if desc else '',
                'chapter': chapter_name or ''
            }
        ))
    
    # Process nested diagnoses
    nested_diags = diag.get("diag", [])
    if nested_diags:
        if not isinstance(nested_diags, list):
            nested_diags = [nested_diags]
        for nested in nested_diags:
            _process_cm_diag(nested, chapter_name, chapter_desc, section_desc, docs, source)


def _extract_notes(note_container) -> List[str]:
    """Extract note text from various note structures"""
    if not note_container:
        return []
    
    notes = []
    
    # Handle if note_container is a list of field nodes
    if isinstance(note_container, list):
        for field_node in note_container:
            if isinstance(field_node, dict):
                note_val = field_node.get("note")
                if note_val:
                    if isinstance(note_val, list):
                        notes.extend(note_val)
                    else:
                        notes.append(note_val)
    # Handle if note_container is a single dict
    elif isinstance(note_container, dict):
        note_val = note_container.get("note")
        if note_val:
            if isinstance(note_val, list):
                notes.extend(note_val)
            else:
                notes.append(note_val)
    
    # Clean up notes: strip whitespace and handle dict structures
    cleaned_notes = []
    for note in notes:
        if isinstance(note, str):
            cleaned_notes.append(note.strip())
        elif isinstance(note, dict):
            text = note.get("$", "") or str(note)
            if text and text != "{}":
                cleaned_notes.append(text.strip())
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys([n for n in cleaned_notes if n]))


def parse_pcs_index_xml_xsd(file_path: str) -> List[Document]:
    """Parse ICD-10-PCS Index XML using XSD schema"""
    print(f"  📄 Parsing PCS Index XML with XSD: {os.path.basename(file_path)}")
    
    # Find XSD file (handle _2026 naming)
    xsd_path = file_path.replace('.xml', '.xsd')
    if not os.path.exists(xsd_path):
        print(f"    ⚠️  XSD not found: {xsd_path}")
        return []
    
    try:
        schema = xmlschema.XMLSchema(xsd_path)
        data_dict = schema.to_dict(file_path)
    except Exception as e:
        print(f"    ⚠️  XSD parsing failed: {e}")
        return []
    
    docs = []
    root = data_dict.get("ICD10PCS.index", data_dict)
    letters = root.get("letter", [])
    if not isinstance(letters, list):
        letters = [letters]
    
    for letter_node in letters:
        letter_title = letter_node.get("title") if letter_node else None
        main_terms = letter_node.get("mainTerm", []) if letter_node else []
        if not isinstance(main_terms, list):
            main_terms = [main_terms] if main_terms else []
        
        for main in main_terms:
            _process_pcs_index_term(main, letter_title, None, 0, docs, os.path.basename(file_path))
    
    print(f"    ✅ Extracted {len(docs)} PCS index entries")
    return docs


def _process_pcs_index_term(term_node: Dict, letter: str, parent_title: str,
                            level: int, docs: List[Document], source: str):
    """Recursively process PCS index terms"""
    if not term_node:
        return
    
    title = term_node.get("title")
    if not title:
        return
    
    # Build structured content
    content_parts = []
    if level == 0:
        content_parts.append(f"Main Term: {title}")
    else:
        content_parts.append(f"Subterm (Level {level}): {title}")
        if parent_title:
            content_parts.append(f"Under: {parent_title}")
    
    if letter:
        content_parts.append(f"Letter: {letter}")
    
    # Extract references
    code = term_node.get("code")
    if code:
        content_parts.append(f"PCS Code: {code}")
    
    codes = term_node.get("codes")
    if codes:
        content_parts.append(f"PCS Codes: {codes}")
    
    tab = term_node.get("tab")
    if tab:
        content_parts.append(f"Table: {tab}")
    
    see = term_node.get("see")
    if see:
        if isinstance(see, dict):
            see_text = see.get("$", "")
            see_tab = see.get("tab", "")
            see_codes = see.get("codes", "")
            if see_text:
                content_parts.append(f"See: {see_text}")
            if see_tab:
                content_parts.append(f"See Table: {see_tab}")
            if see_codes:
                content_parts.append(f"See Codes: {see_codes}")
        elif isinstance(see, str):
            content_parts.append(f"See: {see}")
    
    use = term_node.get("use")
    if use:
        if isinstance(use, dict):
            use_text = use.get("$", "")
            use_tab = use.get("tab", "")
            if use_text:
                content_parts.append(f"Use: {use_text}")
            if use_tab:
                content_parts.append(f"Use Table: {use_tab}")
        elif isinstance(use, str):
            content_parts.append(f"Use: {use}")
    
    content = "\n".join(content_parts)
    
    if content.strip():
        docs.append(Document(
            page_content=content,
            metadata={
                'type': 'pcs_index',
                'source': source,
                'term': str(title)[:100],
                'level': level,
                'letter': letter or '',
                'parent_term': parent_title or ''
            }
        ))
    
    # Recursively process nested terms
    nested = term_node.get("term", [])
    if nested:
        if not isinstance(nested, list):
            nested = [nested]
        for sub in nested:
            _process_pcs_index_term(sub, letter, title, level + 1, docs, source)


def parse_pcs_tables_xml_xsd(file_path: str) -> List[Document]:
    """Parse ICD-10-PCS Tables XML using XSD schema"""
    print(f"  📄 Parsing PCS Tables XML with XSD: {os.path.basename(file_path)}")
    
    xsd_path = file_path.replace('.xml', '.xsd')
    if not os.path.exists(xsd_path):
        print(f"    ⚠️  XSD not found: {xsd_path}")
        return []
    
    try:
        schema = xmlschema.XMLSchema(xsd_path)
        data_dict = schema.to_dict(file_path)
    except Exception as e:
        print(f"    ⚠️  XSD parsing failed: {e}")
        return []
    
    docs = []
    root = data_dict.get("ICD10PCS.tabular", data_dict)
    
    # Process PCS tables
    tables = root.get("pcsTable", [])
    if not isinstance(tables, list):
        tables = [tables] if tables else []
    
    for table in tables:
        _process_pcs_table(table, docs, os.path.basename(file_path))
    
    print(f"    ✅ Extracted {len(docs)} PCS tables")
    return docs


def _process_pcs_table(table: Dict, docs: List[Document], source: str):
    """Process a single PCS table"""
    if not table:
        return
    
    content_parts = []
    
    # Extract the first 3 axes that define the table
    axes = table.get("axis", [])
    if not isinstance(axes, list):
        axes = [axes] if axes else []
    
    table_code = ""
    for axis in sorted([a for a in axes if a.get("pos") in [1, 2, 3]], 
                      key=lambda x: x.get("pos", 0)):
        pos = axis.get("pos")
        title = axis.get("title", "")
        label = axis.get("label", {})
        if isinstance(label, list):
            label = label[0] if label else {}
        
        code_val = label.get("@code", "") if isinstance(label, dict) else ""
        label_text = label.get("$", "") if isinstance(label, dict) else str(label)
        
        table_code += code_val
        content_parts.append(f"Axis {pos} - {title}: {code_val} {label_text}")
        
        # Add definition if present (usually for 3rd axis)
        if pos == 3:
            definition = axis.get("definition", "")
            if definition:
                content_parts.append(f"Definition: {definition}")
    
    content_parts.append(f"\nTable Code: {table_code}")
    
    # Process rows (combinations of axes 4-7)
    rows = table.get("pcsRow", [])
    if not isinstance(rows, list):
        rows = [rows] if rows else []
    
    if rows:
        content_parts.append(f"Valid combinations: {len(rows)} rows")
        # Sample first row to show structure
        if rows:
            first_row = rows[0]
            row_axes = first_row.get("axis", [])
            if not isinstance(row_axes, list):
                row_axes = [row_axes] if row_axes else []
            
            for axis in row_axes:
                title = axis.get("title", "")
                labels = axis.get("label", [])
                if not isinstance(labels, list):
                    labels = [labels] if labels else []
                label_count = len(labels)
                content_parts.append(f"  {title}: {label_count} options")
    
    content = "\n".join(content_parts)
    
    if content.strip():
        docs.append(Document(
            page_content=content,
            metadata={
                'type': 'pcs_tables',
                'source': source,
                'table_code': table_code
            }
        ))


def parse_pcs_definitions_xml_xsd(file_path: str) -> List[Document]:
    """Parse ICD-10-PCS Definitions XML using XSD schema"""
    print(f"  📄 Parsing PCS Definitions XML with XSD: {os.path.basename(file_path)}")
    
    xsd_path = file_path.replace('.xml', '.xsd')
    if not os.path.exists(xsd_path):
        print(f"    ⚠️  XSD not found: {xsd_path}")
        return []
    
    try:
        schema = xmlschema.XMLSchema(xsd_path)
        data_dict = schema.to_dict(file_path)
    except Exception as e:
        print(f"    ⚠️  XSD parsing failed: {e}")
        return []
    
    docs = []
    root = data_dict.get("ICD10PCS.definitions", data_dict)
    
    # Process sections
    sections = root.get("section", [])
    if not isinstance(sections, list):
        sections = [sections] if sections else []
    
    for section in sections:
        _process_pcs_section(section, docs, os.path.basename(file_path))
    
    print(f"    ✅ Extracted {len(docs)} PCS definitions")
    return docs


def _process_pcs_section(section: Dict, docs: List[Document], source: str):
    """Process PCS definitions section"""
    if not section:
        return
    
    section_code = section.get("@code", "")
    section_title = section.get("title", "")
    
    # Process axes in this section
    axes = section.get("axis", [])
    if not isinstance(axes, list):
        axes = [axes] if axes else []
    
    for axis in axes:
        axis_pos = axis.get("@pos", "")
        axis_title = axis.get("title", "")
        
        # Process terms in this axis
        terms_list = axis.get("terms", [])
        if not isinstance(terms_list, list):
            terms_list = [terms_list] if terms_list else []
        
        for terms in terms_list:
            content_parts = []
            content_parts.append(f"Section: {section_code} - {section_title}")
            content_parts.append(f"Axis {axis_pos}: {axis_title}")
            
            titles = terms.get("title", [])
            if not isinstance(titles, list):
                titles = [titles] if titles else []
            if titles:
                content_parts.append(f"Term: {', '.join(str(t) for t in titles)}")
            
            definition = terms.get("definition", "")
            if definition:
                content_parts.append(f"Definition: {definition}")
            
            explanation = terms.get("explanation", "")
            if explanation:
                content_parts.append(f"Explanation: {explanation}")
            
            includes = terms.get("includes", [])
            if not isinstance(includes, list):
                includes = [includes] if includes else []
            if includes:
                content_parts.append(f"Includes: {', '.join(str(i) for i in includes)}")
            
            content = "\n".join(content_parts)
            
            if content.strip():
                docs.append(Document(
                    page_content=content,
                    metadata={
                        'type': 'pcs_definitions',
                        'source': source,
                        'section': section_code,
                        'axis': axis_pos
                    }
                ))


def parse_xml_file(file_path: str, doc_type: str) -> List[Document]:
    """Route to appropriate XSD-aware parser based on file type"""
    filename = os.path.basename(file_path).lower()
    
    # Route to XSD-aware parsers
    if 'icd10cm-index' in filename:
        return parse_cm_index_xml_xsd(file_path)
    elif 'icd10cm-tabular' in filename:
        return parse_cm_tabular_xml_xsd(file_path)
    elif 'icd10pcs' in filename:
        # Route to specific PCS parser based on filename
        if 'index' in filename:
            return parse_pcs_index_xml_xsd(file_path)
        elif 'tables' in filename:
            return parse_pcs_tables_xml_xsd(file_path)
        elif 'definitions' in filename:
            return parse_pcs_definitions_xml_xsd(file_path)
        else:
            print(f"  ⚠️  Unknown PCS file type: {filename}")
            return []
    else:
        print(f"  ⚠️  Unknown XML file type: {filename}")
        return []

def parse_pdf_file(file_path: str, doc_type: str) -> List[Document]:
    """Parse any PDF file by extracting text from each page and chunking by topics."""
    print(f"  📄 Parsing PDF: {os.path.basename(file_path)}")
    reader = PdfReader(file_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and len(text.strip()) > 100:
            # Split text into topics based on headings or significant breaks
            topics = [topic.strip() for topic in text.split('\n') if len(topic.strip()) > 100]
            for topic_idx, topic in enumerate(topics):
                docs.append(Document(
                    page_content=topic,
                    metadata={
                        'page': i + 1,
                        'topic': topic_idx,
                        'type': doc_type,
                        'source': os.path.basename(file_path)
                    }
                ))
    print(f"    ✅ Extracted {len(docs)} topics from {len(reader.pages)} pages")
    return docs


def faiss_store_exists(path):
    return os.path.exists(os.path.join(path, "index.faiss")) and os.path.exists(os.path.join(path, "index.pkl"))

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
    cm_persist_path = os.path.join(FAISS_PERSIST_DIR, "icd10cm")
    pcs_persist_path = os.path.join(FAISS_PERSIST_DIR, "icd10pcs")
    
    if faiss_store_exists(cm_persist_path) and faiss_store_exists(pcs_persist_path):
        print("\n📦 Found existing FAISS vector stores on disk - loading from persistence...")
        try:
            cm_vectorstore = FAISS.load_local(cm_persist_path, embeddings, allow_dangerous_deserialization=True)
            pcs_vectorstore = FAISS.load_local(pcs_persist_path, embeddings, allow_dangerous_deserialization=True)
            cm_count = len(cm_vectorstore.docstore._dict)
            pcs_count = len(pcs_vectorstore.docstore._dict)
            print(f"  ✅ Loaded CM vector store: {cm_count} documents")
            print(f"  ✅ Loaded PCS vector store: {pcs_count} documents")
            print("\n" + "="*60)
            print("🎉 FAISS Vector Stores Loaded from Disk!")
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
    print("💾 Creating persistent vector stores em FAISS...")
    
    # Create persist directory if it doesn't exist
    os.makedirs(FAISS_PERSIST_DIR, exist_ok=True)
    
    if cm_docs:
        print(f"  🔄 Building CM vector store com {len(cm_docs)} documents...")
        cm_vectorstore = FAISS.from_documents(cm_docs, embeddings)
        cm_vectorstore.save_local(cm_persist_path)
        print(f"  ✅ Loaded {len(cm_docs)} CM chunks into FAISS")
        print(f"  💾 Persisted to: {cm_persist_path}")
    else:
        print("  ⚠️  No CM documents found")

    if pcs_docs:
        print(f"  🔄 Building PCS vector store com {len(pcs_docs)} documents...")
        pcs_vectorstore = FAISS.from_documents(pcs_docs, embeddings)
        pcs_vectorstore.save_local(pcs_persist_path)
        print(f"  ✅ Loaded {len(pcs_docs)} PCS chunks into FAISS")
        print(f"  💾 Persisted to: {pcs_persist_path}")
    else:
        print("  ⚠️  No PCS documents found")
    
    print("\n" + "="*60)
    print("🎉 ICD-10 Data Loading Complete!")
    print("="*60 + "\n")