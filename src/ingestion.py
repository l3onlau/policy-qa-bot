import os
import re
import uuid
from typing import List, Dict, Any
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────────────────────────────────────
# Labels that contain meaningful text content — all should be ingested.
# Includes standard text elements as well as footnotes, captions, index
# entries, abbreviations, and OCR text extracted from picture elements.
# ─────────────────────────────────────────────────────────────────────────────
TEXT_BEARING_LABELS = {
    DocItemLabel.TEXT,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.FOOTNOTE,
    DocItemLabel.CAPTION,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.REFERENCE,
    DocItemLabel.FORMULA,
    DocItemLabel.CODE,
    DocItemLabel.KEY_VALUE_REGION,
}


class PolicyIngestor:
    """
    Parses PDF policy documents using Docling and produces parent-child chunk pairs
    with section-aware metadata for the vector store.

    Uses a two-pass strategy:
      Pass 1 (Structured): Item-by-item extraction with heading tracking.
                            Handles all text-bearing labels, tables, and pictures.
      Pass 2 (Reconciliation): Page-level gap detection creates supplementary
                                chunks for pages where the structured pass missed
                                significant content.
    """

    def __init__(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        # Parent chunks: ~750 tokens — what the LLM sees (PRD: 400-800 tokens)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=300,
            length_function=len,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
        )

        # Child chunks: ~200 tokens — what FAISS indexes for precise search
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " "],
        )

    def extract_clause(self, text: str) -> str:
        """Extract a clause number from a heading (e.g., '1.', '3.2', 'Section 4')."""
        match = re.search(
            r"^((?:Section\s*)?\d+(?:\.\d+)*[.:]?|[A-Z]\.?)(\s+|$)",
            text,
            re.IGNORECASE,
        )
        return match.group(1).strip(".:") if match else ""

    # ─────────────────────────────────────────────────────────────────────
    # Shared chunk-creation helper
    # ─────────────────────────────────────────────────────────────────────

    def _make_chunks(
        self,
        full_text: str,
        heading_path: List[str],
        clause: str,
        page: Any,
        file: str,
        label_suffix: str = "",
    ) -> List[Dict[str, Any]]:
        """Create parent-child chunk pairs from text with section metadata."""
        if not full_text.strip():
            return []

        chunks: List[Dict[str, Any]] = []
        current_section = heading_path[-1] if heading_path else "General Policy"
        path_str = " > ".join(heading_path)
        display_path = f"{path_str} ({label_suffix})" if label_suffix else path_str

        parent_chunks = self.parent_splitter.split_text(full_text)

        for parent_text in parent_chunks:
            parent_id = str(uuid.uuid4())
            parent_with_heading = f"## {display_path}\n\n{parent_text}"
            child_chunks = self.child_splitter.split_text(parent_text)

            for child_text in child_chunks:
                child_with_heading = f"## {display_path}\n\n{child_text}"
                chunks.append(
                    {
                        "content": child_with_heading,
                        "parent_content": parent_with_heading,
                        "metadata": {
                            "doc_name": file,
                            "section": current_section,
                            "clause_number": clause,
                            "page": page,
                            "heading_path": path_str,
                            "parent_id": parent_id,
                        },
                    }
                )
        return chunks

    # ─────────────────────────────────────────────────────────────────────
    # Pass-2 helpers: page-level content reconciliation
    # ─────────────────────────────────────────────────────────────────────

    def _collect_all_page_text(self, doc_data) -> Dict[int, List[str]]:
        """
        Collect ALL text from every item on every page, regardless of label type.
        Used by the reconciliation pass to detect gaps in structured extraction.
        """
        page_texts: Dict[int, List[str]] = {}
        for item, _depth in doc_data.iterate_items():
            text = ""
            if hasattr(item, "text") and item.text:
                text = item.text.strip()
            if not text:
                continue
            page = item.prov[0].page_no if item.prov else None
            if page is not None:
                page_texts.setdefault(page, []).append(text)
        return page_texts

    def _build_page_heading_map(self, doc_data) -> Dict[int, List[str]]:
        """
        Build a map of page_no → heading_path that was active on that page.
        The reconciliation pass uses this to assign correct section metadata
        to supplementary chunks.
        """
        heading_path = ["General Policy"]
        page_headings: Dict[int, List[str]] = {}

        for item, depth in doc_data.iterate_items():
            if item.label == DocItemLabel.SECTION_HEADER:
                heading_text = (
                    item.text.strip()
                    if hasattr(item, "text") and item.text
                    else "General Policy"
                )
                if depth <= len(heading_path):
                    heading_path = heading_path[: depth - 1]
                heading_path.append(heading_text)

            page = item.prov[0].page_no if item.prov else None
            if page is not None:
                # Always update to the latest heading for this page
                page_headings[page] = list(heading_path)

        return page_headings

    # ─────────────────────────────────────────────────────────────────────
    # Main processing pipeline
    # ─────────────────────────────────────────────────────────────────────

    def process_pdfs(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Process all PDFs in data_path using a two-pass strategy.

        Returns a list of child documents, each containing:
        - 'content': child chunk text (small, for FAISS indexing)
        - 'parent_content': parent chunk text (large, for LLM context)
        - 'metadata': {doc_name, section, clause_number, page, heading_path, parent_id}
        """
        all_documents: List[Dict[str, Any]] = []

        for file in os.listdir(data_path):
            if not file.endswith(".pdf"):
                continue

            file_documents: List[Dict[str, Any]] = []
            file_path = os.path.join(data_path, file)
            print(f"🧐 Parsing document: {file}...")

            result = self.converter.convert(file_path)
            doc_data = result.document

            # ── Pass 1: Structured item-by-item extraction ────────────────
            heading_path: List[str] = ["General Policy"]
            current_clause = ""
            section_buffer: List[str] = []
            last_page: Any = "??"

            # Track which text was captured per page (for reconciliation)
            structured_page_chars: Dict[int, int] = {}

            def track_content(text: str, page: Any):
                """Accumulate character count captured per page."""
                if isinstance(page, int) and text.strip():
                    structured_page_chars[page] = structured_page_chars.get(
                        page, 0
                    ) + len(text.strip())

            def flush_section_buffer(
                path: List[str], clause: str, buffer: List[str], page: Any
            ):
                if not buffer:
                    return
                full_text = "\n".join(buffer)
                track_content(full_text, page)
                chunks = self._make_chunks(full_text, path, clause, page, file)
                file_documents.extend(chunks)
                buffer.clear()

            for item, depth in doc_data.iterate_items():
                # ── Update page from every item (not just text) ───────────
                item_page = item.prov[0].page_no if item.prov else None
                if item_page is not None:
                    last_page = item_page

                # ── SECTION_HEADER: update heading path ───────────────────
                if item.label == DocItemLabel.SECTION_HEADER:
                    flush_section_buffer(
                        heading_path, current_clause, section_buffer, last_page
                    )
                    heading_text = (
                        item.text.strip()
                        if hasattr(item, "text") and item.text
                        else "General Policy"
                    )

                    if depth <= len(heading_path):
                        heading_path = heading_path[: depth - 1]
                    heading_path.append(heading_text)

                    current_clause = self.extract_clause(heading_text) or current_clause

                # ── Text-bearing labels: buffer for section chunking ──────
                elif item.label in TEXT_BEARING_LABELS:
                    content = item.text if hasattr(item, "text") else ""
                    if content:
                        if item.label == DocItemLabel.LIST_ITEM:
                            content = f"* {content}"
                        elif item.label == DocItemLabel.FOOTNOTE:
                            content = f"[Footnote] {content}"
                        elif item.label == DocItemLabel.CAPTION:
                            content = f"[Caption] {content}"
                        section_buffer.append(content)

                # ── TABLE: flush buffer, then process table separately ────
                elif item.label == DocItemLabel.TABLE:
                    flush_section_buffer(
                        heading_path, current_clause, section_buffer, last_page
                    )
                    if hasattr(item, "export_to_markdown"):
                        table_md = item.export_to_markdown(doc=doc_data)
                        track_content(table_md, last_page)

                        chunks = self._make_chunks(
                            table_md,
                            heading_path,
                            current_clause,
                            last_page,
                            file,
                            label_suffix="Table",
                        )
                        file_documents.extend(chunks)

                # ── PICTURE: extract any OCR text it may carry ────────────
                elif item.label == DocItemLabel.PICTURE:
                    content = item.text if hasattr(item, "text") and item.text else ""
                    # Only ingest if there's meaningful text (>20 chars)
                    if content and len(content.strip()) > 20:
                        track_content(content, last_page)
                        section_buffer.append(content)

            # Flush any remaining buffered text
            flush_section_buffer(
                heading_path, current_clause, section_buffer, last_page
            )

            # ── Pass 2: Page-level reconciliation ─────────────────────────
            # Detect pages where structured extraction captured < 40% of
            # the available text and create supplementary chunks from all
            # text on those pages.  This catches content in exotic label
            # types or layout edge-cases that the structured pass missed.
            all_page_text = self._collect_all_page_text(doc_data)
            page_heading_map = self._build_page_heading_map(doc_data)

            supplementary_count = 0
            for page_no in sorted(all_page_text.keys()):
                total_page_len = sum(len(t) for t in all_page_text[page_no])
                structured_len = structured_page_chars.get(page_no, 0)
                coverage = structured_len / max(total_page_len, 1)

                if coverage < 0.4 and total_page_len > 50:
                    page_heading = page_heading_map.get(page_no, ["General Policy"])
                    combined_text = "\n".join(all_page_text[page_no])

                    chunks = self._make_chunks(
                        combined_text,
                        page_heading,
                        "",
                        page_no,
                        file,
                        label_suffix="Supplementary",
                    )
                    file_documents.extend(chunks)
                    supplementary_count += len(chunks)

            print(
                f"   ✅ Processed {file}: {len(file_documents)} chunks generated"
                f" ({supplementary_count} supplementary)."
            )
            all_documents.extend(file_documents)

        return all_documents
