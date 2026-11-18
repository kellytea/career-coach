"""Adapt uploaded PDFs to the existing parser's ontology output, without source changes."""

from pathlib import Path
import tempfile
from urllib.parse import unquote

import pymupdf

from career_coach.data_extraction.entity_extraction import EntityExtractor
from career_coach.data_extraction.ontology import OntologyDocument, OntologyOutput
from career_coach.data_extraction.ontology_population import OntologyPopulator
from career_coach.data_extraction.parser import ResumeParser


MAX_PDF_BYTES = 6 * 1024 * 1024
MAX_PDF_PAGES = 30


class PDFResumeParser:
    def __init__(self, taxonomy, mode="local"):
        self.resolver = taxonomy.resolver
        self.mode = mode
        self.parser = ResumeParser()
        self.extractor = EntityExtractor()
        self.populator = OntologyPopulator()

    def parse(self, content, filename="resume.pdf"):
        filename = Path(unquote(filename).replace("\\", "/")).name
        if (not filename or len(filename) > 200 or not filename.lower().endswith(".pdf")
                or any(ord(c) < 32 for c in filename)):
            raise ValueError("Choose a resume PDF with a valid filename")
        if not content or len(content) > MAX_PDF_BYTES:
            raise ValueError("Choose a PDF under 6 MB")
        if not content.startswith(b"%PDF-"):
            raise ValueError("This file is not a readable PDF. Please choose a PDF resume")
        try:
            with pymupdf.open(stream=content, filetype="pdf") as document:
                if document.needs_pass:
                    raise ValueError("This PDF is password-protected. Upload an unlocked copy")
                pages = document.page_count
                if not 1 <= pages <= MAX_PDF_PAGES:
                    raise ValueError("Upload a resume PDF with 1 to 30 pages")
                source = "\n".join(page.get_text("text") for page in document)
        except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError) as error:
            raise ValueError("This PDF could not be read. Try exporting a new PDF copy") from error
        if sum(c.isalnum() for c in source) < 20:
            raise ValueError("No selectable resume text was found. Export a text PDF; scanned images need OCR first")
        if len(source) > 200_000:
            raise ValueError("This PDF contains too much text. Upload a shorter resume")

        # The upstream parser is directory-based. Keep the upload only while it runs.
        with tempfile.TemporaryDirectory(prefix="career-coach-pdf-") as directory:
            (Path(directory) / "resume.pdf").write_bytes(content)
            features = self.parser.run(directory).to_dict(orient="records")[0]
        features["cv_id"] = filename
        if self.mode == "full":
            from career_coach.data_extraction.llm_mapping import LLMMappingResolver

            entities = self.extractor.extract_entities(features)
            mappings = self.resolver.resolve_entities(entities, document_id=filename)
            reviewed = LLMMappingResolver(self.resolver).resolve(features, mappings)
        else:
            # Same explicit entities, taxonomy resolution and ontology validation as
            # upstream, with no downloaded NER model or LLM inference in local mode.
            entities = self.extractor.explicit_entities(features)
            mappings = self.resolver.resolve_entities(entities, document_id=filename)
            reviewed = {"cv_id": filename, "entity_mappings": mappings}
        populated = self.populator.populate(features, reviewed)
        output = OntologyOutput(resumes=[OntologyDocument(cv_id=filename, nodes=populated.nodes)])
        return output.model_dump_json(), {"mode": self.mode, "pages": pages, "warnings": []}
