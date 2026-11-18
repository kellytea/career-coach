"""Parse resumes, resolve/review entities, then populate the job-match ontology."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from .parser import ResumeParser
from .entity_extraction import EntityExtractor, TaxonomyResolver


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--rules-only", action="store_true", help="Skip GLiNER and save only parsed features")
    parser.add_argument("--taxonomy", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data/taxonomy/onet/31.0",
                        help="Directory containing the pinned O*NET JSONL lookups")
    args = parser.parse_args()

    parser = ResumeParser()
    results = parser.run(args.input)
    records = results.to_dict(orient="records")

    output = args.input.with_suffix(".json")
    if args.rules_only:
        payload = json.dumps(records, ensure_ascii=False, indent=2)
    else:
        from .llm_mapping import LLMMappingResolver
        from .ontology import OntologyDocument, OntologyOutput
        from .ontology_population import OntologyPopulator

        extractor = EntityExtractor()
        resolver = TaxonomyResolver(args.taxonomy)
        llm = LLMMappingResolver(resolver)
        populator = OntologyPopulator()
        documents = []
        for features in records:
            entities = extractor.extract_entities(features)
            mappings = resolver.resolve_entities(entities, document_id=features["cv_id"])
            print(f"Reviewing explicit entities and mappings for {features['cv_id']} via LLM...")
            reviewed = llm.resolve(features, mappings)
            audit = reviewed["llm_audit"]
            if "timing" in audit:
                print(f"LLM review completed in {audit['timing']['total_seconds']:.2f}s ({audit['calls']} requests)")
            populated = populator.populate(features, reviewed)
            documents.append(OntologyDocument(cv_id=features["cv_id"], nodes=populated.nodes))
        payload = OntologyOutput(resumes=documents).model_dump_json(indent=2)

    # Write only after every resume has completed validation successfully.
    output.write_text(payload + "\n", encoding="utf-8")
    print(f"Saved {len(records)} resumes to {output}")
    print(f"Pipeline completed in {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
