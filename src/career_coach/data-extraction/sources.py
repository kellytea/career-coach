"""Bounded, fixed-host Jobicy connector. No arbitrary URL fetching."""

import json
import csv
import sys

import httpx


def dataset_records(path, *, container_format, text_column, id_column=None, limit=None, sheet=None):
    """Stream CSV/XLSX rows using explicit columns; never infer ground-truth labels."""
    def records(header, rows):
        if not header or len(set(header)) != len(header):
            raise ValueError("Dataset must have a non-empty, unique column header")
        required = {text_column} | ({id_column} if id_column else set())
        if not required.issubset(header):
            raise ValueError(f"Missing dataset columns: {sorted(required - set(header))}")
        for number, cells in enumerate(rows, start=1):
            if limit is not None and number > limit:
                break
            if len(cells) != len(header):
                raise ValueError(f"Column count mismatch at dataset record {number}")
            row = dict(zip(header, cells))
            content = row.get(text_column)
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Empty/non-text document at dataset record {number}")
            identity = row.get(id_column) if id_column else number
            if identity is None or not str(identity).strip() or len(str(identity)) > 512:
                raise ValueError(f"Invalid identity at dataset record {number}")
            yield number, str(identity), content

    if container_format == "xlsx":
        from openpyxl import load_workbook
        # Archive objects have no extension; pass a binary stream to openpyxl.
        with path.open("rb") as stream:
            workbook = load_workbook(stream, read_only=True, data_only=True)
            try:
                if sheet and sheet not in workbook.sheetnames:
                    raise ValueError("Requested worksheet does not exist")
                worksheet = workbook[sheet] if sheet else workbook.active
                rows = worksheet.iter_rows(values_only=True)
                header = [str(value) if value is not None else "" for value in next(rows, ())]
                yield from records(header, rows)
            finally:
                workbook.close()
        return
    if container_format != "csv":
        raise ValueError("Supported dataset containers: csv, xlsx")
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        yield from records(next(reader, []), reader)


def jobicy_pages(filters, archive_response):
    params = {key: value for key, value in filters.items() if key in {"count", "geo", "industry", "tag"} and value}
    seen = set()
    with httpx.Client(timeout=httpx.Timeout(30, connect=10), follow_redirects=False,
                      headers={"Accept": "application/json", "User-Agent": "career-coach/0.1"}) as client:
        for _ in range(filters.get("max_pages", 20)):
            with client.stream("GET", "https://jobicy.com/api/jobs", params=params) as response:
                payload = bytearray()
                for chunk in response.iter_bytes():
                    payload.extend(chunk)
                    if len(payload) > 20_000_000:
                        raise ValueError("Jobicy response exceeds 20 MB limit")
                archive_response(bytes(payload), response.status_code)
                response.raise_for_status()
            data = json.loads(payload)
            if data.get("success") is False or not isinstance(data.get("jobs"), list):
                raise ValueError("Unexpected Jobicy response")
            yield data["jobs"]
            cursor = data.get("nextCursor")
            if not cursor:
                return
            if not isinstance(cursor, str) or cursor in seen:
                raise ValueError("Invalid or repeated Jobicy cursor")
            seen.add(cursor)
            params["cursor"] = cursor
    raise ValueError("Jobicy page limit reached; partial results retained")
