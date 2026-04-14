# Policies Folder

Place your company HR policy PDF files here before running the ingestion script.

Examples:
- annual_leave_policy.pdf
- sick_leave_policy.pdf
- employee_handbook.pdf

Then run:
```
python ingest/ingest_pdfs.py
```

If no PDFs are present, a built-in sample policy is used automatically so you can
test the system straight away.
