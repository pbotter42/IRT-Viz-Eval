# CEJEME Submission Steps

## Empirical results completed

1. GLM-4.5V and Gemma 3 4B each completed the frozen 198-task form through Hugging Face Inference Providers.
2. The empirical preparation script deduplicates response identifiers, requires complete coverage, applies deterministic scoring, and calculates cluster-bootstrap intervals.
3. Do not rerun or replace the frozen model responses unless the manuscript is revised to describe a new administration.
4. Never place Hugging Face or vendor API tokens in the repository or submission files.

## Author details to complete

1. Replace affiliation, correspondence, ORCID, funding, and competing-interest placeholders in `CEJEME_Title_Page.md` and `CEJEME_Cover_Letter.md`.
2. Confirm the blind-review funding and competing-interest statements in the manuscript and complete their substantive versions on the title page.
3. Verify every author has approved the manuscript and submission.
4. Decide whether to provide a Chinese translation after acceptance or use the journal's translation process.

## Build and review

1. Run `bash scripts/build_cejeme_submission.sh`.
2. Confirm `output/submission/CEJEME_Main_Document.pdf` is no more than 40 pages.
3. Check that the abstract is no more than 120 words and there are 3-5 keywords.
4. Inspect all pages, tables, and 14 figures at readable zoom.
5. Confirm the main document contains no author identity or public repository URL.
6. If requested by the editor, upload `output/submission/CEJEME_Anonymous_Code_Supplement.zip` as supplemental material.

## Upload to CEJEME

1. Main Document: `output/submission/CEJEME_Main_Document.pdf`.
2. LaTeX source: `output/submission/CEJEME_LaTeX_Source.zip`.
3. Optional anonymous code supplement: `output/submission/CEJEME_Anonymous_Code_Supplement.zip`.
4. Cover letter: use `CEJEME_Cover_Letter.md` after completing placeholders.
5. Enter the title, abstract, 3-5 keywords, author metadata, and declarations in the submission form.
6. Do not submit the manuscript elsewhere while CEJEME is considering it.

## After acceptance or the end of blind review

1. Make the GitHub repository public and archive a release with Zenodo or another DOI service.
2. Replace the withheld repository statement with the permanent URL/DOI.
3. Review the translated article and translator authorship with the editorial office.
4. Confirm the final licensing and copyright terms before attaching a manuscript license.
