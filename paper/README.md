# Paper: Burst to Flex (CCGrid 2027)

IEEEtran conference, 10 pages including references, double-blind (no names, university, repo links, PDF metadata).
Section files are in `sections/`, each with its page budget as a comment (budgets: docs/track-2/PAPER_PLAN.md section 1).
`numbers.tex` is generated (step S16); never type a result by hand.

## Build
    cd paper
    latexmk -pdf main.tex        # output: main.pdf
    latexmk -c                   # clean aux files

No LaTeX on Windows? Either:
- Install MiKTeX (https://miktex.org/download), then reopen the terminal and check `latexmk -v`. MiKTeX needs Perl for
  latexmk (install Strawberry Perl) and installs missing packages on the fly (allow it when prompted).
- Or use Overleaf: New Project > Upload Project, zip the contents of `paper/` (main.tex, numbers.tex, refs.bib,
  sections/, later figures/out/ and tables/). Set main.tex as the main document and compile with pdfLaTeX.
