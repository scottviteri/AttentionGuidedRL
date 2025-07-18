# Makefile for Mathematical Formulation LaTeX Document

# Variables
TEX_FILE = mathematical_formulation
PDF_FILE = $(TEX_FILE).pdf
TEX_SOURCE = $(TEX_FILE).tex

# Default target
all: $(PDF_FILE)

# Compile LaTeX to PDF
$(PDF_FILE): $(TEX_SOURCE)
	pdflatex $(TEX_SOURCE)
	pdflatex $(TEX_SOURCE)  # Run twice for cross-references

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.out *.toc *.synctex.gz *.fdb_latexmk *.fls

# Clean everything including PDF
clean-all: clean
	rm -f $(PDF_FILE)

# Force rebuild
rebuild: clean-all all

# Open PDF (macOS)
open: $(PDF_FILE)
	open $(PDF_FILE)

# Open PDF (Linux)
view: $(PDF_FILE)
	xdg-open $(PDF_FILE)

# Check for LaTeX installation
check:
	@which pdflatex > /dev/null || (echo "pdflatex not found. Please install TeX Live or MiKTeX" && exit 1)
	@echo "LaTeX installation found"

.PHONY: all clean clean-all rebuild open view check 