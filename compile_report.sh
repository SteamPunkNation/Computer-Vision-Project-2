#!/bin/bash
# Compiles the LaTeX report, puts auxiliary files in generated/, and keeps PDF in root.

# Create generated directory if it doesn't exist
mkdir -p generated

# Compile LaTeX with output directory set to generated/
pdflatex -output-directory=generated project_report.tex

# Move the resulting PDF back to the root directory
mv generated/project_report.pdf .

echo "Compilation complete! Auxiliary files stored in generated/."
