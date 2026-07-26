#!/bin/bash
# Compiles the LaTeX report, puts auxiliary files in generated/, and keeps PDF in root.

# Create generated directory if it doesn't exist
mkdir -p generated

# Compile LaTeX with output directory set to generated/
pdflatex -output-directory=generated Andrew_Donate_CAP_6665_Project_2_Report.tex

# Move the resulting PDF back to the root directory
mv generated/Andrew_Donate_CAP_6665_Project_2_Report.pdf .

echo "Compilation complete! Auxiliary files stored in generated/."
