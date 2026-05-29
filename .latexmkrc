# Keep the final PDF, but remove LaTeX scratch files after successful builds.
# This applies to direct `latexmk writeup.tex` runs and to tools that invoke
# latexmk, including VS Code's LaTeX Workshop extension.
$success_cmd = 'latexmk -c %T >/dev/null 2>&1';

# Include common extras produced by hyperref, biber, makeindex, and synctex.
$clean_ext = 'acn acr alg auxlock bcf glg glo gls ist nav run.xml snm synctex.gz vrb';
