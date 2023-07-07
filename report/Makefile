.POSIX:

OUT = report

all:
	pdflatex --shell-escape --interaction=nonstopmode -jobname $(OUT) --halt-on-error report.tex

clean:
	rm -f *.aux *.log *.out *.toc *.pdf

.PHONY: all clean
