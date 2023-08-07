.POSIX:

OUT = report

all:
	latexmk -pdf -lualatex="lualatex -interaction=nonstopmode" -use-make $(OUT).tex
	biber $(OUT)

clean:
	rm -f *.aux *.log *.out *.toc *.pdf *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.xml

.PHONY: all clean
