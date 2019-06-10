# Makefile for generating presentation $(TARGET).pdf from $(TARGET).tex

TARGET = LilithForRun2_v0
Refs = references.bib
Subfiles = $(TARGET).tex

PLOTS = 

all: $(TARGET).pdf $(Subfiles) $(Refs)

$(TARGET).pdf: $(TARGET).tex $(Refs) $(Subfiles)
	pdflatex $(TARGET).tex
	bibtex $(TARGET)
	pdflatex $(TARGET).tex
	pdflatex $(TARGET).tex
#	dvips $(TARGET).dvi -o
#	ps2pdf $(TARGET).ps $(TARGET).pdf
#	xpdf $(TARGET).pdf &

clean:
	rm -f *.dvi *.log *.aux *~ *.bbl *.blg

