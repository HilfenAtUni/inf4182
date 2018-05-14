# command
CP=/bin/cp
LATEX=xelatex
CC=/usr/bin/gcc
CFLAGS=-Wall
# folder
SOLUTION=solution02
LATEX_FILE=${SOLUTION}.tex


all: pdf solution

pre:

pdf:
	$(LATEX) ${LATEX_FILE}

test:

clean:
	rm -rvf print *.obj *.o *.tail dbform *.aux *.log *.out

solution:
	git add ${SOLUTION}.ipynb ${SOLUTION}.pdf ${SOLUTION}.tex makefile
	# $(CP) -vf ${SOLUTION}.pdf ../${SOLUTION}
	# $(CP) -vf ${SOLUTION}.sql ../${SOLUTION}
	# $(CP) -rvf makefile ../${SOLUTION}

