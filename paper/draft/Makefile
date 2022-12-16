all:
	sed -i '/begin/d' ./evidence.tex
	sed -i '/end/d' ./evidence.tex
	sed -i '/sampler/d' ./evidence.tex
	sed -i '/begin/d' ./bmd.tex
	sed -i '/end/d' ./bmd.tex
	sed -i '/sampler/d' ./bmd.tex
	cp ../plots/non-nested.pdf ./
	cp ../plots/bounds.pdf ./
	cp ../plots/performance.pdf ./
	cp ../plots/loggamma-30_x10_posterior.pdf ./
	cp ../plots/funnel-20_x1_posterior.pdf ./
	cp ../plots/rosenbrock-10_x8_posterior.pdf ./
	cp ../plots/rosenbrock-10_full_posterior.pdf ./
	cp ../plots/exoplanet_x5_posterior.pdf ./
	bibm pdflatex ms.tex

clean:
	rm -f *.log
	rm -f *.bbl
	rm -f *.bcf
	rm -f *.blg
	rm -f *.aux
	rm -f *.xml
	rm -f *.out
	rm -f *.pdf
	rm -f *.zip

