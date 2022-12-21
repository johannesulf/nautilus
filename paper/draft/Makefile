all:
	sed -i '/begin/d' ./evidence.tex
	sed -i '/end/d' ./evidence.tex
	sed -i '/sampler/d' ./evidence.tex
	sed -i '/begin/d' ./bmd.tex
	sed -i '/end/d' ./bmd.tex
	sed -i '/sampler/d' ./bmd.tex
	cp ../figures/non-nested.pdf ./
	cp ../figures/tessellation.pdf ./
	cp ../figures/posterior.pdf ./
	cp ../figures/bounds.pdf ./
	cp ../figures/performance.pdf ./
	cp ../figures/loggamma-30_x10_posterior.pdf ./
	cp ../figures/funnel-20_x1_posterior.pdf ./
	cp ../figures/rosenbrock-10_x8_posterior.pdf ./
	cp ../figures/rosenbrock-10_full_posterior.pdf ./
	cp ../figures/exoplanet_x5_posterior.pdf ./
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

