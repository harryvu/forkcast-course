# Forkcast course. Students mostly run these from a Colab cell with a leading "!".

.PHONY: env check-% start-% catchup-% check-fast reference site site-check

env:                     ## print python / torch version and the device found
	python -m forkcast.env

check-%:                 ## run one lesson's definition-of-done tests, e.g. make check-00
	pytest tests/test_$*.py -q

check-fast:              ## every fast test from every lesson so far (what CI runs)
	pytest -m fast -q

start-%:                 ## copy a lesson's starter files into forkcast/ (the notebooks do this with %%writefile), e.g. make start-03
	cp -r starters/lesson$*/forkcast/. forkcast/

catchup-%:               ## copy the reference solution of a lesson into forkcast/, e.g. make catchup-00
	cp -r solutions/lesson$*/forkcast/. forkcast/
	@echo "forkcast/ now matches the Lesson $* reference solution"

reference:               ## apply every reference solution in order and run the fast tests
	for d in solutions/lesson*/; do cp -r "$$d/forkcast/." forkcast/; done
	pytest -m fast -q

site:                    ## assemble site/ into the single-file course page docs/index.html
	python tools/build_site.py

site-check:              ## verify every internal link in docs/index.html resolves
	python tools/check_site.py
