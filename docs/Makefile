# Makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Generate API documentation from source code
apidoc:
	@echo "Generating API documentation..."
	@python3 ../scripts/generate_docs.py

# Build HTML documentation
html: apidoc
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# Clean build directory
clean:
	@rm -rf $(BUILDDIR)/*
	@echo "Build directory cleaned."

# Clean and rebuild
rebuild: clean html

# Serve documentation locally
serve: html
	@echo "Serving documentation at http://localhost:8000"
	@cd $(BUILDDIR)/html && python3 -m http.server

# Check for broken links
linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)

# Build for GitHub Pages
github: clean apidoc
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@touch $(BUILDDIR)/html/.nojekyll
	@echo "GitHub Pages build complete."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
