FROM python:3.12-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils \
        texlive-fonts-recommended \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-pictures \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/share/doc/* /usr/share/info/* /usr/share/man/*

COPY render_latex_png.py /usr/local/bin/render-latex-png
RUN chmod 0755 /usr/local/bin/render-latex-png \
    && pdflatex --version | head -n 1 \
    && pdftoppm -v 2>&1 | head -n 1

LABEL org.opencontainers.image.title="visual-question-discovery LaTeX/TikZ runtime" \
      org.opencontainers.image.version="0.1" \
      org.opencontainers.image.description="Bounded offline LaTeX, TikZ, PGFPlots, and PDF rasterization for Harbor"
