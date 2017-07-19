FROM python:3.6.1

WORKDIR /tokenizer

COPY tokenizer/ .

RUN pip install nltk
RUN pip install matplotlib
RUN pip install NumPy
RUN pip install SciPy
RUN pip install np
RUN pip install scikit-learn
RUN python setup.py

CMD ["python", "main.py"]
