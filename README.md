# Golem0_BITEhack2023
Rozwiązana hackatonu BITEhack 2023 zespołu Golem0.

Aby uruchomić aplikację lokalnie:

1. Zmień current working directory do folderu application:

            cd application

2. Upewnij się, że masz zainstalowanego Pythona 3.10 lub nowszego, oraz biblioteki:

            joblib
            pandas
            numpy
            keras
            tensorflow
            sklearn
            scikeras
            flask
            openai

3. Podmień klucz OpenAI na własny (w pliku app.py, zmienna openai.api_key) - polityka OpenAI nie pozwala na udostępnianie klucza publicznie na github.

4. Uruchom aplikację za pomocą

            python -m flask run
