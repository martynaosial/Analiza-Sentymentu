# Analiza sentymentu komentarzy z YouTube

Program wyszukuje filmy na YouTube, pobiera komentarze spod wybranego filmu i ocenia, czy są pozytywne, czy negatywne. Klasyfikator to naiwny klasyfikator Bayesa wytrenowany na 1,6 mln tweetów ze zbioru Sentiment140.

Projekt indywidualny na przedmiocie „Programowanie w języku Python”, Uniwersytet SWPS.

*EN: Sentiment analysis of YouTube comments. The app searches YouTube, fetches comments from a chosen video and classifies English comments as positive or negative with a Multinomial Naive Bayes model trained on the Sentiment140 dataset (1.6M tweets).*

## Jak to działa

1. **Trenowanie** (`trenowanie_modelu.py`): wczytuje zbiór Sentiment140 (800 tys. tweetów negatywnych i 800 tys. pozytywnych), zamienia tekst na liczby metodą Bag of Words (`CountVectorizer`), trenuje `MultinomialNB` na 80% danych i sprawdza trafność na pozostałych 20%. Model zapisuje do `moj_model.pkl`.
2. **Analiza** (`analiza_yt.py`):
   - wyszukuje 5 filmów po haśle przez YouTube Data API v3;
   - pobiera komentarze spod wybranego filmu;
   - wybiera komentarze angielskie (`langdetect`), bo model uczył się na angielskich tweetach;
   - ocenia każdy komentarz, zlicza pozytywne i negatywne, pokazuje przykłady i wykres słupkowy (`matplotlib`).

## Technologie

Python, pandas, scikit-learn (CountVectorizer, MultinomialNB), YouTube Data API v3 (google-api-python-client), langdetect, matplotlib, pickle.

## Uruchomienie

```
pip install -r requirements.txt
```

1. **Klucz API:** w Google Cloud Console utwórz klucz do YouTube Data API v3. Skopiuj `yt_api.example.txt` jako `yt_api.txt` i wpisz do niego klucz. Możesz też ustawić zmienną środowiskową `YOUTUBE_API_KEY`. Plik `yt_api.txt` jest w `.gitignore`.
2. **Model:** rozpakuj `moj_model.zip` (powstanie `moj_model.pkl`) albo wytrenuj model od nowa: pobierz [Sentiment140 z Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140), umieść plik `training.1600000.processed.noemoticon.csv` w folderze projektu i uruchom `python trenowanie_modelu.py`.
3. Uruchom `python analiza_yt.py` i wpisz hasło po angielsku.

## Ograniczenia

- Model rozpoznaje tylko dwie klasy (pozytywny / negatywny), bez neutralnych.
- Uczył się na tweetach, a komentarze z YouTube mają trochę inny styl, więc trafność na nich może być niższa niż na zbiorze testowym.
- Analizowane są tylko komentarze po angielsku.
