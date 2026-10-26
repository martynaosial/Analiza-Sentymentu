import pickle
import os
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langdetect import detect

# wczytanie klucza: najpierw ze zmiennej środowiskowej YOUTUBE_API_KEY, potem z pliku yt_api.txt
# (plik yt_api.txt jest w .gitignore, więc klucz nie trafia do repozytorium)
API_KEY = os.environ.get("YOUTUBE_API_KEY")
if not API_KEY:
    try:
        with open("yt_api.txt", "r") as f:
            API_KEY = f.read().strip()
    except FileNotFoundError:
        print("BŁĄD: Nie znaleziono pliku yt_api.txt! Skopiuj yt_api.example.txt jako yt_api.txt i wpisz swój klucz.")
        exit()

# wczytanie modelu z pliku pkl przez pickle
try:
    with open("moj_model.pkl", "rb") as f:
        vectorizer, classifier = pickle.load(f)
except FileNotFoundError:
    print("BŁĄD: Nie znaleziono pliku moj_model.pkl! Rozpakuj moj_model.zip albo uruchom trenowanie_modelu.py.")
    exit()

youtube = build("youtube", "v3", developerKey=API_KEY)

def start():
    # wyszukiwanie po słowie
    slowo = input("Wpisz czego szukasz na YT (hasło po angielsku): ")
    search_request = youtube.search().list(
        part="snippet",
        q=slowo,
        type="video",
        maxResults=5
    )
    search_response = search_request.execute()

    filmy = []
    print("\n ZNALEZIONE FILMY:")
    for i, item in enumerate(search_response.get("items", [])):
        v_id = item["id"]["videoId"]
        tytul = item["snippet"]["title"]
        filmy.append((v_id, tytul))
        print(f"{i+1}. {tytul} (ID: {v_id})")

    # wybór filmu z listy 
    try:
        nr = int(input("\nWybierz numer filmu do analizy sentymentu: ")) - 1
        wybrane_id = filmy[nr][0]
        wybrany_tytul = filmy[nr][1]
    except (ValueError, IndexError):
        print("Niepoprawny wybór.")
        return

    # pobieranie komentarzy
    print(f"\nAnalizuję film: {wybrany_tytul}...")
    
    try:
        comment_request = youtube.commentThreads().list(
            part="snippet",
            videoId=wybrane_id,
            maxResults=600,
            textFormat="plainText"
        )
        comment_response = comment_request.execute()
    except HttpError as e:
        # błąd wyłączonych komentarzy
        if "commentsDisabled" in str(e):
            print("\n[!] BŁĄD: Ten film ma wyłączone komentarze. Wybierz inne wideo!")
        else:
            print(f"\n[!] Błąd API: {e}")
        return

 # listy
    staty = {"Pozytywny": 0, "Negatywny": 0}
    przyklad_poz = ""
    przyklad_neg = ""
    licznik_eng = 0

    # tu się rozpoczyna analiza sentymentu
    for item in comment_response.get("items", []):
        tekst = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        
        try:
            # wybór tylko angielskich komentarzy
            if detect(tekst) == 'en':
                # wektoryzacja i ocena przez model (zamiana tekstu na liczby)
                wektor = vectorizer.transform([tekst])
                ocena = classifier.predict(wektor)[0]
                
                # zliczanie wyników 
                if ocena == 'Pozytywny':
                    staty["Pozytywny"] += 1
                    if przyklad_poz == "":
                        przyklad_poz = tekst
                else:
                    staty["Negatywny"] += 1
                    if przyklad_neg == "":
                        przyklad_neg = tekst
                
                licznik_eng += 1
        except:
            continue

    # wyniki
    print(f"\nPrzeanalizowano {licznik_eng} angielskich komentarzy.")
    print(f"Statystyki: {staty}")

    # wyświetlanie przykładowych komentarzy
    print("\nPRZYKŁADOWE KOMENTARZE:")
    
    print(f"[POZYTYWNY] {przyklad_poz[:200]}" if przyklad_poz else "[POZYTYWNY] Brak") 

    print(f"[NEGATYWNY] {przyklad_neg[:200]}" if przyklad_neg else "[NEGATYWNY] Brak")

    if licznik_eng > 0:
        # tworzenie wykresu
        plt.bar(staty.keys(), staty.values(), color=['green', 'red'])
        plt.title(f"Analiza sentymentu dla: {wybrany_tytul[:30]}...")
        plt.ylabel("Liczba komentarzy")
        plt.show()
    else:
        print("Brak angielskich komentarzy pod tym filmem.")

if __name__ == "__main__":
    start()