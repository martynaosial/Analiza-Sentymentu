import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

class ModelSentymentu:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()
        
    def wczytaj_dane(self, sciezka_do_pliku):
        print(f"Trwa wczytywanie danych z {sciezka_do_pliku}...")
        
        # wczytujemy 25 tys. negatywnych (z początku pliku)
        df_neg = pd.read_csv(
            sciezka_do_pliku, 
            encoding='latin-1', 
            header=None, 
            names=['target', 'id', 'date', 'flag', 'user', 'text'],
            nrows=800000
        )
        
        # wczytujemy 25 tys. pozytywnych (z połowy pliku - od wiersza 800,000)
        #w sumie nie jest potrzebne jak wykorzytuje wszystkie dane, ale przydatne jak
        #chcemy mniejsza ilosc
        df_pos = pd.read_csv(
            sciezka_do_pliku, 
            encoding='latin-1', 
            header=None, 
            names=['target', 'id', 'date', 'flag', 'user', 'text'],
            skiprows=800000,  # Przeskakujemy do sekcji pozytywnej
            nrows=800000
        )
        
    
        df = pd.concat([df_neg, df_pos])
        
        df = df.sample(frac=1).reset_index(drop=True)
        
        # mapowanie 0 -> Negatywny, 4 -> Pozytywny
        df['target'] = df['target'].map({0: 'Negatywny', 4: 'Pozytywny'})
        
        print(f"Wczytano {len(df)} komentarzy (pół na pół pozytywne i negatywne).")
        return df[['text', 'target']]

    def trenuj(self, dane):
        print("Rozpoczynam naukę to może chwilę potrwać")
        
        X = dane['text']
        y = dane['target']
        
        #80% do nauki, 20% do sprawdzenia 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # zamiana słów na liczby za pomoca Bag of Words
        X_train_vectors = self.vectorizer.fit_transform(X_train)
        
        self.classifier.fit(X_train_vectors, y_train)
        
        print("Model nauczony, trwa sprawdzanie dokładności...")
        
        X_test_vectors = self.vectorizer.transform(X_test)
        predykcje = self.classifier.predict(X_test_vectors)
        
        dokladnosc = accuracy_score(y_test, predykcje)
        print(f"Twój model ocenia poprawnie w {dokladnosc * 100:.2f}% przypadków.")

    def zapisz_model(self, nazwa_pliku="moj_model.pkl"):
        print(f"Zapisuję model do pliku {nazwa_pliku}...")
        with open(nazwa_pliku, 'wb') as plik:
            # zapisujemy parę wektoryzator i klasyfikator 
            pickle.dump((self.vectorizer, self.classifier), plik)
        print("Gotowe")

    def ocen_komentarz(self, tekst):
        wektor = self.vectorizer.transform([tekst])
        wynik = self.classifier.predict(wektor)
        return wynik[0]

# główna część programu

if __name__ == "__main__":
    
    moj_model = ModelSentymentu()
    
    plik_csv = "training.1600000.processed.noemoticon.csv" 
    
    try:
        dane = moj_model.wczytaj_dane(plik_csv)
        
        
        moj_model.trenuj(dane)

        #zapisuje model
        moj_model.zapisz_model()
        
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku '{plik_csv}'. Sprawdź nazwę!")