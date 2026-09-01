# =========================================================
# ETAP 1: Wczytanie i przygotowanie danych
# =========================================================
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Pobranie wbudowanego zbioru danych o nowotworach z biblioteki scikit-learn
my_data = load_breast_cancer()

# Konwersja surowej macierzy na czytelną tabelę (DataFrame) z przypisaniem nazw kolumn
data_table = pd.DataFrame(data=my_data.data, columns=my_data.feature_names)

# Dodanie kolumny z etykietami (diagnozą), czyli naszą zmienną docelową (y)
data_table['diagnosis'] = my_data.target


# =========================================================
# ETAP 2: Analiza klas i podział danych
# =========================================================
from sklearn.model_selection import train_test_split

# Sprawdzenie zbalansowania klas (liczba przypadków złośliwych i łagodnych)
diagnosis_count = data_table['diagnosis'].value_counts()

# Wyodrębnienie cech (X - pomiary) oraz etykiet (y - diagnoza) do osobnych zmiennych
X = data_table.drop('diagnosis', axis=1)
y = data_table['diagnosis']

# Podział na zbiór treningowy (80%) i testowy (20%)
# Zapewniamy równy rozkład klas w obu zbiorach używając parametru stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# =========================================================
# ETAP 3: Normalizacja danych
# =========================================================
# normalizacja danych: sprowadzenie cech o drastycznie różnych zakresach (np. powierzchnia vs gładkość)
# do wspólnej skali matematycznej (średnia 0, odchylenie 1), co ułatwia modelowi optymalizację.
from sklearn.preprocessing import StandardScaler

# Utworzenie obiektu skalera
scaler = StandardScaler()

# Obliczenie statystyk wyłącznie na zbiorze treningowym i jednoczesne jego przeskalowanie
X_train = scaler.fit_transform(X_train)

# Przeskalowanie zbioru testowego z użyciem "wyuczonych" wcześniej statystyk (zapobieganie przeciekowi danych)
X_test = scaler.transform(X_test)


# =========================================================
# ETAP 4: Inicjalizacja i trening modelu
# =========================================================
from sklearn.linear_model import LogisticRegression

# Powołanie do życia "pustego" klasyfikatora opartego na Regresji Logistycznej
my_classifier = LogisticRegression()

# Uruchomienie procesu uczenia: model optymalizuje funkcję straty na podstawie danych treningowych
my_classifier.fit(X_train, y_train)


# =========================================================
# ETAP 5: Testowanie modelu
# =========================================================
# testowanie modelu: wystawienie wyuczonego algorytmu na próbę poprzez podanie mu arkusza
# z niewidzianymi wcześniej pacjentami i zmuszenie go do samodzielnego wytypowania diagnozy.
y_predictions = my_classifier.predict(X_test)
# print(y_predictions[:5])


# =========================================================
# ETAP 6: Ewaluacja wyników
# =========================================================
from sklearn.metrics import accuracy_score, confusion_matrix

# Sprawdzenie ogólnej dokładności modelu (procent idealnie trafionych diagnoz)
test_score = accuracy_score(y_test, y_predictions)
# print(test_score)

# Wygenerowanie macierzy pomyłek, aby zdiagnozować krytyczne błędy (np. przeoczenia vs fałszywe alarmy)
matrix = confusion_matrix(y_test, y_predictions)
print(matrix)