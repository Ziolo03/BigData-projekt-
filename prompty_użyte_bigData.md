
## Przebieg rozmowy

W programie Cursor z modelem GPT-6.1

### 1. Przygotowanie podstawowej aplikacji

> Robię projekt na zajęcia z Big Data. Chcę w Pythonie analizować ruch samochodowy na obrazie z kamery dostępnej przez YouTube. Podaj biblioteki które pozwalają na pobierania live na yt w czasie rzeczywistym.

### 2. Odczyt obrazu z YouTube

> Źródłem ma być https://www.youtube.com/watch?v=Q71sLS8h9a4. Dodaj yt-dlp, żeby uzyskać bezpośredni adres strumienia bez pobierania całego filmu. Wybierz obraz do 720p i przeskaluj klatki do szerokości 1280 pikseli z zachowaniem proporcji.

### 3. Śledzenie samochodów

> Teraz chciałbym zaimplementować siec neuronową YOLO, według specyfikacji z ich strony najlepsza do tego będzie Yolov8 pozwalająca na prace w 30 fps. Ogranicz wykrywanie obiektów przez tą siec do jednej klasy pojazdu. Każdy obiekt ma mieć przypisywane id  oraz zieloną ramkę dokoła niego.

### 4. Zaznaczanie strefy postoju

> Chcę sam zaznaczać myszką prostokątną strefę, w której sprawdzamy postój samochodów. Zrób edytor na pierwszej klatce obrazu. Zapisuj współrzędne strefy do JSON i wczytuj je przy następnym uruchomieniu. Współrzędne zapisz jako wartości od 0 do 1, żeby dało się je przeliczyć na rozmiar obrazu. Auta które będą znajdować się w strefie bez ruchu przez więcej niż 5 sekund zaznaczaj na czerwono i zacznij naliczać punkty.

### 5. Wykrywanie postoju i naliczanie punktów

> W strefie postoju sprawdzaj, czy auto pozostaje w tym samym miejscu przez minimum 5 sekund. Dopuść tolerancję 8 pikseli, i wygładzaj położenie. Porównuj je z punktem początkowym postoju, żeby powolny ruch też mógł zakończyć postój. Za każde pełne 5 sekund naliczaj jeden umowny punkt. Pokaż przy aucie czas postoju i liczbę punktów.

### 6. Zapisywanie zdjęć i informacji o postoju

> Gdy postój przekroczy próg, zapisz wycinek zdjęcia samochodu oraz plik informacje.json. Chcę mieć ID trackera, czas postoju, punkty, początek i koniec obserwacji. Na zdjęciu dodaj podpis z ID, czasem i punktami. Aktualizuj ten sam zapis podczas trwania danego postoju, mniej więcej raz na sekundę, zamiast tworzyć nowy folder dla każdej klatki.

### 7. Zakończenie i przerwanie obserwacji

> Obsłuż sytuacje, gdy samochód ruszy, wyjedzie ze strefy albo zniknie z detekcji. Przy braku obserwacji dłuższym niż 2 sekundy zakończ jego postój. Uwzględniaj tylko czas do ostatniej obserwacji.

### 8. Wiele stref pomiarowych

> Oprócz strefy postoju potrzebuję kilku osobnych prostokątnych stref do liczenia samochodów. Rozbuduj edytor o dodawanie stref pomiarowych, zaznaczanie i usuwanie stref oraz zapis. Dodaj przyciski i skróty klawiaturowe. Strefa postoju powinna być żółta, a pomiarowe niebieskie. Każda strefa pomiarowa ma mieć własne ID i nazwę, np. Pomiar 1.

### 9. Jedno zliczenie ID na strefę i sesję

> Doprecyzuj liczenie: auto ma być policzone przy pierwszym wykryciu w strefie, nawet jeśli już w niej jest na początku analizy. To samo ID licz tylko raz w danej strefie przez całą sesję. Zniknięcie i ponowne pojawienie się tego ID nie może zwiększać licznika. Dla każdej strefy trzymaj osobny zbiór policzonych ID.

### 10. Organizacja wyników i eksport danych

> Każde uruchomienie analizy zapisuj jako osobną sesję w folderze wyniki. W sesji zapisz kopię stref, plik pojazdy.csv, podsumowanie.json i podfolder postoje. W CSV chcę czas zapisu UTC, ID i nazwę strefy, ID trackera, czas pierwszego wykrycia w strefie, stan licznika. Użyj średnika jako separatora.

### 11. Ograniczenie FPS i poprawny pomiar czasu

> Ogranicz tempo analizy i wyświetlania do maksymalnie 15 FPS bez celowego pomijania klatek. Czas postoju powinien być liczony według materiału, żeby wolniejsze przetwarzanie nie zawyżało wyniku. Korzystaj ze znaczników czasu klatek, a gdy są niedostępne, z FPS źródła. Jeśli FPS jest niepoprawny, przyjmij 25. Obsłuż też cofnięcie albo duży skok znacznika czasu, kończąc aktualne postoje.

### 12. Edycja stref w trakcie działania i porządkowanie kodu

> Dodaj klawisz E, który kończy bieżącą sesję i wraca do edytora stref, a potem uruchamia nową sesję z nowym trackerem i licznikami. Klawisz Q ma kończyć program.

