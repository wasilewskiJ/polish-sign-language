Dobra, wyniki z raportu po augmentacji wydaja sie byc ok. Jedyna anomalia jaka zauwazylem, to ze CNN nie bierze danych augmentowanych, tylko oryginalne, a podaje, ze pracuje na augmentowanych:

```


======================================================================
CNN - 5-Fold Stratified Cross-Validation
With data augmentation
======================================================================
Classes: 22
Total samples: 2618

--- Fold 1/5 ---
Train: 2094 samples (with augmentation)
Val:   524 samples (original only)
Training... ✅
Results: Acc=0.0458, F1-macro=0.0040
    Saved confusion matrix: confusion_cnn_fold1.png

--- Fold 2/5 ---
Train: 2094 samples (with augmentation)
Val:   524 samples (original only)
Training... ✅
Results: Acc=0.7443, F1-macro=0.7288
    Saved confusion matrix: confusion_cnn_fold2.png

--- Fold 3/5 ---
Train: 2094 samples (with augmentation)
Val:   524 samples (original only)
Training... ✅
Results: Acc=0.0458, F1-macro=0.0040
    Saved confusion matrix: confusion_cnn_fold3.png

--- Fold 4/5 ---
Train: 2095 samples (with augmentation)
Val:   523 samples (original only)
Training... ✅
Results: Acc=0.0459, F1-macro=0.0040
    Saved confusion matrix: confusion_cnn_fold4.png

--- Fold 5/5 ---
Train: 2095 samples (with augmentation)
Val:   523 samples (original only)
Training...

```

Ale juz mniejsza o to w tym momencie.
Prosze Cie teraz, upewnij sie ze wszystkie metody w kodzie i badania sa przeprwoadzane w dobry sposob, zgodny z literatura. Sprawdz to bardzo dokladnie, wszystkie skrypty. Ze wszystko jest ok i nie ma nigdzie jakis glupich pomylek albo oszustw.

---

Upewnij sie teraz ze kod nie ma zadnych glupich komentarzy, oraz innych oczywistych znakow wskazujacych, ze byl pisany przez AI. Jesli takowe ma, to je usun. Kod ma wygladac jak 100% czlowieczy.


---

Sprawdz co jeszcze powinno zostac dodane do repozytorium z plikow untracked, jesli cokolwiek. Czy wszystkie pliki sa w folderze repozytorium, ktore sa konieczne aby uruchomic projekt?

---

Teraz bierzemy sie za pisanie raportu na uczelnie. Raport ma byc bardzo szczegolowy i techniczno-naukowy. Zero lania wody i jakiś poematów, tylko podejście naukowe.
Raport ma byc napisany w Latexu, zgodnie z formatem Springer, ktory sie znajduje w /home/zbyszek/projects/studia/uczenie-maszynowe/projekt/experiments/sprawozdanie
Tak wiec edytuj te pliki dla mnie.

Na górze wymień autorów: Jakub Wasilewski 263852, Igor Włodarczyk 268542
Tytuł: <Wymyśl tytuł badania - coś w stylu "Klasyfikacja liter polskiego języka migowego - badania eksperymentalne na sieciach neuronowych i klasyfikatorach"
Przedmiot: Uczenie Maszynowe, Poniedziałki 17:05, opiekun mgr. Szymon Wojciechowski


Co bym chciał żeby było mniej więcej w sprawozdaniu? Zostawiam Ci co do tego wolną rękę, tak jak wspomniałem ma mieć ono charakter techniczno-naukowy. Ale to co mi przychodzi do głowy to:
* Wstęp - na czym w ogóle polegał problem, czyli o klasyfikacji liter polskeigo języka migowego 
* Przeglad literaturowy na ten temat? Jesli masz dostep do internetu i mzoesz sprawdzac prace naukowe to spoko, jak nie to postaram sie sam to uzupelnic
* Jaki dataset został użyty (czyli ręcznie zebrane zdjęcia)
* Jakie modele/klasyfikatory zostały użyte i dlaczego, do klasyfikacji liter
* Jaki był plan eksperymentów (że zestawić ze sobą różne klasyfikatory i CNN'a)
* Że została dodana augmentacja (opisać w jaki sposób)
* Jakie metryki zostaly uzyte i krotkie ich wyjasnienie
* Przedstawienie wynikow, w formie tabelek, tych macierzy pomylek itd
* Wyciagniecie wnioskow z wynikow i podsumowanie

Przypominam - to tylko moja propozycja, mozesz smialo ja dostosowac!
Mozesz wstawiac smialo snippety kodu (tylko nie za dlugie). Raport ma byc bardzo szczegolowy.
Przygotuj go tak, zebym mogl pokazac go profesorowi jako moja prace badawczą.

W raporcie podczas pisania o CNNie, napisz ze przez blad implementacyjny, nie bylo augmentacji dla CNNa (i nie wstawiaj wynikow dla CNNa).
Raport pisz po polsku i sie nie spiesz - mamy czas :-) Upewnij sie, ze Latex sie kompiluje!
Wymogiem jest, ze raport nie moze przekroczyc 10 stron...
