- zapoznac sie z optymalizacja niestacjonarna (albo dynamiczna)

Jest to taka optymalizacja, która polega na tym, żeby znajdować ulepszać rozwiązania zrobione.

<https://www.sintef.no/projectweb/top/vrptw/25-customers/>

<https://flatland.aicrowd.com/intro.html>

- przemyslec przymiarke do implementacji instancji problemu

- zapoznaie sie z alg mrowkowymi i podstawoymi modyfikacjami tego algorytmu (vide np. biblioteka JACOF)

<https://github.com/HaaLeo/swarmlib>
<https://elib.uni-stuttgart.de/bitstream/11682/10841/1/main-english-digital.pdf>

- przemyslec sposob implementacji samych algorytmow i testowania

Testowanie:

- używamy benchmark Solomona dla VRP (<http://web.cba.neu.edu/~msolomon/problems.htm>) dla sklustrowanych punktów i rozmiarów problemu od 25 do 100
- znajdujemy rozwiązanie początkowe, dla całego problemu za pomocą algorytmu z rodziny algorytmów local search (np. simmulated annealing, albo tabu search)

Testy:

- funkcja kosztu zależna od czasu wykonania

Przebieg testu:

1. Usunięcie losowych węzłów z wybranego zbioru danych i próba poprawienia za pomocą różnych algorytmów ACO.

Jako rozwiązanie początkowe, można użyć albo solver'a OR_TOOLS, albo predefiniowane algorytmy

W wyniku może powstać biblioteka z algorytmami ACO

---
How VRP rescheduling problems are currently solved?
How OR Tools solver works and in what way it differs from genetic algorithms

Search + optimization

How Flatland env looks like

---
<https://github.com/google/or-tools/issues/1867>

Algorytmy:

- <https://link.springer.com/chapter/10.1007/978-981-33-4305-4_22> - literature review
- <https://www.hindawi.com/journals/ddns/2018/1295485/> - Dynamic VRP with enchanced ACO
- <https://ieeexplore.ieee.org/abstract/document/8639011> - Brainstorming ACO

Problem dynamiczny - to taki, w którym w trakcie optymalizacji dochodzą nowe aspekty problemy, które należy uwzględnić w rozwiązaniu końcowym.

Plan prac:

- punkt wyjścia - biblioteka <https://github.com/HaaLeo/swarmlib> w pythonie, albo adaptacja wcześniej pisanego solvera local searchowego <https://github.com/aI-lab-glider/ai-course-local-search-solver>, który obecnie jest używany do cełów dydaktycznych, ale w miarę łatwo może być rozszczerozny do biblioteki

- zbadanie dwóch możliwych benchmarków -- środowisko [Flatland](https://flatland.aicrowd.com/intro.html) które adresuje trasowania pociągów, konkretnie vehicle re-scheduling problem (VRSP) oraz benchmark Solomona, który powinien był być zaadaptowany przez nas pod Dynamic Vehicle Routing Problem,

- przestudiowanie artykułów:

- <https://www.hindawi.com/journals/ddns/2018/1295485/> - Dynamic VRP with enchanced ACO
- <https://ieeexplore.ieee.org/abstract/document/8639011> - Brainstorming ACO

- przeprowadzenie testów w których badana będzie:

- zależność funkcji kosztu od czasu wykonania algorytmu

- adaptacyjność algorytmu do nowych warunków problemu (czyli czas potrzebny algorytmowi na powrót do wartości funkcji kosztu sprzed wprowadzenia zmian do problemu)

jako baseline, do testów może być użyta również inna metoda local search.

-----------

<https://link.springer.com/article/10.1007/s00521-020-04866-y>
