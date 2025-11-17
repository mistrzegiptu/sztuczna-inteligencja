# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: PSI
#     language: python
#     name: psi
# ---

# %% [markdown]
# # Sieci neuronowe

# %% [markdown]
# ## Wstęp
#
# Celem laboratorium jest zapoznanie się z podstawami sieci neuronowych oraz uczeniem głębokim (*deep learning*). Zapoznasz się na nim z następującymi tematami:
# - treningiem prostych sieci neuronowych, w szczególności z:
#   - regresją liniową w sieciach neuronowych
#   - optymalizacją funkcji kosztu
#   - algorytmem spadku wzdłuż gradientu
#   - siecią typu Multilayer Perceptron (MLP)
# - frameworkiem PyTorch, w szczególności z:
#   - ładowaniem danych
#   - preprocessingiem danych
#   - pisaniem pętli treningowej i walidacyjnej
#   - walidacją modeli
# - architekturą i hiperaprametrami sieci MLP, w szczególności z:
#   - warstwami gęstymi (w pełni połączonymi)
#   - funkcjami aktywacji
#   - regularyzacją: L2, dropout

# %% [markdown]
# ## Wykorzystywane biblioteki
#
# Zaczniemy od pisania ręcznie prostych sieci w bibliotece Numpy, służącej do obliczeń numerycznych na CPU. Później przejdziemy do wykorzystywania frameworka PyTorch, służącego do obliczeń numerycznych na CPU, GPU oraz automatycznego różniczkowania, wykorzystywanego głównie do treningu sieci neuronowych.
#
# Wykorzystamy PyTorcha ze względu na popularność, łatwość instalacji i użycia, oraz dużą kontrolę nad niskopoziomowymi aspektami budowy i treningu sieci neuronowych. Framework ten został stworzony do zastosowań badawczych i naukowych, ale ze względu na wygodę użycia stał się bardzo popularny także w przemyśle. W szczególności całkowicie zdominował przetwarzanie języka naturalnego (NLP) oraz uczenie na grafach.
#
# Pierwszy duży framework do deep learningu, oraz obecnie najpopularniejszy, to TensorFlow, wraz z wysokopoziomową nakładką Keras. Są jednak szanse, że Google (autorzy) będzie go powoli porzucać na rzecz ich nowego frameworka JAX ([dyskusja](https://www.reddit.com/r/MachineLearning/comments/vfl57t/d_google_quietly_moving_its_products_from/), [artykuł Business Insidera](https://www.businessinsider.com/facebook-pytorch-beat-google-tensorflow-jax-meta-ai-2022-6?IR=T)), który jest bardzo świeżym, ale ciekawym narzędziem.
#
# Trzecia, ale znacznie mniej popularna od powyższych opcja to Apache MXNet.

# %% [markdown]
# ## Konfiguracja własnego komputera
#
# Zainstaluj zależności z pomocą `uv` lub `pip`. Jeżeli korzystasz z `pip`, to skorzystaj z odpowiedniej komendy (CPU lub NVidia GPU) [ze strony PyTorcha](https://pytorch.org/get-started/locally/). Wykorzystanie GPU nie jest potrzebne, ale nieco przyspieszy niektóre obliczenia.

# %%
# # for conda users
# # !conda install -y matplotlib pandas pytorch torchvision -c pytorch -c conda-forge

# %% [markdown] id="Othm3C2lLAsj"
# ## Wprowadzenie
#
# Zanim zaczniemy naszą przygodę z sieciami neuronowymi, przyjrzyjmy się prostemu przykładowi regresji liniowej na syntetycznych danych:

# %% id="rnJsfxbnLAsj"
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="EaYpEXzBLAsl" outputId="2f8d2922-72f0-4d38-8548-d1262adf522e"
np.random.seed(0)

x = np.linspace(0, 1, 100)
y = x + np.random.normal(scale=0.1, size=x.shape)

plt.scatter(x, y)


# %% [markdown] editable=true id="PEM_-yKELAsl" slideshow={"slide_type": ""}
# W przeciwieństwie do laboratorium 1, tym razem będziemy chcieli rozwiązać ten problem własnoręcznie, bez użycia wysokopoziomowego interfejsu Scikit-learn'a. W tym celu musimy sobie przypomnieć sformułowanie naszego **problemu optymalizacyjnego (optimization problem)**.
#
# W przypadku prostej regresji liniowej (1 zmienna) mamy model postaci $\hat{y} = \alpha x + \beta$, z dwoma parametrami, których będziemy się uczyć. Miarą niedopasowania modelu o danych parametrach jest **funkcja kosztu (cost function)**, nazywana też funkcją celu. Najczęściej używa się **błędu średniokwadratowego (mean squared error, MSE)**:
# $$\large
# MSE = \frac{1}{N} \sum_{i}^{N} (y - \hat{y})^2
# $$
#
# Od jakich $\alpha$ i $\beta$ zacząć? W najprostszym wypadku wystarczy po prostu je wylosować jako niewielkie liczby zmiennoprzecinkowe.

# %% [markdown] editable=true id="PEM_-yKELAsl" slideshow={"slide_type": ""} tags=["ex"]
# ### Zadanie 1 (0.5 punkt)
#
# Uzupełnij kod funkcji `mse`, obliczającej błąd średniokwadratowy. Wykorzystaj Numpy'a w celu wektoryzacji obliczeń dla wydajności.

# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="RaA7Q46TLAsm" outputId="5c57fe58-1934-4d21-9a7b-d14e9a23140b" slideshow={"slide_type": ""} tags=["ex"]
def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    # implement me!
    # your_code
    return np.sum(np.power(y - y_hat, 2)) / len(y)



# %% colab={"base_uri": "https://localhost:8080/", "height": 282} editable=true id="qSGfamGbLAsm" outputId="733ce15f-ae75-466b-e7ee-eb3c9d9d534c" slideshow={"slide_type": ""} tags=["ex"]
a = np.random.rand()
b = np.random.rand()
print(f"MSE: {mse(y, a * x + b):.3f}")

plt.scatter(x, y)
plt.plot(x, a * x + b, color="g", linewidth=4)


# %% [markdown] id="Y--E9Mp9LAsn"
# Losowe parametry radzą sobie nie najlepiej. Jak lepiej dopasować naszą prostą do danych? Zawsze możemy starać się wyprowadzić rozwiązanie analitycznie, i w tym wypadku nawet nam się uda. Jest to jednak szczególny i dość rzadki przypadek, a w szczególności nie będzie to możliwe w większych sieciach neuronowych.
#
# Potrzebna nam będzie **metoda optymalizacji (optimization method)**, dającą wartości parametrów minimalizujące dowolną różniczkowalną funkcję kosztu. Zdecydowanie najpopularniejszy jest tutaj **spadek wzdłuż gradientu (gradient descent)**.
#
# Metoda ta wywodzi się z prostych obserwacji, które tutaj przedstawimy. Bardziej szczegółowe rozwinięcie dla zainteresowanych: [sekcja 4.3 "Deep Learning Book"](https://www.deeplearningbook.org/contents/numerical.html), [ten praktyczny kurs](https://cs231n.github.io/optimization-1/), [analiza oryginalnej publikacji Cauchy'ego](https://www.math.uni-bielefeld.de/documenta/vol-ismp/40_lemarechal-claude.pdf) (oryginał w języku francuskim).
#
# Pochodna jest dokładnie równa granicy funkcji. Dla małego $\epsilon$ można ją przybliżyć jako:
# $$\large
# \frac{f(x)}{dx} \approx \frac{f(x+\epsilon) - f(x)}{\epsilon}
# $$
#
# Przyglądając się temu równaniu widzimy, że: 
# * dla funkcji rosnącej ($f(x+\epsilon) > f(x)$) wyrażenie $\frac{f(x)}{dx}$ będzie miało znak dodatni 
# * dla funkcji malejącej ($f(x+\epsilon) < f(x)$) wyrażenie $\frac{f(x)}{dx}$ będzie miało znak ujemny 
#
# Widzimy więc, że potrafimy wskazać kierunek zmniejszenia wartości funkcji, patrząc na znak pochodnej. Zaobserwowano także, że amplituda wartości w $\frac{f(x)}{dx}$ jest tym większa, im dalej jesteśmy od minimum (maximum). Pochodna wyznacza więc, w jakim kierunku funkcja najszybciej rośnie, zaś przeciwny zwrot to ten, w którym funkcja najszybciej spada.
#
# Stosując powyższe do optymalizacji, mamy:
# $$\large
# x_{t+1} = x_{t} -  \alpha * \frac{f(x)}{dx}
# $$
#
# $\alpha$ to niewielka wartość (rzędu zwykle $10^{-5}$ - $10^{-2}$), wprowadzona, aby trzymać się założenia o małej zmianie parametrów ($\epsilon$). Nazywa się ją **stałą uczącą (learning rate)** i jest zwykle najważniejszym hiperparametrem podczas nauki sieci.
#
# Metoda ta zakłada, że używamy całego zbioru danych do aktualizacji parametrów w każdym kroku, co nazywa się po prostu GD (od *gradient descent*) albo *full batch GD*. Wtedy każdy krok optymalizacji nazywa się **epoką (epoch)**.
#
# Im większa stała ucząca, tym większe nasze kroki podczas minimalizacji. Możemy więc uczyć szybciej, ale istnieje ryzyko, że będziemy "przeskakiwać" minima. Mniejsza stała ucząca to wolniejszy, ale dokładniejszy trening. Jednak nie zawsze ona pozwala osiągnąć lepsze wyniki, bo może okazać się, że utkniemy w minimum lokalnym. Można także zmieniać stałą uczącą podczas treningu, co nazywa się **learning rate scheduling (LR scheduling)**. Obrazowo:
#
# ![learning_rate](http://www.bdhammel.com/assets/learning-rate/lr-types.png)

# %% [markdown] id="496qEjkVLAso"
# ![interactive LR](http://cdn-images-1.medium.com/max/640/1*eeIvlwkMNG1wSmj3FR6M2g.gif)

# %% [markdown] id="RYkyAHKzLAsp"
# Policzmy więc pochodną dla naszej funkcji kosztu MSE. Pochodną liczymy po parametrach naszego modelu, bo to właśnie ich chcemy dopasować tak, żeby koszt był jak najmniejszy:
#
# $$\large
# MSE = \frac{1}{N} \sum_{i}^{N} (y_i - \hat{y_i})^2
# $$
#
# W powyższym wzorze tylko $y_i$ jest zależny od $a$ oraz $b$. Możemy wykorzystać tu regułę łańcuchową (*chain rule*) i policzyć pochodne po naszych parametrach w sposób następujący:
#
# $$\large
# \frac{\text{d} MSE}{\text{d} a} = \frac{1}{N} \sum_{i}^{N} \frac{\text{d} (y_i - \hat{y_i})^2}{\text{d} \hat{y_i}} \frac{\text{d} \hat{y_i}}{\text{d} a}
# $$
#
# $$\large
# \frac{\text{d} MSE}{\text{d} b} = \frac{1}{N} \sum_{i}^{N} \frac{\text{d} (y_i - \hat{y_i})^2}{\text{d} \hat{y_i}} \frac{\text{d} \hat{y_i}}{\text{d} b}
# $$
#
# Policzmy te pochodne po kolei:
#
# $$\large
# \frac{\text{d} (y_i - \hat{y_i})^2}{\text{d} \hat{y_i}} = -2 \cdot (y_i - \hat{y_i})
# $$
#
# $$\large
# \frac{\text{d} \hat{y_i}}{\text{d} a} = x_i
# $$
#
# $$\large
# \frac{\text{d} \hat{y_i}}{\text{d} b} = 1
# $$
#
# Łącząc powyższe wyniki dostaniemy:
#
# $$\large
# \frac{\text{d} MSE}{\text{d} a} = \frac{-2}{N} \sum_{i}^{N} (y_i - \hat{y_i}) \cdot {x_i}
# $$
#
# $$\large
# \frac{\text{d} MSE}{\text{d} b} = \frac{-2}{N} \sum_{i}^{N} (y_i - \hat{y_i})
# $$
#
# Aktualizacja parametrów wygląda tak:
#
# $$\large
# a' = a - \alpha * \left( \frac{-2}{N} \sum_{i=1}^N (y_i - \hat{y}_i) \cdot x_i \right)
# $$
# $$\large
# b' = b - \alpha * \left( \frac{-2}{N} \sum_{i=1}^N (y_i - \hat{y}_i) \right)
# $$
#
# Liczymy więc pochodną funkcji kosztu, a potem za pomocą reguły łańcuchowej "cofamy się", dochodząc do tego, jak każdy z parametrów wpływa na błąd i w jaki sposób powinniśmy go zmienić. Nazywa się to **propagacją wsteczną (backpropagation)** i jest podstawowym mechanizmem umożliwiającym naukę sieci neuronowych za pomocą spadku wzdłuż gradientu. Więcej możesz o tym przeczytać [tutaj](https://cs231n.github.io/optimization-2/).

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# ### Zadanie 2 (1.0 punkt)
#
# Zaimplementuj funkcję realizującą jedną epokę treningową. Zauważ, że `x` oraz `y` są wektorami. Oblicz predykcję przy aktualnych parametrach oraz zaktualizuj je zgodnie z powyższymi wzorami.

# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="4qbdWOSULAsp" outputId="055607ae-87aa-470a-e6da-25682c82d470" slideshow={"slide_type": ""} tags=["ex"]
def optimize(
    x: np.ndarray, y: np.ndarray, a: float, b: float, learning_rate: float = 0.1
):
    y_hat = a * x + b
    errors = y - y_hat
    # implement me!
    # your_code
    n = len(y)
    
    a_prim = a - learning_rate * (-2) * np.sum(errors * x) / n
    b_prim = b - learning_rate * (-2) * np.sum(errors) / n

    return a_prim, b_prim



# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
for i in range(1000):
    loss = mse(y, a * x + b)
    a, b = optimize(x, y, a, b)
    if i % 100 == 0:
        print(f"step {i} loss: ", loss)

print("final loss:", loss)

# %% tags=["ex"]
assert 0.0 < loss < 2.0

print("Solution is correct!")

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} editable=true id="xOgRcPC1LAsq" outputId="85b0b3e4-aa0d-467a-d8ff-5f01be17b243" slideshow={"slide_type": ""} tags=["ex"]
plt.scatter(x, y)
plt.plot(x, a * x + b, color="g", linewidth=4)

# %% [markdown] id="vOr2fWYpLAsq"
# Udało ci się wytrenować swoją pierwszą sieć neuronową. Czemu? Otóż neuron to po prostu wektor parametrów, a zwykle robimy iloczyn skalarny tych parametrów z wejściem. Dodatkowo na wyjście nakłada się **funkcję aktywacji (activation function)**, która przekształca wyjście. Tutaj takiej nie było, a właściwie była to po prostu funkcja identyczności.
#
# Oczywiście w praktyce korzystamy z odpowiedniego frameworka, który w szczególności:
# - ułatwia budowanie sieci, np. ma gotowe klasy dla warstw neuronów
# - ma zaimplementowane funkcje kosztu oraz ich pochodne
# - sam różniczkuje ze względu na odpowiednie parametry i aktualizuje je odpowiednio podczas treningu
#

# %% [markdown] id="NJBYJabuLAsr"
# ## Wprowadzenie do PyTorcha

# %% [markdown] id="EB-99XqhLAsr"
# PyTorch to w gruncie rzeczy narzędzie do algebry liniowej z [automatycznym rożniczkowaniem](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html), z możliwością przyspieszenia obliczeń z pomocą GPU. Na tych fundamentach zbudowany jest pełny framework do uczenia głębokiego. Można spotkać się ze stwierdzenie, że PyTorch to NumPy + GPU + opcjonalne różniczkowanie, co jest całkiem celne. Plus można łatwo debugować printem :)
#
# PyTorch używa dynamicznego grafu obliczeń, który sami definiujemy w kodzie. Takie podejście jest bardzo wygodne, elastyczne i pozwala na łatwe eksperymentowanie. Odbywa się to potencjalnie kosztem wydajności, ponieważ pozostawia kwestię optymalizacji programiście. Więcej na ten temat dla zainteresowanych na końcu laboratorium.
#
# Samo API PyTorcha bardzo przypomina Numpy'a, a podstawowym obiektem jest `Tensor`, klasa reprezentująca tensory dowolnego wymiaru. Dodatkowo niektóre tensory będą miały automatycznie obliczony gradient. Co ważne, tensor jest na pewnym urządzeniu, CPU lub GPU, a przenosić między nimi trzeba explicite.
#
# Najważniejsze moduły:
# - `torch` - podstawowe klasy oraz funkcje, np. `Tensor`, `from_numpy()`
# - `torch.nn` - klasy związane z sieciami neuronowymi, np. `Linear`, `Sigmoid`
# - `torch.optim` - wszystko związane z optymalizacją, głównie spadkiem wzdłuż gradientu

# %% id="FwuIt8S-LAss"
import torch
import torch.nn as nn
import torch.optim as optim

# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="bfCiUFXULAss" outputId="83f6231d-ecc4-461a-b758-fdc4bc2a88a4" slideshow={"slide_type": ""}
ones = torch.ones(10)
noise = torch.ones(10) * torch.rand(10)

# elementwise sum
print(ones + noise)

# elementwise multiplication
print(ones * noise)

# dot product
print(ones @ noise)

# %% editable=true slideshow={"slide_type": ""}
type(x)

# %% editable=true id="ynNd_kD0LAst" slideshow={"slide_type": ""}
# beware - shares memory with original Numpy array!
# very fast, but modifications are visible to original variable
x = torch.from_numpy(x)
y = torch.from_numpy(y)

# %% [markdown] editable=true id="W9kkxczELAsu" slideshow={"slide_type": ""}
# Jeżeli dla stworzonych przez nas tensorów chcemy śledzić operacje i obliczać gradient, to musimy oznaczyć `requires_grad=True`.

# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="8HtZL-KfLAsu" outputId="47c6d930-5678-452a-95bc-227935138b40" slideshow={"slide_type": ""}
a = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)
a, b

# %% [markdown] editable=true id="Nl1guWZ_LAsv" slideshow={"slide_type": ""}
# PyTorch zawiera większość powszechnie używanych funkcji kosztu, np. MSE. Mogą być one używane na 2 sposoby, z czego pierwszy jest popularniejszy:
# - jako klasy wywoływalne z modułu `torch.nn`
# - jako funkcje z modułu `torch.nn.functional`
#
# Po wykonaniu poniższego kodu widzimy, że zwraca on nam tensor z dodatkowymi atrybutami. Co ważne, jest to skalar (0-wymiarowy tensor), bo potrzebujemy zwyczajnej liczby do obliczania propagacji wstecznych (pochodnych czątkowych).

# %% editable=true slideshow={"slide_type": ""}
mse = nn.MSELoss()
mse(y, a * x + b)

# %% [markdown] editable=true id="vS35r49nLAsw" slideshow={"slide_type": ""}
# Atrybutu `grad_fn` nie używamy wprost, bo korzysta z niego w środku PyTorch, ale widać, że tensor jest "świadomy", że liczy się na nim pochodną. Możemy natomiast skorzystać z atrybutu `grad`, który zawiera faktyczny gradient. Zanim go jednak dostaniemy, to trzeba powiedzieć PyTorchowi, żeby policzył gradient. Służy do tego metoda `.backward()`, wywoływana na obiekcie zwracanym przez funkcję kosztu.

# %% editable=true id="Qb7l6Xg1LAsx" slideshow={"slide_type": ""}
loss = mse(y, a * x + b)
loss.backward()

# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="6LfQbLVoLAsx" outputId="d5b87fb7-d284-423c-f467-b677384b2f67" slideshow={"slide_type": ""}
print(a.grad)

# %% [markdown] editable=true id="Kdf1iweELAsy" slideshow={"slide_type": ""}
# Ważne jest, że PyTorch nie liczy za każdym razem nowego gradientu, tylko dodaje go do istniejącego, czyli go akumuluje. Jest to przydatne w niektórych sieciach neuronowych, ale zazwyczaj trzeba go zerować. Jeżeli tego nie zrobimy, to dostaniemy coraz większe gradienty.
#
# Do zerowania służy metoda `.zero_()`. W PyTorchu wszystkie metody modyfikujące tensor w miejscu mają `_` na końcu nazwy. Jest to dość niskopoziomowa operacja dla pojedynczych tensorów - zobaczymy za chwilę, jak to robić łatwiej dla całej sieci.

# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="DiCQZKJsLAsy" outputId="2f779622-480d-43fc-b9d0-a0e36ff4b28b" slideshow={"slide_type": ""}
loss = mse(y, a * x + b)
loss.backward()
a.grad

# %% [markdown] editable=true id="xNC3Ag8uLAsz" slideshow={"slide_type": ""}
# Zobaczmy, jak wyglądałaby regresja liniowa, ale napisana w PyTorchu. Jest to oczywiście bardzo niskopoziomowa implementacja - za chwilę zobaczymy, jak to wygląda w praktyce.

# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="AKnxyeboLAsz" outputId="2f939474-901a-4773-9704-686a40ae6e8e" slideshow={"slide_type": ""}
learning_rate = 0.1
for i in range(1000):
    loss = mse(y, a * x + b)

    # compute gradients
    loss.backward()

    # update parameters
    a.data -= learning_rate * a.grad
    b.data -= learning_rate * b.grad

    # zero gradients
    a.grad.data.zero_()
    b.grad.data.zero_()

    if i % 100 == 0:
        print(f"step {i} loss: ", loss)

print("final loss:", loss)

# %% [markdown] editable=true id="2DXNVhshmmI-" slideshow={"slide_type": ""}
# Trening modeli w PyTorchu jest dosyć schematyczny i najczęściej rozdziela się go na kilka bloków, dających razem **pętlę uczącą (training loop)**, powtarzaną w każdej epoce:
# 1. Forward pass - obliczenie predykcji sieci
# 2. Loss calculation
# 3. Backpropagation - obliczenie pochodnych oraz zerowanie gradientów
# 4. Optimalization - aktualizacja wag
# 5. Other - ewaluacja na zbiorze walidacyjnym, logging etc.

# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="2etpw7TNLAs0" outputId="8ac35c12-6c70-41ec-bf57-414456fc3c96" slideshow={"slide_type": ""}
# initialization
learning_rate = 0.1
a = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)
optimizer = torch.optim.SGD([a, b], lr=learning_rate)
best_loss = float("inf")

# training loop in each epoch
for i in range(1000):
    # forward pass
    y_hat = a * x + b

    # loss calculation
    loss = mse(y, y_hat)

    # backpropagation
    loss.backward()

    # optimization
    optimizer.step()
    optimizer.zero_grad()  # zeroes all gradients - very convenient!

    if i % 100 == 0:
        if loss < best_loss:
            best_model = (a.clone(), b.clone())
            best_loss = loss
        print(f"step {i} loss: {loss.item():.4f}")

print("final loss:", loss)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Przejdziemy teraz do budowy sieci neuronowej do klasyfikacji. Typowo implementuje się ją po prostu jako sieć dla regresji, ale zwracającą tyle wyników, ile mamy klas, a potem aplikuje się na tym funkcję sigmoidalną (2 klasy) lub softmax (>2 klasy). W przypadku klasyfikacji binarnej zwraca się czasem tylko 1 wartość, przepuszczaną przez sigmoidę - wtedy wyjście z sieci to prawdopodobieństwo klasy pozytywnej.
#
# Funkcją kosztu zwykle jest **entropia krzyżowa (cross-entropy)**, stosowana też w klasycznej regresji logistycznej. Co ważne, sieci neuronowe, nawet tak proste, uczą się szybciej i stabilniej, gdy dane na wejściu (a przynajmniej zmienne numeryczne) są **ustandaryzowane (standardized)**. Operacja ta polega na odjęciu średniej i podzieleniu przez odchylenie standardowe (tzw. *Z-score transformation*).
#
# **Uwaga - PyTorch wymaga tensora klas będącego liczbami zmiennoprzecinkowymi!**

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Zbiór danych

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Na tym laboratorium wykorzystamy zbiór [Adult Census](https://archive.ics.uci.edu/ml/datasets/adult). Dotyczy on przewidywania na podstawie danych demograficznych, czy dany człowiek zarabia powyżej 50 tysięcy dolarów rocznie, czy też mniej. Jest to cenna informacja np. przy planowaniu kampanii marketingowych. Jak możesz się domyślić, zbiór pochodzi z czasów, kiedy inflacja była dużo niższa :)
#
# Poniżej znajduje się kod do ściągnięcia i preprocessingu zbioru. Nie musisz go dokładnie analizować.

# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="4DNsaZAnLAs0" outputId="70822008-530d-4173-deb9-8149a9fe5b41" slideshow={"slide_type": ""}
# #!wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

# %% editable=true slideshow={"slide_type": ""}
import pandas as pd


columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "wage"
]

"""
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
"""

df = pd.read_csv("adult.data", header=None, names=columns)
df.wage.unique()

# %% editable=true slideshow={"slide_type": ""}
# attribution: https://www.kaggle.com/code/royshih23/topic7-classification-in-python
df['education'].replace('Preschool', 'dropout',inplace=True)
df['education'].replace('10th', 'dropout',inplace=True)
df['education'].replace('11th', 'dropout',inplace=True)
df['education'].replace('12th', 'dropout',inplace=True)
df['education'].replace('1st-4th', 'dropout',inplace=True)
df['education'].replace('5th-6th', 'dropout',inplace=True)
df['education'].replace('7th-8th', 'dropout',inplace=True)
df['education'].replace('9th', 'dropout',inplace=True)
df['education'].replace('HS-Grad', 'HighGrad',inplace=True)
df['education'].replace('HS-grad', 'HighGrad',inplace=True)
df['education'].replace('Some-college', 'CommunityCollege',inplace=True)
df['education'].replace('Assoc-acdm', 'CommunityCollege',inplace=True)
df['education'].replace('Assoc-voc', 'CommunityCollege',inplace=True)
df['education'].replace('Bachelors', 'Bachelors',inplace=True)
df['education'].replace('Masters', 'Masters',inplace=True)
df['education'].replace('Prof-school', 'Masters',inplace=True)
df['education'].replace('Doctorate', 'Doctorate',inplace=True)

df['marital-status'].replace('Never-married', 'NotMarried',inplace=True)
df['marital-status'].replace(['Married-AF-spouse'], 'Married',inplace=True)
df['marital-status'].replace(['Married-civ-spouse'], 'Married',inplace=True)
df['marital-status'].replace(['Married-spouse-absent'], 'NotMarried',inplace=True)
df['marital-status'].replace(['Separated'], 'Separated',inplace=True)
df['marital-status'].replace(['Divorced'], 'Separated',inplace=True)
df['marital-status'].replace(['Widowed'], 'Widowed',inplace=True)

# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="LiOxs_6mLAs1" outputId="c95418cf-2632-41d0-de0a-9caf109de113" slideshow={"slide_type": ""}
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


X = df.copy()
y = (X.pop("wage") == ' >50K').astype(int).values

train_valid_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=train_valid_size, 
    random_state=0, 
    shuffle=True, 
    stratify=y
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, 
    test_size=train_valid_size, 
    random_state=0, 
    shuffle=True, 
    stratify=y_train
)

continuous_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
continuous_X_train = X_train[continuous_cols]
categorical_X_train = X_train.loc[:, ~X_train.columns.isin(continuous_cols)]

continuous_X_valid = X_valid[continuous_cols]
categorical_X_valid = X_valid.loc[:, ~X_valid.columns.isin(continuous_cols)]

continuous_X_test = X_test[continuous_cols]
categorical_X_test = X_test.loc[:, ~X_test.columns.isin(continuous_cols)]

categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
continuous_scaler = StandardScaler() #MinMaxScaler(feature_range=(-1, 1))

categorical_encoder.fit(categorical_X_train)
continuous_scaler.fit(continuous_X_train)

continuous_X_train = continuous_scaler.transform(continuous_X_train)
continuous_X_valid = continuous_scaler.transform(continuous_X_valid)
continuous_X_test = continuous_scaler.transform(continuous_X_test)

categorical_X_train = categorical_encoder.transform(categorical_X_train)
categorical_X_valid = categorical_encoder.transform(categorical_X_valid)
categorical_X_test = categorical_encoder.transform(categorical_X_test)

X_train = np.concatenate([continuous_X_train, categorical_X_train], axis=1)
X_valid = np.concatenate([continuous_X_valid, categorical_X_valid], axis=1)
X_test = np.concatenate([continuous_X_test, categorical_X_test], axis=1)

X_train.shape, y_train.shape

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Uwaga co do typów - PyTorchu wszystko w sieci neuronowej musi być typu `float32`. W szczególności trzeba uważać na konwersje z Numpy'a, który używa domyślnie typu `float64`. Może ci się przydać metoda `.float()`.
#
# Uwaga co do kształtów wyjścia - wejścia do `nn.BCELoss` muszą być tego samego kształtu. Może ci się przydać metoda `.squeeze()` lub `.unsqueeze()`.

# %% id="qfRA3xEoLAs1"
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float().unsqueeze(-1)

X_valid = torch.from_numpy(X_valid).float()
y_valid = torch.from_numpy(y_valid).float().unsqueeze(-1)

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float().unsqueeze(-1)

# %% [markdown]
# Podobnie jak w laboratorium 2, mamy tu do czynienia z klasyfikacją niezbalansowaną:

# %%
import matplotlib.pyplot as plt

y_pos_perc = 100 * y_train.sum().item() / len(y_train)
y_neg_perc = 100 - y_pos_perc

plt.title("Class percentages")
plt.bar(["<50k", ">=50k"], [y_neg_perc, y_pos_perc])
plt.show()

# %% [markdown]
# W związku z powyższym będziemy używać odpowiednich metryk, czyli AUROC, precyzji i czułości.

# %% [markdown] id="XLexWff-LAs0" tags=["ex"]
# ### Zadanie 3 (1.0 punkt)
#
# Zaimplementuj regresję logistyczną dla tego zbioru danych, używając PyTorcha. Dane wejściowe zostały dla ciebie przygotowane w komórkach poniżej.
#
# Sama sieć składa się z 2 elementów:
# - warstwa liniowa `nn.Linear`, przekształcająca wektor wejściowy na 1 wyjście - logit
# - aktywacja sigmoidalna `nn.Sigmoid`, przekształcająca logit na prawdopodobieństwo klasy pozytywnej
#
# Użyj binarnej entropii krzyżowej `nn.BCELoss` jako funkcji kosztu. Użyj optymalizatora SGD ze stałą uczącą `1e-3`. Trenuj przez 3000 epok. Pamiętaj, aby przekazać do optymalizatora `torch.optim.SGD` parametry sieci (metoda `.parameters()`). Dopisz logowanie kosztu raz na 100 epok.

# %% colab={"base_uri": "https://localhost:8080/"} id="NbABKz5-LAs2" outputId="086dc0f3-0184-4072-9fd3-275b60dee2e4" tags=["ex"]
learning_rate = 1e-3

model = nn.Linear(in_features=X_train.shape[1], out_features=1)
activation = nn.Sigmoid()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()

# implement me!
# your_code
num_epochs = 3000
for i in range(num_epochs):

    logit = model(X_train)
    y_hat = activation(logit)
    
    loss = loss_fn(y_hat, y_train)
    
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if i % 100 == 0:
        print(f"step {i} loss: {loss.item():.4f}")

print(f"step {num_epochs} loss: {loss.item():.4f}")


# %% [markdown] tags=["ex"]
# Teraz trzeba sprawdzić, jak poszło naszej sieci. W PyTorchu sieć pracuje zawsze w jednym z dwóch trybów: treningowym lub ewaluacyjnym (predykcyjnym). Ten drugi wyłącza niektóre mechanizmy, które są używane tylko podczas treningu, w szczególności regularyzację dropout. Do przełączania służą metody modelu `.train()` i `.eval()`.
#
# Dodatkowo podczas liczenia predykcji dobrze jest wyłączyć liczenie gradientów, bo nie będą potrzebne, a oszczędza to czas i pamięć. Używa się do tego menadżera kontekstu `with torch.no_grad():`.

# %% colab={"base_uri": "https://localhost:8080/"} id="zH37zDX4LAs2" outputId="b1f93309-6f04-4ffc-b0ca-08d0a32120a0" tags=["ex"]
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, roc_auc_score


model.eval()
with torch.no_grad():
    y_score = activation(model(X_test))

auroc = roc_auc_score(y_test, y_score)
print(f"AUROC: {auroc:.2%}")

# %% tags=["ex"]
assert isinstance(model, nn.Linear)
assert isinstance(activation, nn.Sigmoid)
assert isinstance(optimizer, torch.optim.SGD)
assert isinstance(loss_fn, torch.nn.BCELoss)

assert model.out_features == 1
assert optimizer.param_groups[0]["lr"] == 1e-3

assert 0.0 < loss.item() < 0.5
assert 0.8 < auroc < 0.9

print("Solution is correct!")

# %% [markdown]
# Jest to całkiem dobry wynik, a może być jeszcze lepszy. Sprawdźmy dla pewności jeszcze inne metryki: precyzję, recall oraz F1-score. Dodatkowo narysujemy krzywą precision-recall, czyli jak zmieniają się te metryki w zależności od przyjętego progu (threshold) prawdopodobieństwa, powyżej którego przyjmujemy klasę pozytywną. Taką krzywą należy rysować na zbiorze walidacyjnym, bo później chcemy wykorzystać tę informację do doboru progu, a nie chcemy mieć wycieku danych testowych (data leakage).
#
# Poniżej zaimplementowano także funkcję `get_optimal_threshold()`, która sprawdza, dla którego progu uzyskujemy maksymalny F1-score, i zwraca indeks oraz wartość optymalnego progu. Przyda ci się ona w dalszej części laboratorium.

# %%
from sklearn.metrics import PrecisionRecallDisplay


def get_optimal_threshold(
    precisions: np.array, 
    recalls: np.array, 
    thresholds: np.array
) -> Tuple[int, float]:
    
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_idx, optimal_threshold


def plot_precision_recall_curve(y_true, y_pred_score) -> None:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_score)
    optimal_idx, optimal_threshold = get_optimal_threshold(precisions, recalls, thresholds)

    disp = PrecisionRecallDisplay(precisions, recalls)
    disp.plot()
    plt.title(f"Precision-recall curve (opt. thresh.: {optimal_threshold:.4f})")
    plt.axvline(recalls[optimal_idx], color="green", linestyle="-.")
    plt.axhline(precisions[optimal_idx], color="green", linestyle="-.")
    plt.show()



# %%
model.eval()
with torch.no_grad():
    y_pred_valid_score = activation(model(X_valid))

plot_precision_recall_curve(y_valid, y_pred_valid_score)

# %% [markdown] id="vfQPIUQ_LAs2"
# Jak widać, chociaż AUROC jest wysokie, to dla optymalnego F1-score recall nie jest zbyt wysoki, a precyzja jest już dość niska. Być może wynik uda się poprawić, używając modelu o większej pojemności - pełnej, głębokiej sieci neuronowej.

# %% [markdown]
# ## Sieci neuronowe

# %% [markdown] id="YP298w6Cq7T6"
# Wszystko zaczęło się od inspirowanych biologią [sztucznych neuronów](https://en.wikipedia.org/wiki/Artificial_neuron), których próbowano użyć do symulacji mózgu. Naukowcy szybko odeszli od tego podejścia (sam problem modelowania okazał się też znacznie trudniejszy, niż sądzono), zamiast tego używając neuronów jako jednostek reprezentującą dowolną funkcję parametryczną $f(x, \Theta)$. Każdy neuron jest zatem bardzo elastyczny, bo jedyne wymagania to funkcja różniczkowalna, a mamy do tego wektor parametrów $\Theta$.
#
# W praktyce najczęściej można spotkać się z kilkoma rodzinami sieci neuronowych:
# 1. Perceptrony wielowarstwowe (*MultiLayer Perceptron*, MLP) - najbardziej podobne do powyższego opisu, niezbędne do klasyfikacji i regresji
# 2. Konwolucyjne (*Convolutional Neural Networks*, CNNs) - do przetwarzania danych z zależnościami przestrzennymi, np. obrazów czy dźwięku
# 3. Rekurencyjne (*Recurrent Neural Networks*, RNNs) - do przetwarzania danych z zależnościami sekwencyjnymi, np. szeregi czasowe, oraz kiedyś do języka naturalnego
# 4. Transformacyjne (*Transformers*), oparte o mechanizm atencji (*attention*) - do przetwarzania języka naturalnego (NLP), z którego wyparły RNNs, a coraz częściej także do wszelkich innych danych, np. obrazów, dźwięku
# 5. Grafowe (*Graph Neural Networks*, GNNS) - do przetwarzania grafów
#
# Na tym laboratorium skupimy się na najprostszej architekturze, czyli MLP. Jest ona powszechnie łączona z wszelkimi innymi architekturami, bo pozwala dokonywać klasyfikacji i regresji. Przykładowo, klasyfikacja obrazów to zwykle CNN + MLP, klasyfikacja tekstów to transformer + MLP, a regresja na grafach to GNN + MLP.
#
# Dodatkowo, pomimo prostoty MLP są bardzo potężne - udowodniono, że perceptrony (ich powszechna nazwa) są [uniwersalnym aproksymatorem](https://www.sciencedirect.com/science/article/abs/pii/0893608089900208), będącym w stanie przybliżyć dowolną funkcję z odpowiednio małym błędem, zakładając wystarczającą wielkość warstw sieci. Szczególne ich wersje potrafią nawet [reprezentować drzewa decyzyjne](https://www.youtube.com/watch?v=_okxGdHM5b8).
#
# Dla zainteresowanych polecamy [doskonałą książkę "Dive into Deep Learning", z implementacjami w PyTorchu](https://d2l.ai/chapter_multilayer-perceptrons/index.html), [klasyczną książkę "Deep Learning Book"](https://www.deeplearningbook.org/contents/mlp.html), oraz [ten filmik](https://www.youtube.com/watch?v=BFHrIxKcLjA), jeśli zastanawiałeś/-aś się, czemu używamy deep learning, a nie naprzykład (wide?) learning. (aka. czemu staramy się budować głębokie sieci, a nie płytkie za to szerokie)

# %% [markdown] id="S_ZjoGBU5upj"
# ### Sieci MLP
#
# Dla przypomnienia, na wejściu mamy punkty ze zbioru treningowego, czyli $d$-wymiarowe wektory. W klasyfikacji chcemy znaleźć granicę decyzyjną, czyli krzywą, która oddzieli od siebie klasy. W wejściowej przestrzeni może być to trudne, bo chmury punktów z poszczególnych klas mogą być ze sobą dość pomieszane. Pamiętajmy też, że regresja logistyczna jest klasyfikatorem liniowym, czyli w danej przestrzeni potrafi oddzielić punkty tylko linią prostą.
#
# Sieć MLP składa się z warstw. Każda z nich dokonuje nieliniowego przekształcenia przestrzeni (można o tym myśleć jak o składaniu przestrzeni jakąś prostą/łamaną), tak, aby w finalnej przestrzeni nasze punkty były możliwie liniowo separowalne. Wtedy ostatnia warstwa z sigmoidą będzie potrafiła je rozdzielić od siebie.
#
# ![1_x-3NGQv0pRIab8xDT-f_Hg.png](attachment:1_x-3NGQv0pRIab8xDT-f_Hg.png)
#
# Poszczególne neurony składają się z iloczynu skalarnego wejść z wagami neuronu, oraz nieliniowej funkcji aktywacji. W PyTorchu są to osobne obiekty - `nn.Linear` oraz np. `nn.Sigmoid`. Funkcja aktywacji przyjmuje wynik iloczynu skalarnego i przekształca go, aby sprawdzić, jak mocno reaguje neuron na dane wejście. Musi być nieliniowa z dwóch powodów. Po pierwsze, tylko nieliniowe przekształcenia są na tyle potężne, żeby umożliwić liniową separację danych w ostatniej warstwie. Po drugie, liniowe przekształcenia zwyczajnie nie działają. Aby zrozumieć czemu, trzeba zobaczyć, co matematycznie oznacza sieć MLP.
#
# ![perceptron](https://www.saedsayad.com/images/Perceptron_bkp_1.png)
#
# Zapisane matematycznie MLP to:
#
# $\large
# h_1 = f_1(x) \\
# h_2 = f_2(h_1) \\
# h_3 = f_3(h_2) \\
# ... \\
# h_n = f_n(h_{n-1})
# $
#
# gdzie $x$ to wejście $f_i$ to funkcja aktywacji $i$-tej warstwy, a $h_i$ to wyjście $i$-tej warstwy, nazywane **ukrytą reprezentacją (hidden representation)**, lub *latent representation*. Nazwa bierze się z tego, że w środku sieci wyciągamy cechy i wzorce w danych, które nie są widoczne na pierwszy rzut oka na wejściu.
#
# Załóżmy, że uczymy się na danych $x$ o jednym wymiarze (dla uproszczenia wzorów) oraz nie mamy funkcji aktywacji, czyli wykorzystujemy tak naprawdę aktywację liniową $f(x) = x$. Zobaczmy jak będą wyglądać dane przechodząc przez kolejne warstwy:
#
# $\large
# h_1 = f_1(xw_1) = xw_1 \\
# h_2 = f_2(h_1w_2) = xw_1w_2 \\
# ... \\
# h_n = f_n(h_{n-1}w_n) = xw_1w_2...w_n
# $
#
# gdzie $w_i$ to jest parametr $i$-tej warstwy sieci, $x$ to są dane (w naszym przypadku jedna liczba) wejściowa, a $h_i$ to wyjście $i$-tej warstwy.
#
# Jak widać, taka sieć o $n$ warstwach jest równoważna sieci o jednej warstwie z parametrem $w = w_1w_2...w_n$. Wynika to z tego, że złożenie funkcji liniowych jest także funkcją liniową - patrz notatki z algebry :)
#
# Jeżeli natomiast użyjemy nieliniowej funkcji aktywacji, często oznaczanej jako $\sigma$, to wszystko będzie działać. Co ważne, ostatnia warstwa, dająca wyjście sieci, ma zwykle inną aktywację od warstw wewnątrz sieci, bo też ma inne zadanie - zwrócić wartość dla klasyfikacji lub regresji. Na wyjściu korzysta się z funkcji liniowej (regresja), sigmoidalnej (klasyfikacja binarna) lub softmax (klasyfikacja wieloklasowa).
#
# Wewnątrz sieci używano kiedyś sigmoidy oraz tangensa hiperbolicznego `tanh`, ale okazało się to nieefektywne przy uczeniu głębokich sieci o wielu warstwach. Nowoczesne sieci korzystają zwykle z funkcji ReLU (*rectified linear unit*), która jest zaskakująco prosta: $ReLU(x) = \max(0, x)$. Okazało się, że bardzo dobrze nadaje się do treningu nawet bardzo głębokich sieci neuronowych. Nowsze funkcje aktywacji są głównie modyfikacjami ReLU.
#
# ![relu](https://www.nomidl.com/wp-content/uploads/2022/04/image-10.png)

# %% [markdown]
# ### MLP w PyTorchu
#
# Warstwę neuronów w MLP nazywa się warstwą gęstą (*dense layer*) lub warstwą w pełni połączoną (*fully-connected layer*), i taki opis oznacza zwykle same neurony oraz funkcję aktywacji. PyTorch, jak już widzieliśmy, definiuje osobno transformację liniową oraz aktywację, a więc jedna warstwa składa się de facto z 2 obiektów, wywoływanych jeden po drugim. Inne frameworki, szczególnie wysokopoziomowe (np. Keras) łączą to często w jeden obiekt.
#
# MLP składa się zatem z sekwencji obiektów, które potem wywołuje się jeden po drugim, gdzie wyjście poprzedniego to wejście kolejnego. Ale nie można tutaj używać Pythonowych list! Z perspektywy PyTorcha to wtedy niezależne obiekty i nie zostanie wtedy przekazany między nimi gradient. Trzeba tutaj skorzystać z `nn.Sequential`, aby tworzyć taki pipeline.
#
# Rozmiary wejścia i wyjścia dla każdej warstwy trzeba w PyTorchu podawać explicite. Jest to po pierwsze edukacyjne, a po drugie często ułatwia wnioskowanie o działaniu sieci oraz jej debugowanie - mamy jasno podane, czego oczekujemy. Niektóre frameworki (np. Keras) obliczają to automatycznie.
#
# Co ważne, ostatnia warstwa zwykle nie ma funkcji aktywacji. Wynika to z tego, że obliczanie wielu funkcji kosztu (np. entropii krzyżowej) na aktywacjach jest często niestabilne numerycznie. Z tego powodu PyTorch oferuje funkcje kosztu zawierające w środku aktywację dla ostatniej warstwy, a ich implementacje są stabilne numerycznie. Przykładowo, `nn.BCELoss` przyjmuje wejście z zaaplikowanymi już aktywacjami, ale może skutkować under/overflow, natomiast `nn.BCEWithLogitsLoss` przyjmuje wejście bez aktywacji, a w środku ma specjalną implementację łączącą binarną entropię krzyżową z aktywacją sigmoidalną. Oczywiście w związku z tym aby dokonać potem predykcji w praktyce, trzeba pamiętać o użyciu funkcji aktywacji. Często korzysta się przy tym z funkcji z modułu `torch.nn.functional`, które są w tym wypadku nieco wygodniejsze od klas wywoływalnych z `torch.nn`.
#
# Całe sieci w PyTorchu tworzy się jako klasy dziedziczące po `nn.Module`. Co ważne, obiekty, z których tworzymy sieć, np. `nn.Linear`, także dziedziczą po tej klasie. Pozwala to na bardzo modułową budowę kodu, zgodną z zasadami OOP. W konstruktorze najpierw trzeba zawsze wywołać konstruktor rodzica - `super().__init__()`, a później tworzy się potrzebne obiekty i zapisuje jako atrybuty. Każdy atrybut dziedziczący po `nn.Module` lub `nn.Parameter` jest uważany za taki, który zawiera parametry sieci, a więc przy wywołaniu metody `parameters()` - parametry z tych atrybutów pojawią się w liście wszystkich parametrów. Musimy też zdefiniować metodę `forward()`, która przyjmuje tensor `x` i zwraca wynik. Typowo ta metoda po prostu używa obiektów zdefiniowanych w konstruktorze.
#
#
# **UWAGA: nigdy w normalnych warunkach się nie woła metody `forward` ręcznie**

# %% [markdown] id="J8niDgExAMDO" tags=["ex"]
# ### Zadanie 4 (0.5 punktu)
#
# Uzupełnij implementację 3-warstwowej sieci MLP. Użyj rozmiarów:
# * pierwsza warstwa: input_size x 256
# * druga warstwa: 256 x 128
# * trzecia warstwa: 128 x 1
#
# Użyj funkcji aktywacji ReLU.
#
# Przydatne klasy:
# - `nn.Sequential`
# - `nn.Linear`
# - `nn.ReLU`

# %% tags=["ex"]
from torch import sigmoid


class MLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()

        # implement me!
        # your_code
        self.model = nn.Sequential(nn.Linear(in_features=input_size, out_features=256), nn.ReLU(),
                                   nn.Linear(in_features=256, out_features=128), nn.ReLU(),
                                   nn.Linear(in_features=128, out_features=1))

    def forward(self, x):
        # implement me!
        # your_code
        return self.model(x)

    def predict_proba(self, x):
        return sigmoid(self(x))
    
    def predict(self, x, threshold: float = 0.5):
        y_pred_score = self.predict_proba(x)
        return (y_pred_score > threshold).to(torch.int32)



# %% tags=["ex"]
learning_rate = 1e-3
model = MLP(input_size=X_train.shape[1])
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# note that we are using loss function with sigmoid built in
loss_fn = torch.nn.BCEWithLogitsLoss()
num_epochs = 2000
evaluation_steps = 200

for i in range(num_epochs):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if i % evaluation_steps == 0:
        print(f"Epoch {i} train loss: {loss.item():.4f}")

print(f"final loss: {loss.item():.4f}")

# %% colab={"base_uri": "https://localhost:8080/"} id="LP5GSup24dXU" outputId="05f332c4-5d94-41f6-f85b-17793d3c4b49" tags=["ex"]
model.eval()
with torch.no_grad():
    # positive class probabilities
    y_pred_valid_score = model.predict_proba(X_valid)
    y_pred_test_score = model.predict_proba(X_test)

auroc = roc_auc_score(y_test, y_pred_test_score)
print(f"AUROC: {auroc:.2%}")

plot_precision_recall_curve(y_valid, y_pred_valid_score)

# %% tags=["ex"]
assert any(
    isinstance(module, nn.Linear) and module.in_features == X_train.shape[1] and module.out_features == 256
    for module in model.modules()
)

assert any(
    isinstance(module, nn.Linear) and module.in_features == 256 and module.out_features == 128
    for module in model.modules()
)

assert any(
    isinstance(module, nn.Linear) and module.in_features == 128 and module.out_features == 1
    for module in model.modules()
)

assert any(isinstance(module, nn.ReLU) for module in model.modules())

assert 0.5 < loss.item() < 0.6
assert 0.6 < auroc < 0.9

print("Solution is correct!")

# %% [markdown]
# AUROC jest podobne, a precision i recall spadły - wypadamy wręcz gorzej od regresji liniowej! Skoro dodaliśmy więcej warstw, to może pojemność modelu jest teraz za duża i trzeba by go zregularyzować?
#
# Sieci neuronowe bardzo łatwo przeuczają, bo są bardzo elastycznymi i pojemnymi modelami. Dlatego mają wiele różnych rodzajów regularyzacji, których używa się razem. Co ciekawe, udowodniono eksperymentalnie, że zbyt duże sieci z mocną regularyzacją działają lepiej niż mniejsze sieci, odpowiedniego rozmiaru, za to ze słabszą regularyzacją.
#
# Pierwszy rodzaj regularyzacji to znana nam już **regularyzacja L2**, czyli penalizacja zbyt dużych wag. W kontekście sieci neuronowych nazywa się też ją czasem *weight decay*. W PyTorchu dodaje się ją jako argument do optymalizatora.
#
# Regularyzacja specyficzna dla sieci neuronowych to **dropout**. Polega on na losowym wyłączaniu zadanego procenta neuronów podczas treningu. Pomimo prostoty okazała się niesamowicie skuteczna, szczególnie w treningu bardzo głębokich sieci. Co ważne, jest to mechanizm używany tylko podczas treningu - w trakcie predykcji za pomocą sieci wyłącza się ten mechanizm i dokonuje normalnie predykcji całą siecią. Podejście to można potraktować jak ensemble learning, podobny do lasów losowych - wyłączając losowe części sieci, w każdej iteracji trenujemy nieco inną sieć, co odpowiada uśrednianiu predykcji różnych algorytmów. Typowo stosuje się dość mocny dropout, rzędu 25-50%. W PyTorchu implementuje go warstwa `nn.Dropout`, aplikowana zazwyczaj po funkcji aktywacji.
#
# Ostatni, a być może najważniejszy rodzaj regularyzacji to **wczesny stop (early stopping)**. W każdym kroku mocniej dostosowujemy terenową sieć do zbioru treningowego, a więc zbyt długi trening będzie skutkował przeuczeniem. W metodzie wczesnego stopu używamy wydzielonego zbioru walidacyjnego (pojedynczego, metoda holdout), sprawdzając co określoną liczbę epok wynik na tym zbiorze. Jeżeli nie uzyskamy wyniku lepszego od najlepszego dotychczas uzyskanego przez określoną liczbę epok, to przerywamy trening. Okres, przez który czekamy na uzyskanie lepszego wyniku, to cierpliwość (*patience*). Im mniejsze, tym mocniejszy jest ten rodzaj regularyzacji, ale trzeba z tym uważać, bo łatwo jest przesadzić i zbyt szybko przerywać trening. Niektóre implementacje uwzględniają tzw. *grace period*, czyli gwarantowaną minimalną liczbę epok, przez którą będziemy trenować sieć, niezależnie od wybranej cierpliwości.
#
# Dodatkowo ryzyko przeuczenia można zmniejszyć, używając mniejszej stałej uczącej.

# %% [markdown] tags=["ex"]
# ### Zadanie 5 (1.5 punktu)
#
# Zaimplementuj funkcję `evaluate_model()`, obliczającą metryki na zbiorze testowym:
# - wartość funkcji kosztu (loss)
# - AUROC
# - optymalny próg
# - F1-score przy optymalnym progu
# - precyzję oraz recall dla optymalnego progu
#
# Jeżeli podana jest wartość argumentu `threshold`, to użyj jej do zamiany prawdopodobieństw na twarde predykcje. W przeciwnym razie użyj funkcji `get_optimal_threshold` i oblicz optymalną wartość progu.
#
# Pamiętaj o przełączeniu modelu w tryb ewaluacji oraz o wyłączeniu obliczania gradientów.

# %% tags=["ex"]
from typing import Optional

from sklearn.metrics import precision_score, recall_score, f1_score
from torch import sigmoid


def evaluate_model(
    model: nn.Module, 
    X: torch.Tensor, 
    y: torch.Tensor, 
    loss_fn: nn.Module,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    # implement me!
    # your_code
    model.eval()
    
    with torch.no_grad():
        y_hat = model(X)
        y_prob = model.predict_proba(X)
        loss = loss_fn(y_hat, y)
        
        auroc = roc_auc_score(y, y_prob)
        
        if threshold is None:
            precisions, recalls, thresholds = precision_recall_curve(y, y_prob)
            optimal_idx, threshold = get_optimal_threshold(precisions, recalls, thresholds)
        
        y_pred = model.predict(X, threshold)
        
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
    
    return {
        'loss': loss,
        'AUROC': auroc,
        'threshold': threshold,
        'F1-score': f1,
        'precision': precision,
        'recall': recall
    }



# %% tags=["ex"]
eval_result = evaluate_model(model, X_train, y_train, loss_fn)

assert 0.5 < eval_result["loss"] < 0.6
assert 0.6 < eval_result["AUROC"] < 0.9
assert 0.3 < eval_result["precision"] < 0.6
assert 0.6 < eval_result["recall"] < 0.9
assert 0.4 < eval_result["F1-score"] < 0.7

print("Solution is correct!")


# %% [markdown] tags=["ex"]
# ### Zadanie 6 (0.5 punktu)
#
# Zaimplementuj 3-warstwową sieć MLP z dropout (50%). Rozmiary warstw ukrytych mają wynosić 256 i 128.

# %% tags=["ex"]
class RegularizedMLP(nn.Module):
    def __init__(self, input_size: int, dropout_p: float = 0.5):
        super().__init__()

        # implement me!
        # your_code
        self.model = nn.Sequential(nn.Linear(in_features=input_size, out_features=256), nn.ReLU(),
                                   nn.Dropout(p=dropout_p),
                                   nn.Linear(in_features=256, out_features=128), nn.ReLU(),
                                   nn.Dropout(p=dropout_p),
                                   nn.Linear(in_features=128, out_features=1))
    
    def forward(self, x):
        # implement me!
        # your_code
        return self.model(x)

    def predict_proba(self, x):
        return sigmoid(self(x))
    
    def predict(self, x, threshold: float = 0.5):
        y_pred_score = self.predict_proba(x)
        return (y_pred_score > threshold).to(torch.int32)



# %% tags=["ex"]
verification_input_size = 143
verification_dropout_p = 0.125
verification_model = RegularizedMLP(
    input_size=verification_input_size,
    dropout_p=verification_dropout_p,
)

assert any(
    isinstance(module, nn.Linear) and module.in_features == verification_input_size and module.out_features == 256
    for module in verification_model.modules()
)

assert any(
    isinstance(module, nn.Linear) and module.in_features == 256 and module.out_features == 128
    for module in verification_model.modules()
)

assert any(
    isinstance(module, nn.Linear) and module.in_features == 128 and module.out_features == 1
    for module in verification_model.modules()
)

assert any(isinstance(module, nn.Dropout) and module.p == verification_dropout_p for module in verification_model.modules())
assert any(isinstance(module, nn.ReLU) for module in verification_model.modules())

print("Solution is correct!")

# %% [markdown] id="rEk9azaULAsz"
# Opisaliśmy wcześniej podstawowy optymalizator w sieciach neuronowych - spadek wzdłuż gradientu. Jednak wymaga on użycia całego zbioru danych, aby obliczyć gradient, co jest często niewykonalne przez rozmiar zbioru. Dlatego wymyślono **stochastyczny spadek wzdłuż gradientu (stochastic gradient descent, SGD)**, w którym używamy 1 przykładu naraz, liczymy gradient tylko po nim i aktualizujemy parametry. Jest to oczywiście dość grube przybliżenie gradientu, ale pozwala robić szybko dużo małych kroków. Kompromisem, którego używa się w praktyce, jest **minibatch gradient descent**, czyli używanie batchy np. 32, 64 czy 128 przykładów.
#
# Rzadko wspominanym, a ważnym faktem jest także to, że stochastyczność metody optymalizacji jest sama w sobie też [metodą regularyzacji](https://arxiv.org/abs/2101.12176), a więc `batch_size` to także hiperparametr.
#
# Obecnie najpopularniejszą odmianą SGD jest [Adam](https://arxiv.org/abs/1412.6980), gdyż uczy on szybko sieć oraz daje bardzo dobre wyniki nawet przy niekoniecznie idealnie dobranych hiperparametrach. W PyTorchu najlepiej korzystać z jego implementacji `AdamW`, która jest nieco lepsza niż implementacja `Adam`. Jest to zasadniczo zawsze wybór domyślny przy treningu współczesnych sieci neuronowych.
#
# Na razie użyjemy jednak minibatch SGD.

# %% [markdown]
# Poniżej znajduje się implementacja prostej klasy dziedziczącej po `Dataset` - tak w PyTorchu implementuje się własne zbiory danych. Użycie takich klas umożliwia użycie klas ładujących dane (`DataLoader`), które z kolei pozwalają łatwo ładować batche danych. Trzeba w takiej klasie zaimplementować metody:
# - `__len__` - zwraca ilość punktów w zbiorze
# - `__getitem__` - zwraca przykład ze zbioru pod danym indeksem oraz jego klasę
#

# %%
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, y):
        super().__init__()
        
        self.data = data
        self.y = y
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.y[idx]



# %% [markdown] tags=["ex"]
# ### Zadanie 7 (1.5 punktu)
#
# Zaimplementuj pętlę treningowo-walidacyjną dla sieci neuronowej. Wykorzystaj podane wartości hiperparametrów do treningu (stała ucząca, prawdopodobieństwo dropoutu, regularyzacja L2, rozmiar batcha, maksymalna liczba epok). Użyj optymalizatora SGD.
#
# Dodatkowo zaimplementuj regularyzację przez early stopping. Sprawdzaj co epokę wynik na zbiorze walidacyjnym. Użyj podanej wartości patience, a jako metryki po prostu wartości funkcji kosztu. Może się tutaj przydać zaimplementowana funkcja `evaluate_model()`.
#
# Pamiętaj o tym, aby przechowywać najlepszy dotychczasowy wynik walidacyjny oraz najlepszy dotychczasowy model. Zapamiętaj też optymalny próg do klasyfikacji dla najlepszego modelu.

# %% tags=["ex"]
from copy import deepcopy

from torch.utils.data import DataLoader


learning_rate = 1e-3
dropout_p = 0.5
l2_reg = 1e-4
batch_size = 128
max_epochs = 300

early_stopping_patience = 4

# %% tags=["ex"]
model = RegularizedMLP(
    input_size=X_train.shape[1], 
    dropout_p=dropout_p
)
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=learning_rate, 
    weight_decay=l2_reg
)
loss_fn = torch.nn.BCEWithLogitsLoss()

train_dataset = MyDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

steps_without_improvement = 0

best_val_loss = np.inf
best_model = None
best_threshold = None

for epoch_num in range(max_epochs):
    model.train()
    
    # note that we are using DataLoader to get batches
    for X_batch, y_batch in train_dataloader:
        # model training
        # implement me!
        # your_code
        y_hat = model(X_batch)
        loss = loss_fn(y_hat, y_batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # model evaluation, early stopping
    # implement me!
    # your_code
    metrics = evaluate_model(model, X_valid, y_valid, loss_fn)

    if best_val_loss > metrics['loss']:
        best_model = deepcopy(model)
        best_val_loss = metrics['loss']
        best_threshold = metrics['threshold']
        steps_without_improvement = 0
    elif steps_without_improvement < early_stopping_patience:
        steps_without_improvement += 1
    else:
        print(f"Early stop for epoch {epoch_num}")
        break

    if epoch_num % 10 == 0:
        print(f"Epoch {epoch_num} train loss: {loss.item():.4f}, eval loss {metrics['loss']}")


# %% tags=["ex"]
test_metrics = evaluate_model(best_model, X_test, y_test, loss_fn, best_threshold)

print(f"AUROC: {test_metrics['AUROC']:.2%}")
print(f"F1: {test_metrics['F1-score']:.2%}")
print(f"Precision: {test_metrics['precision']:.2%}")
print(f"Recall: {test_metrics['recall']:.2%}")

# %% tags=["ex"]
assert test_metrics["AUROC"] > 0.8
assert test_metrics["F1-score"] > 0.6
assert test_metrics["precision"] > 0.5
assert test_metrics["recall"] > 0.6

print("Solution is correct!")


# %% [markdown]
# Wyniki wyglądają już dużo lepiej.
#
# Na koniec laboratorium dołożymy do naszego modelu jeszcze 3 powrzechnie używane techniki, które są bardzo proste, a pozwalają często ulepszyć wynik modelu.
#
# Pierwszą z nich są **warstwy normalizacji (normalization layers)**. Powstały one początkowo z założeniem, że przez przekształcenia przestrzeni dokonywane przez sieć zmienia się rozkład prawdopodobieństw pomiędzy warstwami, czyli tzw. *internal covariate shift*. Później okazało się, że zastosowanie takiej normalizacji wygładza powierzchnię funkcji kosztu, co ułatwia i przyspiesza optymalizację. Najpowszechniej używaną normalizacją jest **batch normalization (batch norm)**.
#
# Drugim ulepszeniem jest dodanie **wag klas (class weights)**. Mamy do czynienia z problemem klasyfikacji niezbalansowanej, więc klasa mniejszościowa, ważniejsza dla nas, powinna dostać większą wagę. Implementuje się to trywialnie prosto - po prostu mnożymy wartość funkcji kosztu dla danego przykładu przez wagę dla prawdziwej klasy tego przykładu. Praktycznie każdy klasyfikator operujący na jakiejś ważonej funkcji może działać w ten sposób, nie tylko sieci neuronowe.
#
# Ostatnim ulepszeniem jest zamiana SGD na optymalizator Adam, a konkretnie na optymalizator `AdamW`. Jest to przykład **optymalizatora adaptacyjnego (adaptive optimizer)**, który potrafi zaadaptować stałą uczącą dla każdego parametru z osobna w trakcie treningu. Wykorzystuje do tego gradienty - w uproszczeniu, im większa wariancja gradientu, tym mniejsze kroki w tym kierunku robimy.

# %% [markdown] tags=["ex"]
# ### Zadanie 8 (0.5 punktu)
#
# Zaimplementuj model `NormalizingMLP`, o takiej samej strukturze jak `RegularizedMLP`, ale dodatkowo z warstwami `BatchNorm1d` pomiędzy warstwami `Linear` oraz `ReLU`.
#
# Za pomocą funkcji `compute_class_weight()` oblicz wagi dla poszczególnych klas. Użyj opcji `"balanced"`. Przekaż do funkcji kosztu wagę klasy pozytywnej (pamiętaj, aby zamienić ją na tensor).
#
# Zamień używany optymalizator na `AdamW`.
#
# Na koniec skopiuj resztę kodu do treningu z poprzedniego zadania, wytrenuj sieć i oblicz wyniki na zbiorze testowym.

# %% tags=["ex"]
class NormalizingMLP(nn.Module):
    def __init__(self, input_size: int, dropout_p: float = 0.5):
        super().__init__()

        # implement me!
        # your_code
        self.mlp = nn.Sequential(nn.Linear(in_features=input_size, out_features=256), nn.BatchNorm1d(num_features=256),
                                   nn.ReLU(), nn.Dropout(p=dropout_p),
                                   nn.Linear(in_features=256, out_features=128), nn.BatchNorm1d(num_features=128),
                                   nn.ReLU(), nn.Dropout(p=dropout_p),
                                   nn.Linear(in_features=128, out_features=1))
    
    def forward(self, x):
        return self.mlp(x)

    def predict_proba(self, x):
        return sigmoid(self(x))
    
    def predict(self, x, threshold: float = 0.5):
        y_pred_score = self.predict_proba(x)
        return (y_pred_score > threshold).to(torch.int32)



# %% tags=["ex"]
# define all the hyperparameters
# your_code
learning_rate = 1e-3
dropout_p = 0.5
l2_reg = 1e-4
batch_size = 128
max_epochs = 300

early_stopping_patience = 4


# %% tags=["ex"]
# training loop
# your_code
model = NormalizingMLP(
    input_size=X_train.shape[1], 
    dropout_p=dropout_p
)
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=learning_rate, 
    weight_decay=l2_reg
)
loss_fn = torch.nn.BCEWithLogitsLoss()

train_dataset = MyDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

steps_without_improvement = 0

best_val_loss = np.inf
best_model = None
best_threshold = None

for epoch_num in range(max_epochs):
    model.train()
    
    # note that we are using DataLoader to get batches
    for X_batch, y_batch in train_dataloader:
        # model training
        # implement me!
        # your_code
        y_hat = model(X_batch)
        loss = loss_fn(y_hat, y_batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # model evaluation, early stopping
    # implement me!
    # your_code
    metrics = evaluate_model(model, X_valid, y_valid, loss_fn)

    if best_val_loss > metrics['loss']:
        best_model = deepcopy(model)
        best_val_loss = metrics['loss']
        best_threshold = metrics['threshold']
        steps_without_improvement = 0
    elif steps_without_improvement < early_stopping_patience:
        steps_without_improvement += 1
    else:
        print(f"Early stop for epoch {epoch_num}")
        break

    if epoch_num % 10 == 0:
        print(f"Epoch {epoch_num} train loss: {loss.item():.4f}, eval loss {metrics['loss']}")


# %% tags=["ex"]
test_metrics = evaluate_model(best_model, X_test, y_test, loss_fn, best_threshold)

print(f"AUROC: {test_metrics['AUROC']:.2%}")
print(f"F1: {test_metrics['F1-score']:.2%}")
print(f"Precision: {test_metrics['precision']:.2%}")
print(f"Recall: {test_metrics['recall']:.2%}")

# %% tags=["ex"]
verification_input_size = 143
verification_dropout_p = 0.125
verification_model = NormalizingMLP(
    input_size=verification_input_size,
    dropout_p=verification_dropout_p,
)

assert any(
    isinstance(module, nn.Linear) and module.in_features == verification_input_size and module.out_features == 256
    for module in verification_model.modules()
)

assert any(
    isinstance(module, nn.Linear) and module.in_features == 256 and module.out_features == 128
    for module in verification_model.modules()
)

assert any(
    isinstance(module, nn.Linear) and module.in_features == 128 and module.out_features == 1
    for module in verification_model.modules()
)

assert any(isinstance(module, nn.Dropout) and module.p == verification_dropout_p for module in verification_model.modules())
assert any(isinstance(module, nn.ReLU) for module in verification_model.modules())

assert any(isinstance(module, nn.BatchNorm1d) for module in verification_model.modules())

assert test_metrics["AUROC"] > 0.8
assert test_metrics["F1-score"] > 0.6
assert test_metrics["precision"] > 0.5
assert test_metrics["recall"] > 0.6

print("Solution is correct!")

# %% [markdown] id="XyoRnHT4GFR9"
# ## Akceleracja sprzętowa (dla zainteresowanych)

# %% [markdown]
# Jak wcześniej wspominaliśmy, użycie akceleracji sprzętowej, czyli po prostu GPU do obliczeń, jest bardzo efektywne w przypadku sieci neuronowych. Karty graficzne bardzo efektywnie mnożą macierze, a sieci neuronowe to, jak można było się przekonać, dużo mnożenia macierzy.
#
# W PyTorchu jest to dosyć łatwe, ale trzeba robić to explicite. Służy do tego metoda `.to()`, która przenosi tensory między CPU i GPU. Poniżej przykład, jak to się robi (oczywiście trzeba mieć skonfigurowane GPU, żeby działało):

# %%
import time 


class CudaMLP(nn.Module):
    def __init__(self, input_size: int, dropout_p: float = 0.5):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        return self.mlp(x)

    def predict_proba(self, x):
        return sigmoid(self(x))
    
    def predict(self, x, threshold: float = 0.5):
        y_pred_score = self.predict_proba(x)
        return (y_pred_score > threshold).to(torch.int32)


model = CudaMLP(X_train.shape[1]).to('cuda')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# note that we are using loss function with sigmoid built in
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(weights)[1].to('cuda'))

step_counter = 0
time_from_eval = time.time()
for epoch_id in range(30):
    for batch_x, batch_y in train_dataloader:
        batch_x = batch_x.to('cuda')
        batch_y = batch_y.to('cuda')
        
        loss = loss_fn(model(batch_x), batch_y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        if step_counter % evaluation_steps == 0:
            print(f"Epoch {epoch_id} train loss: {loss.item():.4f}, time: {time.time() - time_from_eval}")
            time_from_eval = time.time()

        step_counter += 1

test_res = evaluate_model(model.to('cpu'), X_test, y_test, loss_fn.to('cpu'), threshold=0.5)

print(f"AUROC: {test_res['AUROC']:.2%}")
print(f"F1: {test_res['F1-score']:.2%}")
print(test_res)

# %% [markdown]
# Co prawda ten model nie będzie tak dobry jak ten z laboratorium, ale zwróć uwagę, o ile jest większy, a przy tym szybszy.
#
# Dla zainteresowanych polecamy [tę serie artykułów](https://medium.com/@adi.fu7/ai-accelerators-part-i-intro-822c2cdb4ca4)

# %% [markdown] tags=["ex"]
# ## Zadanie dla chętnych

# %% [markdown] tags=["ex"]
# Jak widzieliśmy, sieci neuronowe mają bardzo dużo hiperparametrów. Przeszukiwanie ich grid search'em jest więc niewykonalne, a chociaż random search by działał, to potrzebowałby wielu iteracji, co też jest kosztowne obliczeniowo.
#
# Zaimplementuj inteligentne przeszukiwanie przestrzeni hiperparametrów za pomocą biblioteki [Optuna](https://optuna.org/). Implementuje ona między innymi algorytm Tree Parzen Estimator (TPE), należący do grupy algorytmów typu Bayesian search. Typowo osiągają one bardzo dobre wyniki, a właściwie zawsze lepsze od przeszukiwania losowego. Do tego wystarcza im często niewielka liczba kroków.
#
# Zaimplementuj 3-warstwową sieć MLP, gdzie pierwsza warstwa ma rozmiar ukryty N, a druga N // 2. Ucz ją optymalizatorem Adam przez maksymalnie 300 epok z cierpliwością 10.
#
# Przeszukaj wybrane zakresy dla hiperparametrów:
# - rozmiar warstw ukrytych (N)
# - stała ucząca
# - batch size
# - siła regularyzacji L2
# - prawdopodobieństwo dropoutu
#
# Wykorzystaj przynajmniej 30 iteracji. Następnie przełącz algorytm na losowy (Optuna także jego implementuje), wykonaj 30 iteracji i porównaj jakość wyników.
#
# Przydatne materiały:
# - [Optuna code examples - PyTorch](https://optuna.org/#code_examples)
# - [Auto-Tuning Hyperparameters with Optuna and PyTorch](https://www.youtube.com/watch?v=P6NwZVl8ttc)
# - [Hyperparameter Tuning of Neural Networks with Optuna and PyTorch](https://towardsdatascience.com/hyperparameter-tuning-of-neural-networks-with-optuna-and-pytorch-22e179efc837)
# - [Using Optuna to Optimize PyTorch Hyperparameters](https://medium.com/pytorch/using-optuna-to-optimize-pytorch-hyperparameters-990607385e36)

# %% tags=["ex"]
