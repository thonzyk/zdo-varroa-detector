# zdo-varroa-detector

Computer vision platform for Varroa mite detection

## Návod ke spuštění

Program spouštěte pomocí skriptu **run.py** s následujícími argumenty:

1. cesta k zdrojové složce (s obrázky ve formátu JPG)

2. cesta k výstupní složce (2D binary ndarray se segmentací)

## Návod ke spuštění evaluace

Vyhodnocení můžete spustit pomocí skriptu **eval.py** s následujícími argumenty:

1. ground true segmentace (.png formát)

2. predikce algoritmu (2D binary ndarray)

## Návod ke spuštění testu

Testovací scénáře se nacházejí ve scriptu **eval_test.py**