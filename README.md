# zdo-varroa-detector

Computer vision platform for Varroa mite detection

## Návod ke spuštění

Program spouštěte pomocí skriptu **run.py** s následujícími argumenty:

1. cesta k zdrojové složce (s obrázky ve formátu JPG)

2. cesta k výstupní složce ve které budou pouze výstupní soubory .npy (2D binary ndarray se segmentací)

## Návod ke spuštění evaluace

Vyhodnocení můžete spustit pomocí skriptu **eval.py** s následujícími argumenty:

1. cesta ke složce s referenčními anotacemi (.png formát)

2. cesta ke složce s predikcemi algoritmu (2D binary ndarray, shodná s výstupní složkou z run.py)

## Návod ke spuštění testu

Testovací scénáře se nacházejí ve scriptu **eval_test.py**