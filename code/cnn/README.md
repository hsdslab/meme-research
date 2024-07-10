Ez a mappa tartalmazza a 2 fejű CNN osztályozóhoz tartozó kódokat.

A scriptek futtatásához szükséges könyvtárak listája a requirements.txt fájlban található.

A modellel való inferáláshoz jó példa az `sm-inference.ipnyb` notebook.

# Tartalom

### models mappa:
- binary.py: bináris (mém/nemmém) osztályozó 
- multi.py: többosztályos (sablon) osztályozó
- combined.py: bináris és többosztályos osztályozó egyesítése

### notebooks mappa:
- sm-inference.ipynb: inferálás a modellel
- inference-on-sm-1000.ipynb: inferálás a 1000 képes adathalmazon (amit manuálisan is kiértékeltünk)
- upload_to_hf.ipynb: modell feltöltése HuggingFace-re

### preprocessing mappa:

- dataloaders.py: Dataset és DataModule osztályok a modellekhez
- mean_std_comp.py: Pixel átlag és szórás számítása az adathalmazra az RGB csatornákon
- mean_std.txt: Az adathalmaz kiszámolt pixel átlag és szórása az RGB csatornákon

### training mappa:
- binary_cross_val_main.py: bináris osztályozó keresztvalidációval történő tanítása
- cross_val_main.py: többosztályos osztályozó keresztvalidációval történő tanítása

