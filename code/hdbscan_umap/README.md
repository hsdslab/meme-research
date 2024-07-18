A HDBSCAN+UMAP+{valamilyen beágyazás} kísérletekhez tartozó kódok a `hdbscan_umap` mappában találhatók. A mappában a következő fájlok vannak:
- `bertopic.ipynb`: A futtatásokhoz a BERTopic pipelineját vettük alapul. Kicsit meg vannak keverve benne a dolgok tekintve hogy az eval szekció előbb van mint a tanítás de ez ne zavarjon senkit. A starttól kell kezdeni, mint minden normális board game-nél. A deprecateddel jelölt részt nem kell futtatni, az még a KYM-es adathaalmazra vonatkozott.
- `inference-on-labeled-sm.ipynb`: A fittelt Bertopic modellt használva berakosgatjuk a manuálisan címkézett képeket a klaszterekbe.

DISCLAIMER: ezeket a fájlokat nem volt időm rendesen refaktorálni.