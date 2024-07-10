Ez a mappa tartalmazza a [pHash cikkben](https://arxiv.org/abs/1805.12512) szereplő módszer megvalósítását.



A scriptek/notebookok a következő sorrendben futtathatóak:
1. `calculate_phash.py`: A képek pHash értékének kiszámítása.
2. `preprocessing.ipynb`: A képek felosztása klaszterezendő illetve annotáló részhalmazokra.
3. `cluster_phases.py`: A képek klaszterezése phashük alapján HDBSCAN algoritmus használatával.
4. `annotate_clusters.py`: A klaszterek annotálása a félrerakott annotáló képek alapján.
5. `evaluation.ipynb`: A klaszterezés eredményének kiértékelése.