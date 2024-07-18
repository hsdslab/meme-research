A Sparse matching [cikkhez](https://dl.acm.org/doi/abs/10.1145/3178876.3186021) tartozó kódok a `sparse-matching` mappában találhatók. A mappában a következő fájlok vannak:
- `FaceRecognition_SparseRepresentation.ipynb`: A cikkben hivatkozott arcfelismerő algoritmus implementációja, amit ők is használtak.
- `sparse-matching.ipynb`: A saját implementációnk a cikkben leírt algoritmusnak, néhány művelet vektorizálva lett + sima Lasso helyett LassoLars-ot használtunk a skálázható futtatás érdekében.
- `inference.ipynb`: A saját implementációnkkal való inferálás a mémes adathalmazon. 
- `inference.py`: A saját implementációnkkal való inferálás script formában. Ezzel a megoldással végül csak 108294 képet inferáltunk a teljes SM adathalmazból, egyrészt mert a futási idő másrészt mert láttuk, hogy nem ez lesz a befutó modell és 1000 kép is elég volt a manuális kiértékeléshez.
- `sm-prediction.ipynb`: A predikciók mellé még hozzácsaptam a sourcet (Facebook, Reddit, Twitter).

DISCLAIMER: ezeket a fájlokat nem volt időm rendesen refaktorálni.