Ez a mappa tartalmazza a Courtois & Frissen (cikke)[https://www.tandfonline.com/doi/full/10.1080/19312458.2022.2122423] által inspirált megoldásunk kódját. Végül csak a feature matching részét használtuk fel a megoldásuknak és a feature distancekből képzett image distanceket egy RNN segítségével osztályoztuk. A kódok a következők:
1. `clean_images.py`: Az eredeti cikk szerzői leírták hogy a mémek esetében fontos ennek a feature matching algoritmusnak hogy a rajtuk levő szövegek el legyenek homályosítva, hogy ne a szövegek hasonló betűtípusa miatt legyenek matchek a képek között. Ennek a sciptnek a futtatásához szükség van egy szöveg szegmentáló modellre ami itt letöltehető: https://github.com/oyyd/frozen_east_text_detection.pb2
2. `distance-nb.ipynb`: A feature distancek kiszámítása a blurrölt képeken. Nagyon lassú a futtatása, mert minden képet minden képpel összehasonlít, ezért a kimenetét elmentettük.
3. `store-res-in-parquet.ipynb`: Az előző notebook kimenetét (txt fájlok) fogja össze parquet fájlokba.
4. `aggregate-matches.ipynb`: A képek közti featureök távolsága alapján származtat egy aggregált távolságot a két kép között.
5. `rnn.ipynb`: Az aggregált távolságok alapján egy RNN segítségével osztályozza a képeket.

DISCLAIMER: ezeket a fájlokat nem volt időm rendesen refaktorálni.