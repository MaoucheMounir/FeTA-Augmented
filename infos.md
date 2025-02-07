# Fichiers modifiés par Mounir:
3D_UNet_1dataloader_1label.py
3D_U-Net_inference.py

# Fichiers Créés par Mounir:
- data.py
- eveluation.py
- loggingFunctions.py
- modelClasses.py
- variables.py
Ces fichiers contiennent dese fonctions qui étaient auparavant contenus à l'intérieur des scripts, ainsi que les fonctions que j'ai créées liées au log et aux transformations de données avec torchIO

- visualize_results.ipynb

# Dossiers créés par Mounir
- Cropped Data 2D et 3D: Certaines données n'étaient pas prétraitées (on peut le voir aux images avec de petits cerveaux et de gros bords noirs), je les ai donc toutes prétraitées.

- FeTA_full_test_mounir: contient le contenu de FeTA_2024_test et FeTA_2022_test combinés et prétraités

- Garance: J'ai déplacé les dossiers "runs" et "weights" de garance pour qu'il n'y ait pas de confusion



- logs: contient les fichiers de logs des entrainement (produits par le fichier 3D_UNet_1dataloader_1label.py) avec tous les outputs du terminal et les eventuelles erreurs
- results_inference: contient les fichiers de log et les resultats (segmentations, scores...) produits lors de l'inférence avec le fichier 3D_U-Net_inference.py
- runs: Contient les fichiers propres à TensorBoard. Généré par le fichier d'entraînement  3D_UNet_1dataloader_1label.py
- weights_mounir: contient les poids des modèles enregistrés. Généré par le fichier d'entraînement 3D_UNet_1dataloader_1label.py

# Reste des fichiers
Inchangés par Mounir, restés tels que créés par Garance
