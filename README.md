

````markdown name=README.md
# Face Detection & Recognition (face_detection15D.py)

Αυτό το repository περιέχει το script `face_detection15D.py` — ένα πρόγραμμα ανίχνευσης και αναγνώρισης προσώπων που χρησιμοποιεί YOLO (Ultralytics) για ανίχνευση και ArcFace (ONNX) για embeddings.

Περιεχόμενα:
- face_detection15D.py — κύριο script
- .gitignore — προτεινόμενα patterns για να μην ανεβάσεις μεγάλα αρχεία / προσωπικά δεδομένα

Προσοχή πριν το ανέβασμα:
- Το script περιέχει απόλυτο path για `KNOWN_FACES_DIR` (π.χ. r"C:\Users\..."). Ενημέρωσέ το ή μετακίνησε σε σχετική διαδρομή.
- Μην ανεβάσεις αρχεία δεδομένων ή μοντέλα (όπως `models/arcface.onnx`, `models/yolov8n-face-lindevs.pt`, `Known_faces/`, `Unknown_snapshots/`, `facebank.npz`) στο δημόσιο repo. Χρησιμοποίησε `.gitignore` και Git LFS αν χρειάζεται.
- Έλεγξε για ευαίσθητα δεδομένα (API keys, προσωπικά paths, κλπ) πριν το push.

Απαιτήσεις (π.χ. pip):
- Python 3.8+
- opencv-python
- onnxruntime
- ultralytics
- torch
- numpy
- scipy
- scikit-learn
- pillow

Πώς να τρέξεις το script:
1. Βεβαιώσου ότι έχεις εγκαταστήσει τις απαιτούμενες βιβλιοθήκες.
2. Τοποθέτησε τα μοντέλα στο φάκελο `models/`.
3. Τρέξε:
   python face_detection15D.py
4. Ακολούθησε τις επιλογές στο τερματικό για πηγή βίντεο και διαχείριση αγνώστων προσώπων.

Προτεινόμενες βελτιώσεις:
- Μεταφορά ρυθμίσεων και paths σε ένα εξωτερικό `configuration.py` ή `.env`.
- Χρήση Git LFS για μεγάλα μοντέλα.
- Refactor για modularity (διαχωρισμός I/O, model wrapper, UI).