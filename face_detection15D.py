import os
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import torch
import json
import datetime
import sys
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import threading
import time

from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine

# Προσπάθεια φόρτωσης ρυθμίσεων από το αρχείο configuration.py
try:
    import configuration
    print("[CONFIG] Φορτώθηκαν ρυθμίσεις από το configuration.py")
    CONFIG = configuration.load_settings()
except ImportError:
    print("[CONFIG] Δεν βρέθηκε το αρχείο configuration.py, χρήση προεπιλεγμένων ρυθμίσεων")
    CONFIG = {}

# ======== Σταθερή αντιστοίχιση indexes ========
INDEX_PC = 1
INDEX_USB = 2
INDEX_DROIDCAM = 0

VIDEO_DIR = "video"  # <-- προσθήκη αυτής της γραμμής κοντά στην αρχή, πριν το main

def initialize_capture(idx_or_url):
    cap = cv2.VideoCapture(idx_or_url)
    if cap.isOpened():
        print(f"[OK] ✅ Camera ({idx_or_url}) άνοιξε!")
        return cap
    print(f"[ERROR] ❌ Camera ({idx_or_url}) ΔΕΝ άνοιξε.")
    return None

def try_open(index, label):
    cap = initialize_capture(index)
    if cap:
        print(f"[INFO] ➡️ '{label}' επιλέχθηκε και άνοιξε με index {index}.")
    else:
        print(f"[ERROR] ❌ Η '{label}' (index {index}) δεν είναι διαθέσιμη.")
    return cap

class ArcFaceONNX:
    def __init__(self, model_path, use_gpu=False):
        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def get_embedding(self, img):
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        embedding = self.session.run(None, {self.input_name: img})[0][0]
        return embedding / np.linalg.norm(embedding)

# === Βοηθητικές συναρτήσεις για meta δεδομένα ===
def compute_faces_meta(known_faces_dir, valid_exts={".jpg", ".jpeg", ".png"}):
    faces_meta = {}
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        images = []
        for file in os.listdir(person_dir):
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_exts:
                images.append(file)
        if images:
            faces_meta[person_name] = sorted(images)
    return faces_meta

def load_saved_meta(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_meta(meta, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

# === Facebank ===
def load_facebank(known_faces_dir, arcface, valid_exts={".jpg", ".jpeg", ".png"}):
    facebank = {}
    print(f"[INFO] Διαβάζονται και υπολογίζονται embeddings για όλα τα γνωστά πρόσωπα από: {os.path.abspath(known_faces_dir)}")
    if not os.path.exists(known_faces_dir):
        print(f"[ERROR] Ο φάκελος γνωστών προσώπων δεν υπάρχει: {os.path.abspath(known_faces_dir)}")
        return facebank
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        embeddings = []
        for file in os.listdir(person_dir):
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_exts:
                img_path = os.path.join(person_dir, file)
                img = cv2.imread(img_path)
                if img is not None:
                    emb = arcface.get_embedding(img)
                    embeddings.append(emb)
                else:
                    print(f"[WARN] Δεν διαβάστηκε εικόνα: {img_path}")
            elif ext == ".npy":
                try:
                    emb = np.load(os.path.join(person_dir, file))
                    embeddings.append(emb)
                except Exception as e:
                    print(f"[WARN] Σφάλμα στο .npy embedding {file}: {e}")
        if embeddings:
            facebank[person_name] = embeddings
        else:
            print(f"[WARN] Δεν βρέθηκαν έγκυρες εικόνες ή embeddings για: {person_name}")
    print(f"[INFO] Ολοκληρώθηκε η δημιουργία embeddings για όλα τα γνωστά πρόσωπα. Βρέθηκαν: {list(facebank.keys())}")
    return facebank

# --- ΜΟΝΟ ΜΙΑ ΥΛΟΠΟΙΗΣΗ add_to_facebank με face_crop=None ---
def add_to_facebank(facebank, name, emb, face_crop=None):
    if name not in facebank:
        facebank[name] = []
    facebank[name].append(emb)
    print(f"[INFO] Facebank τώρα περιέχει {len(facebank[name])} embeddings για '{name}'.")
    
    # Δημιουργία φακέλου για το πρόσωπο εάν δεν υπάρχει
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Αποθήκευση του embedding
    emb_path = os.path.join(person_dir, f"embedding_{len(facebank[name])}.npy")
    np.save(emb_path, emb)
    print(f"[SYNC] Το embedding αποθηκεύτηκε και στον φάκελο Known_faces: {emb_path}")
    
    # Αποθήκευση της εικόνας αν παρέχεται
    if face_crop is not None:
        # Βεβαιωνόμαστε ότι η εικόνα είναι υψηλής ποιότητας
        enhanced_face = enhance_face_image(face_crop)
        
        # Έλεγχος μεγέθους για καλή ποιότητα
        h, w = enhanced_face.shape[:2]
        min_save_size = 200  # Ελάχιστο μέγεθος για αποθήκευση
        if min(h, w) < min_save_size:
            scale = min_save_size / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            enhanced_face = cv2.resize(enhanced_face, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Αποθήκευση σε διαφορετικές μορφές για βέλτιστη χρήση
        
        # 1. Αποθήκευση JPEG για συμβατότητα (υψηλή ποιότητα)
        img_path = os.path.join(person_dir, f"face_{len(facebank[name])}.jpg")
        cv2.imwrite(img_path, enhanced_face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[SYNC] Το βελτιωμένο πρόσωπο αποθηκεύτηκε στον φάκελο Known_faces: {img_path}")
        
        # 2. Αποθήκευση PNG για μέγιστη ποιότητα χωρίς απώλειες
        img_path_png = os.path.join(person_dir, f"face_{len(facebank[name])}_highquality.png")
        cv2.imwrite(img_path_png, enhanced_face)
        
        # 3. Αποθήκευση της αρχικής μη επεξεργασμένης εικόνας για μελλοντική χρήση
        raw_path = os.path.join(person_dir, f"face_{len(facebank[name])}_original.png")
        cv2.imwrite(raw_path, face_crop)
        
        print(f"[QUALITY] Αποθηκεύτηκαν 3 εκδόσεις του προσώπου: βελτιωμένη JPG, υψηλή ποιότητα PNG, και αρχική")
        
        # Καταγραφή μεταδεδομένων για το πρόσωπο
        meta_file = os.path.join(person_dir, "face_info.txt")
        with open(meta_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n--- Face #{len(facebank[name])} | {timestamp} ---\n")
            f.write(f"Resolution: {w}x{h} -> {enhanced_face.shape[1]}x{enhanced_face.shape[0]}\n")
            f.write(f"Mean brightness: {np.mean(enhanced_face):.2f}\n")
            f.write(f"Contrast (std): {np.std(enhanced_face):.2f}\n")
            f.write(f"Files: {os.path.basename(img_path)}, {os.path.basename(img_path_png)}, {os.path.basename(raw_path)}\n")
            
        print(f"[META] Καταγράφηκαν πληροφορίες για το πρόσωπο στο {meta_file}")

def save_facebank(facebank, path):
    # Save as npz for speed (not human-readable)
    np.savez_compressed(path, **{k: np.stack(v) for k, v in facebank.items()})

def load_facebank_npz(path):
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=True)
    # Μετατροπή κάθε entry σε λίστα
    return {k: [emb for emb in data[k]] for k in data.files}

def recognize_face(emb, facebank, threshold=0.5):
    best_name = "Unknown"
    best_score = -1
    for name, embs in facebank.items():
        for ref_emb in embs:
            score = 1 - cosine(emb, ref_emb)
            if score > best_score:
                best_score = score
                best_name = name
    if best_score < threshold:
        return "Unknown", best_score
    return best_name, best_score

# --- Ορισμός threshold για κάθε επιλογή ξεχωριστά ---
DEFAULT_THRESHOLD_1 = 0.5
DEFAULT_THRESHOLD_2 = 0.32
DEFAULT_THRESHOLD_3 = 0.5

THRESHOLD_FILE_1 = "threshold_1.txt"
THRESHOLD_FILE_2 = "threshold_2.txt"
THRESHOLD_FILE_3 = "threshold_3.txt"

def get_threshold(choice):
    # Προτεραιότητα: argument > αρχείο > default ανά επιλογή
    threshold_label = "1"
    if len(sys.argv) > 1:
        try:
            t = float(sys.argv[1])
            print(f"[CONFIG] Χρήση threshold από argument: {t}")
            return t, "arg"
        except Exception:
            print(f"[CONFIG] Μη έγκυρο threshold argument, χρησιμοποιείται default.")
    cwd = os.getcwd()  # ΜΟΝΟ το current working directory (εκεί που τρέχει το script)
    if choice == "1":
        candidates = [os.path.join(cwd, THRESHOLD_FILE_1)]
        default = DEFAULT_THRESHOLD_1
        threshold_label = "1"
    elif choice == "2":
        candidates = [os.path.join(cwd, THRESHOLD_FILE_2)]
        default = DEFAULT_THRESHOLD_2
        threshold_label = "2"
    elif choice == "3":
        candidates = [os.path.join(cwd, THRESHOLD_FILE_3)]
        default = DEFAULT_THRESHOLD_3
        threshold_label = "3"
    else:
        candidates = []
        default = DEFAULT_THRESHOLD_1
        threshold_label = "1"

    found = False
    for threshold_file in candidates:
        print(f"[DEBUG] Αναζήτηση threshold στο: {threshold_file}")
        if os.path.exists(threshold_file):
            try:
                with open(threshold_file, "r", encoding="utf-8") as f:
                    value = f.read().strip().replace(",", ".")
                    print(f"[DEBUG] Περιεχόμενο threshold αρχείου: '{value}'")
                    t = float(value)
                    print(f"[CONFIG] Χρήση threshold από αρχείο {threshold_file}: {t}")
                    return t, threshold_label
            except Exception as e:
                print(f"[CONFIG] Μη έγκυρο threshold στο αρχείο {threshold_file} ({e}), χρησιμοποιείται default: {default}")
            found = True
            break
    if not found:
        print(f"[WARN] Δεν βρέθηκε threshold αρχείο σε καμία διαδρομή: {candidates}")
    print(f"[CONFIG] Χρήση default threshold: {default}")
    return default, threshold_label

# === Μοντέλα ===
arcface = ArcFaceONNX("models/arcface.onnx", use_gpu=torch.cuda.is_available())
model = YOLO("models/yolov8n-face-lindevs.pt")
# Χρησιμοποίησε absolute path για Known_faces
KNOWN_FACES_DIR = r"C:\Users\dioni\Desktop\YOLO PROJECT\Known_faces"
FACEBANK_PATH = "facebank.npz"
valid_exts = {".jpg", ".jpeg", ".png"}
UNKNOWN_SNAPSHOT_DIR = "Unknown_snapshots"
os.makedirs(UNKNOWN_SNAPSHOT_DIR, exist_ok=True)

# Φόρτωση του facebank κατά την αρχικοποίηση
facebank = {}
if os.path.exists(FACEBANK_PATH):
    facebank = load_facebank_npz(FACEBANK_PATH)
    print(f"[INFO] Φορτώθηκε facebank με {len(facebank)} πρόσωπα: {list(facebank.keys())}")
else:
    print(f"[INFO] Δεν βρέθηκε αποθηκευμένο facebank στο {FACEBANK_PATH}, δημιουργία νέου.")

# === Ρυθμίσεις συμπεριφοράς αναγνώρισης ===
# Φόρτωση από configuration ή χρήση προεπιλεγμένων τιμών
AUTO_IGNORE_CLOSE_SCORES = CONFIG.get('AUTO_IGNORE_CLOSE_SCORES', True)  # Αυτόματη παράλειψη άγνωστων προσώπων με score κοντά στο threshold
IGNORE_MARGIN = CONFIG.get('IGNORE_MARGIN', 0.02)  # Περιθώριο για αυτόματη παράλειψη (|threshold - score| < IGNORE_MARGIN)
ENHANCED_UI = CONFIG.get('ENHANCED_UI', True)  # Χρήση βελτιωμένου γραφικού περιβάλλοντος για άγνωστα πρόσωπα
MIN_FACE_SIZE = CONFIG.get('MIN_FACE_SIZE', 60)  # Ελάχιστο μέγεθος προσώπου για καλύτερη ποιότητα

# === Ρυθμίσεις ποιότητας εικόνας ===
HIGH_QUALITY_IMAGES = CONFIG.get('HIGH_QUALITY_IMAGES', True)  # Ενεργοποίηση για αποθήκευση εικόνων υψηλής ποιότητας
MIN_IMAGE_SAVE_SIZE = CONFIG.get('MIN_IMAGE_SAVE_SIZE', 200)  # Ελάχιστο μέγεθος για αποθήκευση εικόνων προσώπων (px)
MIN_BRIGHTNESS = CONFIG.get('MIN_BRIGHTNESS', 50)  # Ελάχιστη φωτεινότητα για αποδεκτή εικόνα
MIN_CONTRAST = CONFIG.get('MIN_CONTRAST', 30)  # Ελάχιστη αντίθεση για αποδεκτή εικόνα

# Εμφάνιση των τρεχουσών ρυθμίσεων
print(f"[CONFIG] AUTO_IGNORE_CLOSE_SCORES: {AUTO_IGNORE_CLOSE_SCORES}")
print(f"[CONFIG] IGNORE_MARGIN: {IGNORE_MARGIN}")
print(f"[CONFIG] ENHANCED_UI: {ENHANCED_UI}")
print(f"[CONFIG] MIN_FACE_SIZE: {MIN_FACE_SIZE}")
print(f"[CONFIG] HIGH_QUALITY_IMAGES: {HIGH_QUALITY_IMAGES}")
print(f"[CONFIG] MIN_IMAGE_SAVE_SIZE: {MIN_IMAGE_SAVE_SIZE}")
print(f"[CONFIG] MIN_BRIGHTNESS: {MIN_BRIGHTNESS}")
print(f"[CONFIG] MIN_CONTRAST: {MIN_CONTRAST}")

# === Διαχείριση δυναμικού threshold ===
DYNAMIC_THRESHOLD = False  # Προεπιλεγμένη τιμή
last_threshold = None

try:
    from threshold import get_recommended_threshold, adjust_threshold
    DYNAMIC_THRESHOLD = True
    print("[INFO] Φορτώθηκε το module δυναμικού threshold")
except ImportError:
    print("[INFO] Το module δυναμικού threshold δεν βρέθηκε - χρήση σταθερών τιμών")

# Βελτιωμένη συνάρτηση για βελτίωση ποιότητας εικόνας προσώπου
def enhance_face_image(face_crop):
    """Βελτιώνει την ποιότητα της εικόνας του προσώπου για καλύτερη εμφάνιση
    Εφαρμόζει προηγμένες τεχνικές βελτίωσης αντίθεσης, χρωμάτων, αφαίρεσης θορύβου
    και υπερανάλυσης για καλύτερη ποιότητα αποθήκευσης."""
    try:
        if face_crop is None or face_crop.size == 0:
            print("[ERROR] Κενό face_crop στη συνάρτηση enhance_face_image")
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        # Έλεγχος αν το μέγεθος είναι πολύ μικρό
        h, w = face_crop.shape[:2]
        scale = max(1.0, MIN_FACE_SIZE / min(h, w))
        if scale > 1.0:
            # Χρήση INTER_CUBIC για καλύτερη ποιότητα σε μεγέθυνση
            face_crop = cv2.resize(face_crop, (int(w * scale), int(h * scale)), 
                                 interpolation=cv2.INTER_CUBIC)
        
        # Έλεγχος αν το crop είναι πολύ σκοτεινό ή χαμηλής αντίθεσης
        mean_val = np.mean(face_crop)
        std_val = np.std(face_crop)
        
        # Δημιουργία αντιγράφου για επεξεργασία
        enhanced = face_crop.copy()
        
        # Αφαίρεση θορύβου πριν τις βελτιώσεις για καλύτερα αποτελέσματα
        # Non-Local Means Denoising για προσεκτική αφαίρεση θορύβου χωρίς απώλεια λεπτομερειών
        if std_val < 50:  # Αν η εικόνα έχει θόρυβο (χαμηλή αντίθεση συχνά σημαίνει περισσότερο θόρυβο)
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 15)
        
        # Βελτίωση φωτεινότητας για σκοτεινές εικόνες - Εξελιγμένο
        if mean_val < 90:  # Αυξήσαμε το κατώφλι για να πιάνει περισσότερες σκοτεινές εικόνες
            # Μετατροπή σε HSV και αύξηση της τιμής (Value) για περισσότερη φωτεινότητα
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Δυναμική κλιμάκωση της τιμής V για αύξηση φωτεινότητας
            target_brightness = 140  # Στόχος μέσης φωτεινότητας
            factor = min(2.0, target_brightness / max(1, mean_val))  # Ανώτατο όριο για αποφυγή υπερβολικής φωτεινότητας
            
            # Προσθετική αύξηση για πολύ σκοτεινές περιοχές
            beta_value = 0
            if mean_val < 60:
                beta_value = 20  # Μεγαλύτερη αύξηση για πολύ σκοτεινές εικόνες
            
            v = cv2.convertScaleAbs(v, alpha=factor, beta=beta_value)
            
            # Συγχώνευση καναλιών και επιστροφή σε BGR
            hsv = cv2.merge([h, s, v])
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Βελτίωση αντίθεσης με προσαρμοστικό CLAHE
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Δυναμική προσαρμογή του clipLimit βάσει της τυπικής απόκλισης
        clip_limit = 3.0
        if std_val < 30:
            clip_limit = 4.0  # Μεγαλύτερο clip limit για εικόνες με χαμηλή αντίθεση
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Βελτίωση χρωματικών καναλιών για πιο ζωντανά χρώματα
        # Αυτό βοηθά ειδικά σε χαμηλής ποιότητας κάμερες με ξεθωριασμένα χρώματα
        a = cv2.convertScaleAbs(a, alpha=1.1, beta=0)  # Ελαφριά ενίσχυση πράσινου-κόκκινου καναλιού
        b = cv2.convertScaleAbs(b, alpha=1.1, beta=0)  # Ελαφριά ενίσχυση μπλε-κίτρινου καναλιού
        
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Εξελιγμένο sharpening με προσαρμοζόμενο πυρήνα
        if std_val < 40:  # Χαμηλή αντίθεση
            # Ισχυρότερο sharpening για χαμηλής αντίθεσης εικόνες
            kernel = np.array([[-1, -1, -1], 
                              [-1,  9, -1],
                              [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        else:
            # Πιο ήπιο sharpening για εικόνες με ικανοποιητική αντίθεση
            kernel = np.array([[-0.5, -0.5, -0.5], 
                              [-0.5,  5, -0.5],
                              [-0.5, -0.5, -0.5]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Αύξηση της ανάλυσης για καλύτερη ποιότητα αποθήκευσης - Super Resolution προσομοίωση
        # Αυτό βοηθά ειδικά για μικρά πρόσωπα από κάμερες χαμηλής ανάλυσης
        h, w = enhanced.shape[:2]
        if min(h, w) < 100:  # Αν η εικόνα είναι μικρή, αύξησε την ανάλυση
            sr_scale = min(2.0, 120 / min(h, w))  # Μέγιστο διπλασιασμό, αλλά όχι πάνω από 120px
            if sr_scale > 1.1:  # Αν χρειάζεται σημαντική αύξηση
                new_h, new_w = int(h * sr_scale), int(w * sr_scale)
                # Χρησιμοποιούμε INTER_CUBIC για προσομοίωση υπερανάλυσης
                enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                # Εφαρμογή ενός ήπιου φίλτρου ενίσχυσης άκρων μετά την αύξηση μεγέθους
                # για βελτίωση των λεπτομερειών
                kernel = np.array([[0, -0.25, 0], 
                                  [-0.25, 2, -0.25],
                                  [0, -0.25, 0]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Τελικό φίλτρο μείωσης θορύβου - πολύ ελαφρύ για διατήρηση λεπτομερειών
        # Bilateral filter διατηρεί τις άκρες ενώ αφαιρεί θόρυβο
        enhanced = cv2.bilateralFilter(enhanced, 5, 25, 25)
        
        return enhanced
    except Exception as e:
        print(f"[ERROR] Σφάλμα κατά την επεξεργασία της εικόνας: {str(e)}")
        # Σε περίπτωση σφάλματος, επιστρέφουμε την αρχική εικόνα
        return face_crop

# Νέα συνάρτηση για εμφάνιση προσώπου με βελτιωμένη αξιοπιστία
def show_face_for_training(face_image):
    """
    Εμφανίζει το πρόσωπο σε ξεχωριστό παράθυρο με έμφαση στην αξιοπιστία εμφάνισης.
    Χρησιμοποιεί πολλαπλές τεχνικές για να διασφαλίσει ότι το παράθυρο εμφανίζεται.
    
    Επιστρέφει: True αν το παράθυρο εμφανίστηκε επιτυχώς, False διαφορετικά.
    """
    try:
        # Δημιουργία βελτιωμένης εικόνας με πληροφορίες
        h, w = face_image.shape[:2]
        display_img = np.zeros((h + 80, w, 3), dtype=np.uint8)
        display_img[0:h, 0:w] = face_image
        
        # Προσθήκη οδηγιών και πληροφοριών
        cv2.putText(display_img, "ΑΓΝΩΣΤΟ ΠΡΟΣΩΠΟ", (10, h + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(display_img, "Δωστε ονομα στο τερματικο", (10, h + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Διαγραφή παλιών παραθύρων με το ίδιο όνομα (για καλό και για κακό)
        try:
            cv2.destroyWindow("Εκπαιδευση Προσωπου")
            cv2.waitKey(1)
        except:
            pass
            
        # Δημιουργία παραθύρου με συγκεκριμένες ιδιότητες
        cv2.namedWindow("Εκπαιδευση Προσωπου", cv2.WINDOW_NORMAL)
        
        # Αλλαγή μεγέθους για καλύτερη εμφάνιση
        cv2.resizeWindow("Εκπαιδευση Προσωπου", w, h+80)
        
        # Εμφάνιση της εικόνας
        cv2.imshow("Εκπαιδευση Προσωπου", display_img)
        
        # Μετακίνηση στο προσκήνιο
        try:
            cv2.setWindowProperty("Εκπαιδευση Προσωπου", cv2.WND_PROP_TOPMOST, 1)
        except:
            pass
            
        # Μετακίνηση παραθύρου σε καλή θέση
        cv2.moveWindow("Εκπαιδευση Προσωπου", 100, 100)
        
        # Πολλαπλές ανανεώσεις για να εξασφαλίσουμε την εμφάνιση
        for _ in range(10):
            cv2.imshow("Εκπαιδευση Προσωπου", display_img)
            cv2.waitKey(10)
            
        # Μικρή καθυστέρηση για την εμφάνιση
        time.sleep(0.5)
        
        # Τελευταία ανανέωση
        cv2.imshow("Εκπαιδευση Προσωπου", display_img)
        cv2.waitKey(1)
        
        return True
    except Exception as e:
        print(f"[ERROR] Σφάλμα κατά την εμφάνιση παραθύρου: {str(e)}")
        return False

# Νέα συνάρτηση για γραφικό περιβάλλον επιβεβαίωσης άγνωστου προσώπου
def show_unknown_face_dialog(face_crop, score, threshold):
    """Εμφανίζει γραφικό διάλογο για επιβεβαίωση άγνωστου προσώπου
    Χρησιμοποιεί απλό παράθυρο OpenCV αντί για tkinter για αποφυγή προβλημάτων με threads.
    Εμφανίζει εικόνα υψηλής ποιότητας με επιπλέον πληροφορίες.
    """
    try:
        # Έλεγχος ποιότητας εικόνας πριν τη βελτίωση
        mean_val = np.mean(face_crop)
        std_val = np.std(face_crop)
        quality_issues = []
        
        if mean_val < MIN_BRIGHTNESS:
            quality_issues.append(f"Χαμηλή φωτεινότητα ({mean_val:.1f}<{MIN_BRIGHTNESS})")
        if std_val < MIN_CONTRAST:
            quality_issues.append(f"Χαμηλή αντίθεση ({std_val:.1f}<{MIN_CONTRAST})")
        if min(face_crop.shape[0], face_crop.shape[1]) < MIN_FACE_SIZE:
            quality_issues.append(f"Μικρό μέγεθος ({min(face_crop.shape[0], face_crop.shape[1])}px<{MIN_FACE_SIZE}px)")
            
        # Μετατροπή εικόνας σε βελτιωμένη έκδοση
        enhanced_face = enhance_face_image(face_crop)
        
        # Επιπλέον βελτιώσεις για το παράθυρο διαλόγου
        if min(enhanced_face.shape[0], enhanced_face.shape[1]) < MIN_IMAGE_SAVE_SIZE:
            h, w = enhanced_face.shape[:2]
            scale = MIN_IMAGE_SAVE_SIZE / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            enhanced_face = cv2.resize(enhanced_face, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Δημιουργία εικόνας με επεξηγηματικό κείμενο και πληροφορίες ποιότητας
        h, w = enhanced_face.shape[:2]
        
        # Προσθήκη περισσότερου χώρου για πληροφορίες ποιότητας
        info_height = 150
        if quality_issues:
            info_height += 30  # Επιπλέον χώρος για προβλήματα ποιότητας
            
        display_img = np.zeros((h + info_height, w, 3), dtype=np.uint8)
        display_img[0:h, 0:w] = enhanced_face
        
        # Προσθήκη κειμένου με περισσότερες πληροφορίες
        line_y = h + 30
        cv2.putText(display_img, "ΑΓΝΩΣΤΟ ΠΡΟΣΩΠΟ", (10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        line_y += 30
        
        # Πληροφορίες ποιότητας εικόνας
        quality_color = (0, 255, 0)  # Πράσινο αν είναι καλή η ποιότητα
        if quality_issues:
            quality_color = (0, 0, 255)  # Κόκκινο αν υπάρχουν προβλήματα
            cv2.putText(display_img, f"Ζητήματα: {', '.join(quality_issues)}", (10, line_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 1)
            line_y += 30
        
        # Τεχνικές πληροφορίες
        cv2.putText(display_img, f"Score: {score:.3f} | Threshold: {threshold:.3f} | Διαφορά: {abs(score-threshold):.3f}", 
                  (10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        line_y += 30
        
        # Μέτρηση ποιότητας εικόνας
        quality_info = f"Ανάλυση: {w}x{h}px | Φωτεινότητα: {mean_val:.1f} | Αντίθεση: {std_val:.1f}"
        cv2.putText(display_img, quality_info, (10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        line_y += 30
        
        # Οδηγίες χρήσης
        cv2.putText(display_img, "A: Προσθηκη | S: Παραλειψη | Q: Εξοδος", (10, line_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        line_y += 30
        cv2.putText(display_img, "Πληκτρολογηστε ονομα & Enter", (10, line_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Εμφάνιση παραθύρου
        window_name = "Αγνωστο Προσωπο"
        cv2.imshow(window_name, display_img)
        cv2.waitKey(1)  # Αυτό είναι απαραίτητο για να δημιουργηθεί το παράθυρο πριν το moveWindow
        
        try:
            cv2.moveWindow(window_name, 100, 100)
        except:
            print("[WARN] Δεν ήταν δυνατή η μετακίνηση του παραθύρου")
        
        # Περιμένουμε για είσοδο από το χρήστη
        name = ""
        action = None
        
        while action is None:
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('a'):  # Προσθήκη
                # Ζητάμε το όνομα με εμφάνιση μηνύματος στο τερματικό
                print("\n➡️ Δώστε όνομα για το άγνωστο πρόσωπο:")
                name = input("Όνομα: ").strip()
                if name:
                    action = "add"
                else:
                    print("⚠️ Δεν δόθηκε όνομα. Συνεχίστε με 'a' για προσθήκη, 's' για παράλειψη ή 'q' για έξοδο.")
            
            elif key == ord('s'):  # Παράλειψη
                action = "skip"
            
            elif key == ord('q'):  # Έξοδος
                action = "exit"
        
        # Κλείσιμο παραθύρου
        try:
            cv2.destroyWindow(window_name)
        except:
            print("[WARN] Δεν ήταν δυνατό το κλείσιμο του παραθύρου")
        
        return action, name
    except Exception as e:
        print(f"[ERROR] Σφάλμα στο show_unknown_face_dialog: {str(e)}")
        # Απλοποιημένη εναλλακτική έκδοση χωρίς γραφικό περιβάλλον
        print("\n➡️ Εντοπίστηκε άγνωστο πρόσωπο!")
        print(f"Score: {score:.2f} | Threshold: {threshold:.2f}")
        print("A: Προσθήκη | S: Παράλειψη | Q: Έξοδος")
        
        while True:
            key_input = input("Επιλογή (a/s/q): ").lower()
            if key_input == 'a':
                name = input("Όνομα: ").strip()
                if name:
                    return "add", name
                else:
                    print("⚠️ Δεν δόθηκε όνομα. Προσπαθήστε ξανά.")
            elif key_input == 's':
                return "skip", ""
            elif key_input == 'q':
                return "exit", ""
            else:
                print("⚠️ Μη έγκυρη επιλογή. Δοκιμάστε ξανά (a/s/q).")

# Αντικατάσταση της συνάρτησης process_frame με νέα έκδοση
def process_frame(frame, cam_name="camera", threshold=0.5, auto_learn=False, threshold_label="1"):
    results = model(frame)
    annotated = frame.copy()
    unknown_faces = []
    
    # Δυναμική προσαρμογή threshold αν είναι ενεργοποιημένη
    global last_threshold
    if DYNAMIC_THRESHOLD:
        if 'last_threshold' not in globals():
            last_threshold = get_recommended_threshold(threshold_label)
            threshold = last_threshold
            print(f"[THRESHOLD] Προτεινόμενο threshold: {threshold:.3f}")
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            conf = float(box.conf[0])
            if conf < 0.5: continue
            
            # Διεύρυνση του bounding box για να συμπεριλάβει περισσότερα από το πρόσωπο
            # Αυτό βοηθά στην καλύτερη αναγνώριση και ποιότητα της εικόνας
            face_width, face_height = x2 - x1, y2 - y1
            # Προσθήκη περιθωρίου 20% γύρω από το πρόσωπο
            margin_x = int(face_width * 0.20)
            margin_y = int(face_height * 0.20)
            
            # Εφαρμογή περιορισμών στα όρια του frame
            y1 = max(0, y1 - margin_y)
            y2 = min(frame.shape[0], y2 + margin_y)
            x1 = max(0, x1 - margin_x)
            x2 = min(frame.shape[1], x2 + margin_x)
            
            if y2 <= y1 or x2 <= x1: continue
            face_crop = frame[y1:y2, x1:x2]            # Έλεγχοι ποιότητας εικόνας
            crop_mean = np.mean(face_crop)
            crop_std = np.std(face_crop)
            if crop_mean < 30 or crop_std < 15:
                print(f"[DEBUG] Σκοτεινό/άδειο crop (mean={crop_mean:.2f}, std={crop_std:.2f}) bbox=({x1},{y1},{x2},{y2})")
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, "Dark/Empty", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                continue

            # Έλεγχος αν το frame είναι BGR, BGRA ή grayscale
            if len(face_crop.shape) == 2:
                print("[WARN] Το crop είναι grayscale!")
            elif face_crop.shape[-1] == 4:
                print("[WARN] Το crop είναι BGRA, μετατροπή σε BGR.")
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGRA2BGR)
            elif face_crop.shape[-1] == 3:
                pass  # BGR, OK
            else:
                print(f"[WARN] Άγνωστος αριθμός καναλιών: {face_crop.shape}")

            # Έλεγχος για πολύ σκοτεινό ή "άδειο" crop
            if np.mean(face_crop) < 10 or np.std(face_crop) < 10:
                print(f"[WARN] Πολύ σκοτεινό ή ομοιόμορφο crop (mean={np.mean(face_crop):.2f}, std={np.std(face_crop):.2f})")
                print(f"Bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                continue
            if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10: continue
            try:
                emb = arcface.get_embedding(face_crop)
                emb = normalize([emb])[0]
            except Exception: continue
            name, score = recognize_face(emb, facebank, threshold=threshold)
            
            # Δυναμική προσαρμογή threshold
            if DYNAMIC_THRESHOLD and score > 0.2:  # Αγνοούμε πολύ χαμηλά scores
                # Αν το πρόσωπο αναγνωρίστηκε και το score είναι αρκετά υψηλό
                is_correct = input(f"Είναι σωστή η αναγνώριση του προσώπου ως '{name}'? (y/n): ").lower() == 'y'
                new_threshold = adjust_threshold(threshold, is_correct, score, threshold_label)
                print(f"[THRESHOLD] Προσαρμογή: {threshold:.3f} -> {new_threshold:.3f}")
                threshold = new_threshold
                last_threshold = new_threshold
            
            # Εμφάνιση πληροφοριών για κάθε πρόσωπο
            print(f"[INFO] Πρόσωπο: {name} | Score: {score:.2f} | Threshold: {threshold:.2f}")
            if name == "Unknown":
                label = f"Unknown (score: {score:.2f}, th_{threshold_label}:{threshold:.2f})"
            else:
                label = f"{name} (score: {score:.2f}, th_{threshold_label}:{threshold:.2f})"
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
            
            # Προσθήκη στη λίστα unknown_faces ΜΟΝΟ αν το πρόσωπο είναι άγνωστο
            if name == "Unknown":
                # Αποθήκευση snapshot και embedding
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                fname = f"{cam_name}_unknown_{score:.2f}_{ts}.jpg"
                save_path = os.path.join(UNKNOWN_SNAPSHOT_DIR, fname)
                
                # Βελτιωμένο face_crop για καλύτερη ποιότητα
                enhanced_face = enhance_face_image(face_crop)
                
                # Αύξηση μεγέθους εάν η εικόνα είναι μικρή για καλύτερη αποθήκευση
                h, w = enhanced_face.shape[:2]
                min_save_size = 200  # Ελάχιστο μέγεθος για αποθήκευση
                if min(h, w) < min_save_size:
                    scale = min_save_size / min(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    enhanced_face = cv2.resize(enhanced_face, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                # Προσθήκη μεταδεδομένων στην εικόνα για αναγνώριση
                # Προσθήκη κειμένου για score και timestamp
                h, w = enhanced_face.shape[:2]
                info_bar = np.zeros((40, w, 3), dtype=np.uint8)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(info_bar, f"Score: {score:.2f} | {timestamp}", (10, 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Συνδυασμός εικόνας με πληροφορίες
                enhanced_with_info = np.vstack([enhanced_face, info_bar])
                
                # Αποθήκευση με υψηλή ποιότητα
                # Η παράμετρος [cv2.IMWRITE_JPEG_QUALITY, 95] ορίζει υψηλή ποιότητα JPEG (0-100)
                cv2.imwrite(save_path, enhanced_with_info, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Αποθήκευση του embedding για μελλοντική χρήση
                np.save(os.path.splitext(save_path)[0] + ".npy", emb)
                
                # Επίσης αποθηκεύουμε μια έκδοση PNG χωρίς απώλειες για ιδιαίτερα σημαντικά πρόσωπα
                # όταν το score είναι κοντά στο threshold (πιθανό σημαντικό πρόσωπο)
                if abs(score - threshold) < 0.1:
                    png_path = os.path.splitext(save_path)[0] + ".png"
                    cv2.imwrite(png_path, enhanced_with_info)
                    print(f"[SNAPSHOT+] Αποθηκεύτηκε επιπλέον PNG έκδοση υψηλής ποιότητας: {os.path.basename(png_path)}")
                
                # Συνοπτική αναφορά αποθήκευσης
                print(f"[SNAPSHOT] {os.path.basename(save_path)} ({h}x{w+40}px)")
                
                # Έλεγχος αν το score είναι αρκετά κοντά στο threshold για αυτόματη παράλειψη
                should_process_face = True
                
                # Υπολογίζουμε τη διαφορά μεταξύ threshold και score
                score_diff = abs(threshold - score)
                
                # Ελέγχουμε αν η διαφορά είναι μικρότερη από IGNORE_MARGIN
                # Για παράδειγμα: αν score=0.59 και threshold=0.60, τότε η διαφορά είναι 0.01
                # Αν IGNORE_MARGIN=0.05, τότε θα πρέπει να αγνοηθεί γιατί 0.01 < 0.05
                if AUTO_IGNORE_CLOSE_SCORES and score_diff < IGNORE_MARGIN:
                    # Χρησιμοποιούμε abs() για να πιάσουμε και περιπτώσεις όπου το score είναι μεγαλύτερο του threshold
                    # αλλά πολύ κοντά σε αυτό (π.χ. score=0.61, threshold=0.60)
                    print(f"\n[AUTO-IGNORE] Το score {score:.2f} είναι κοντά στο threshold {threshold:.2f} (διαφορά: {score_diff:.2f} < {IGNORE_MARGIN:.2f})")
                    print(f"[AUTO-IGNORE] Το άγνωστο πρόσωπο αγνοήθηκε αυτόματα λόγω κοντινού score.")
                    # Η αποθήκευση του snapshot έχει ήδη γίνει, απλά δεν θα εμφανίσουμε το παράθυρο
                    should_process_face = False
                
                # Έλεγχος αν το face_crop είναι επαρκές για επεξεργασία
                has_valid_crop = face_crop is not None and face_crop.size > 0 and face_crop.shape[0] >= 30 and face_crop.shape[1] >= 30
                if not has_valid_crop:
                    print("[WARN] Το snapshot είναι άδειο ή πολύ μικρό (< 30px).")
                    
                # Εμφάνιση snapshot και επεξεργασία άμεσα μόνο αν είναι επαρκές ΚΑΙ δεν πρέπει να αγνοηθεί
                if should_process_face and has_valid_crop:
                    # Εμφάνιση του παραθύρου με το άγνωστο πρόσωπο
                    cv2.namedWindow("Unknown Snapshot", cv2.WINDOW_NORMAL)
                    cv2.imshow("Unknown Snapshot", enhanced_face)
                    cv2.moveWindow("Unknown Snapshot", 100, 100)
                    
                    # Χρήση πολλαπλών waitKey για να διασφαλίσουμε ότι το παράθυρο εμφανίζεται
                    for _ in range(3):
                        cv2.waitKey(1)
                        
                    print(f"\n[TRAINING] ⚠️ Βρέθηκε ΑΓΝΩΣΤΟ πρόσωπο! Score: {score:.2f} (threshold: {threshold:.2f})")
                    print(f"➡️ Διαφορά από threshold: {threshold-score:.2f} (Ανοχή: {IGNORE_MARGIN:.2f})")
                    print("➡️ Πάτησε 'n' για να καταχωρήσεις το πρόσωπο, 'q' για έξοδο ή οποιοδήποτε άλλο πλήκτρο για παράλειψη.")
                    
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        cv2.destroyWindow("Unknown Snapshot")
                        print("[EXIT] Έγινε έξοδος από το πρόγραμμα λόγω q.")
                        sys.exit(0)
                    elif key == ord('n'):
                        name_input = input("Όνομα: ").strip()
                        if name_input:
                            add_to_facebank(facebank, name_input, emb, enhanced_face)
                            save_facebank(facebank, FACEBANK_PATH)
                            retrain_facebank_from_known_faces()
                            print(f"✅ Το πρόσωπο '{name_input}' προστέθηκε στο facebank.")
                        else:
                            print("⚠️ Δεν δόθηκε όνομα. Παράλειψη.")
                    else:
                        print("Παράλειψη καταχώρησης snapshot.")
                    
                    # Κλείσιμο παραθύρου με ασφαλή τρόπο
                    try:
                        cv2.destroyWindow("Unknown Snapshot")
                        cv2.waitKey(1)  # Αυτό βοηθά στο να εφαρμοστεί το destroyWindow
                    except Exception as e:
                        print(f"[WARN] Μη κρίσιμο σφάλμα κατά το κλείσιμο του παραθύρου: {str(e)}")
                # Το else έχει αφαιρεθεί για να μην εμφανίζεται περιττό μήνυμα σφάλματος
                # όταν γίνεται αυτόματη παράλειψη με AUTO_IGNORE_CLOSE_SCORES
                
                # Προσθήκη στη λίστα unknown_faces για συμβατότητα με το υπόλοιπο πρόγραμμα
                unknown_faces.append((emb, (x1, y1, x2, y2), face_crop))
                
                # Επιστροφή του τρέχοντος frame με τις σημειώσεις και τη λίστα unknown_faces
                return annotated, unknown_faces
    
    return annotated, unknown_faces

def retrain_facebank_from_known_and_unknown():
    """
    Επαναϋπολογίζει το facebank από Known_faces ΚΑΙ Unknown_snapshots (όλα τα .npy).
    Τα unknown μπαίνουν στο facebank ως 'Unknown_...' key.
    """
    global facebank
    facebank = {}

    # --- Known Faces ---
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        embeddings = []
        for file in os.listdir(person_dir):
            ext = os.path.splitext(file)[1].lower()
            file_path = os.path.join(person_dir, file)
            if ext == ".npy":
                try:
                    emb = np.load(file_path)
                    embeddings.append(emb)
                except Exception:
                    pass
            elif ext in valid_exts:
                img = cv2.imread(file_path)
                if img is not None:
                    emb = arcface.get_embedding(img)
                    embeddings.append(emb)
        if embeddings:
            facebank[person_name] = embeddings

    # --- Unknown Faces (από snapshots) ---
    for file in os.listdir(UNKNOWN_SNAPSHOT_DIR):
        if file.endswith(".npy"):
            emb_path = os.path.join(UNKNOWN_SNAPSHOT_DIR, file)
            emb = np.load(emb_path)
            # Δημιουργεί ξεχωριστό label για κάθε snapshot: Unknown_...
            name = os.path.splitext(file)[0]  # Unique name π.χ. Camera_PC_unknown_...
            facebank[name] = [emb]

    save_facebank(facebank, FACEBANK_PATH)
    print("🔄 Έγινε retrain όλων των προσώπων και unknown στο facebank.")

def retrain_facebank_from_known_faces():
    """
    Επαναϋπολογίζει το facebank ΜΟΝΟ από τον φάκελο Known_faces (χωρίς τα unknown).
    """
    global facebank
    facebank = load_facebank(KNOWN_FACES_DIR, arcface, valid_exts)
    save_facebank(facebank, FACEBANK_PATH)
    print("🔄 Έγινε retrain όλων των γνωστών προσώπων στο facebank.")

def run_video_loop(caps, names, threshold=0.5, auto_learn=False, threshold_label="1"):
    print("\nΠατήστε 'q' για έξοδο, 'r' για retrain όλων των προσώπων.")
    print("Το πρόγραμμα θα σταματήσει αυτόματα όταν βρει άγνωστο πρόσωπο για να το εκπαιδεύσετε.")
    
    global facebank
    last_frame = {}  # Αποθήκευση του τελευταίου frame για κάθε κάμερα
    
    while True:
        active = 0
        unknown_faces_all = []
        
        # Διάβασμα και επεξεργασία frames από όλες τις κάμερες
        for i, cap in enumerate(caps):
            if cap is None or not cap.isOpened(): 
                continue
                
            ret, frame = cap.read()
            if not ret:
                print(f"[ERROR] Κάμερα '{names[i]}' σταμάτησε να στέλνει εικόνα.")
                cap.release()
                continue
                
            # Αποθήκευση του τρέχοντος frame
            last_frame[names[i]] = frame
            
            # Επεξεργασία frame - αυτό θα εντοπίσει και θα χειριστεί τα άγνωστα πρόσωπα απευθείας
            annotated, unknown_faces = process_frame(
                frame, cam_name=names[i], threshold=threshold, auto_learn=auto_learn, threshold_label=threshold_label
            )
            
            cv2.imshow(names[i], annotated)
            unknown_faces_all.extend(unknown_faces)
            active += 1
        
        # Έλεγχος αν όλες οι κάμερες έχουν κλείσει
        if active == 0:
            print("❌ Όλες οι κάμερες έχουν κλείσει.")
            break
        
        # Χειρισμός πλήκτρων
        key = cv2.waitKey(1) & 0xFF
        
        # Χειρισμός για το πλήκτρο 'n' - προσθήκη άγνωστου προσώπου χειροκίνητα
        # (αυτό είναι για επιπλέον περιπτώσεις, η βασική λειτουργία γίνεται αυτόματα στο process_frame)
        if key == ord('n') and unknown_faces_all:
            emb, (x1, y1, x2, y2), face_crop = unknown_faces_all[0]
            enhanced_face = enhance_face_image(face_crop)
            
            # Εμφάνιση προσώπου
            cv2.namedWindow("Manual Face Add", cv2.WINDOW_NORMAL)
            cv2.imshow("Manual Face Add", enhanced_face)
            cv2.moveWindow("Manual Face Add", 100, 100)
            cv2.waitKey(1)
            
            print("➡️ Χειροκίνητη προσθήκη προσώπου. Δώστε όνομα (ή πατήστε Enter για παράλειψη):")
            try:
                name = input("Όνομα: ").strip()
            except EOFError:
                print("⚠️ Δεν δόθηκε όνομα (EOF). Συνέχεια...")
                name = ""
            
            # Κλείσιμο του παραθύρου
            try:
                cv2.destroyWindow("Manual Face Add")
                cv2.waitKey(1)
            except:
                pass
                
            if name:
                add_to_facebank(facebank, name, emb, face_crop)
                save_facebank(facebank, FACEBANK_PATH)
                print(f"✅ Το πρόσωπο '{name}' προστέθηκε στο facebank.")
                print(f"[INFO] Facebank keys: {list(facebank.keys())}")
                retrain_facebank_from_known_faces()
                print("🔄 Έγινε retrain όλων των προσώπων στο facebank.")
            else:
                print("⚠️ Δεν δόθηκε όνομα. Το πρόσωπο αγνοήθηκε.")
        
        # Χειρισμός για το πλήκτρο 'r' - retrain facebank
        if key == ord('r'):
            retrain_facebank_from_known_faces()
            print("🔄 Έγινε retrain όλων των προσώπων στο facebank.")
            
        # Χειρισμός για το πλήκτρο 'q' - έξοδος
        if key == ord('q'):
            print("Πατήθηκε 'q' - έξοδος από όλες τις κάμερες.")
            break
            
        if key == ord('r'):
            retrain_facebank_from_known_faces()
        if key == ord('q'):
            print("Πατήθηκε 'q' - έξοδος από όλες τις κάμερες.")
            break
    # Απελευθέρωση πόρων κάμερας
    for cap in caps:
        if cap:
            try:
                cap.release()
            except:
                pass
                
    # Κλείσιμο όλων των παραθύρων με πολλαπλές προσπάθειες για να βεβαιωθούμε ότι κλείνουν
    for _ in range(3):
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Αυτό είναι απαραίτητο για να εφαρμοστεί το destroyAllWindows
        except:
            pass
    
    print("[EXIT] Έγινε έξοδος από όλες τις κάμερες.")
    print("[INFO] Ξεκινά retrain για όλα τα γνωστά και άγνωστα πρόσωπα...")
    
    # Τελικό retrain για αποθήκευση όλων των δεδομένων
    try:
        retrain_facebank_from_known_and_unknown()
        print("[INFO] Ολοκληρώθηκε retrain για όλα τα γνωστά και άγνωστα πρόσωπα.")
    except Exception as e:
        print(f"[ERROR] Σφάλμα κατά το retrain: {str(e)}")
        print("[INFO] Αποθήκευση του τρέχοντος facebank...")
        try:
            save_facebank(facebank, FACEBANK_PATH)
            print("[INFO] Το facebank αποθηκεύτηκε επιτυχώς.")
        except:
            print("[ERROR] Αποτυχία αποθήκευσης του facebank.")

# === Main ===
if __name__ == "__main__":
    print("\n========== ΣΥΣΤΗΜΑ ΑΝΑΓΝΩΡΙΣΗΣ ΠΡΟΣΩΠΩΝ ==========")
    print(f"Το σύστημα έχει ρυθμιστεί με περιθώριο ανοχής: {IGNORE_MARGIN}")
    if AUTO_IGNORE_CLOSE_SCORES:
        print(f"Αυτόματη παράλειψη άγνωστων προσώπων: ΕΝΕΡΓΗ")
        print(f"Τα άγνωστα πρόσωπα με score που απέχει λιγότερο από {IGNORE_MARGIN} από το threshold θα αγνοούνται αυτόματα.")
        print(f"Παράδειγμα με threshold={0.6}:")
        print(f"  - Score={0.59}: διαφορά={0.01} < {IGNORE_MARGIN}, άρα ΘΑ αγνοηθεί")
        print(f"  - Score={0.57}: διαφορά={0.03} > {IGNORE_MARGIN}, άρα ΔΕΝ θα αγνοηθεί")
        print(f"  - Score={0.61}: διαφορά={0.01} < {IGNORE_MARGIN}, άρα ΘΑ αγνοηθεί επίσης (άνω του threshold)")
        print(f"Για αλλαγή της τιμής του περιθωρίου ανοχής, επεξεργαστείτε το αρχείο configuration.py")
    else:
        print(f"Αυτόματη παράλειψη άγνωστων προσώπων: ΑΝΕΝΕΡΓΗ")
    print("=================================================\n")
    
    print("Επιλέξτε πηγή βίντεο για την ανίχνευση:")
    print("1: Κάμερα υπολογιστή / Εξωτερική / Και οι δύο")
    print("2: Αποθηκευμένο βίντεο")
    print("3: Κλειστό σύστημα παρακολούθησης (RTSP)")
    choice = input("Δώσε επιλογή (1, 2 ή 3): ").strip()

    threshold, threshold_label = get_threshold(choice)
    caps, names = [], []

    if choice == "1":
        print("\nΕπιλέξτε κάμερα για χρήση:")
        print("1: Κάμερα υπολογιστή")
        print("2: Εξωτερική κάμερα (1: USB, 2: Wi-Fi/DroidCam)")
        print("3: Και οι δύο κάμερες")
        camera_choice = input("Δώσε επιλογή (1, 2 ή 3): ").strip()

        if camera_choice == "1":
            cap_pc = try_open(INDEX_PC, "Κάμερα υπολογιστή")
            if cap_pc:
                caps.append(cap_pc)
                names.append("Camera_PC")

        elif camera_choice == "2":
            conn_type = input("Επιλέξτε τύπο σύνδεσης (1: USB, 2: Wi-Fi/DroidCam): ").strip()
            if conn_type in {"1", "2"}:
                cap_droid = try_open(INDEX_DROIDCAM, "DroidCam")
                if cap_droid:
                    caps.append(cap_droid)
                    names.append("Camera_DroidCam")
            else:
                print("[ERROR] Μη έγκυρη επιλογή σύνδεσης.")

        elif camera_choice == "3":
            cap_pc = try_open(INDEX_PC, "Κάμερα υπολογιστή")
            cap_droid = try_open(INDEX_DROIDCAM, "DroidCam")
            if cap_pc:
                caps.append(cap_pc)
                names.append("Camera_PC")
            if cap_droid:
                caps.append(cap_droid)
                names.append("Camera_DroidCam")
        else:
            print("❌ Μη έγκυρη επιλογή.")
            exit()

        if not caps:
            print("❌ Καμία κάμερα δεν είναι διαθέσιμη.")
            exit()

        run_video_loop(caps, names, threshold=threshold, auto_learn=True, threshold_label=threshold_label)
        # Μετά το τέλος του loop, μηνύματα retrain θα εμφανιστούν από το run_video_loop

    elif choice == "2":
        print("📂 Διαθέσιμα βίντεο στον φάκελο 'video':")
        for file in os.listdir(VIDEO_DIR):
            if file.endswith((".mp4", ".avi", ".mkv")):
                print(f"  - {file}")
        video_filename = input("🎥 Δώσε το όνομα του αρχείου βίντεο (π.χ., video1.mp4): ").strip()
        video_path = os.path.join(VIDEO_DIR, video_filename)
        cap = initialize_capture(video_path)
        if cap:
            print("Πατήστε 'q' για έξοδο, 'n' για να δώσεις όνομα όταν εμφανιστεί άγνωστο πρόσωπο, 'r' για retrain όλων των προσώπων.")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, unknown_faces = process_frame(
                    frame, cam_name=f"Video: {video_filename}", threshold=threshold, threshold_label=threshold_label
                )
                cv2.imshow(f"Video: {video_filename}", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('n') and unknown_faces:
                    emb, (x1, y1, x2, y2), face_crop = unknown_faces[0]
                    print("➡️ Εντοπίστηκε άγνωστο πρόσωπο. Δώσε όνομα (ή πάτα Enter για παράλειψη):")
                    try:
                        name = input("Όνομα: ").strip()
                    except EOFError:
                        print("⚠️ Δεν δόθηκε όνομα (EOF). Συνέχεια...")
                        name = ""
                    if name:
                        add_to_facebank(facebank, name, emb, face_crop)
                        save_facebank(facebank, FACEBANK_PATH)
                        print(f"✅ Το πρόσωπο '{name}' προστέθηκε στο facebank.")
                        print(f"[INFO] Facebank keys: {list(facebank.keys())}")
                        retrain_facebank_from_known_faces()
                        print("🔄 Έγινε retrain όλων των προσώπων στο facebank.")
                if key == ord('r'):
                    retrain_facebank_from_known_faces()
                    print("🔄 Έγινε retrain όλων των προσώπων στο facebank.")
                if key == ord('q'):
                    print("Πατήθηκε 'q' - έξοδος από το βίντεο.")
                    break
            cap.release()
            cv2.destroyAllWindows()
            print("[EXIT] Έγινε έξοδος από το βίντεο.")
            print("[INFO] Ξεκινά retrain για όλα τα γνωστά και άγνωστα πρόσωπα...")
            retrain_facebank_from_known_and_unknown()
            print("[INFO] Ολοκληρώθηκε retrain για όλα τα γνωστά και άγνωστα πρόσωπα.")
        else:
            print("❌ Δεν ήταν δυνατή η ανάγνωση του αρχείου βίντεο.")
    elif choice == "3":
        num_cameras = int(input("Δώσε τον αριθμό των καμερών στο σύστημα παρακολούθησης: ").strip())
        rtsp_urls = [input(f"Δώσε το RTSP URL για την κάμερα {i + 1}: ").strip() for i in range(num_cameras)]
        for i, url in enumerate(rtsp_urls):
            cap = initialize_capture(url)
            if cap:
                caps.append(cap)
                names.append(f"RTSP_Cam_{i+1}")
        if not caps:
            print("❌ Δεν είναι δυνατή η πρόσβαση σε καμία κάμερα.")
            exit()
        run_video_loop(caps, names, threshold=threshold, auto_learn=True, threshold_label=threshold_label)
        # Μετά το τέλος του loop, μηνύματα retrain θα εμφανιστούν από το run_video_loop
    else:
        print("❌ Μη έγκυρη επιλογή. Παρακαλώ τρέξε το πρόγραμμα ξανά.")
        exit()

    print("✅ Το πρόγραμμα τερματίστηκε επιτυχώς.")
    cv2.destroyAllWindows()
    exit()
# EOF

# Επεξήγηση score και threshold:
#
# - Το score είναι το αποτέλεσμα της ομοιότητας (cosine similarity) μεταξύ του embedding του ανιχνευμένου προσώπου και των embeddings των γνωστών προσώπων.
#   - score = 1 - cosine(embedding1, embedding2)
#   - Τιμές κοντά στο 1 σημαίνουν μεγάλη ομοιότητα (ίδιο πρόσωπο), τιμές κοντά στο 0 σημαίνουν διαφορετικά πρόσωπα.
#
# - Το threshold είναι το όριο που ορίζεις εσύ για να αποφασίσεις αν ένα πρόσωπο θεωρείται "γνωστό" ή "άγνωστο".
#   - Αν το score είναι μικρότερο από το threshold, το πρόσωπο χαρακτηρίζεται ως "Unknown".
#   - Αν το score είναι μεγαλύτερο ή ίσο με το threshold, το πρόσωπο αναγνωρίζεται ως το γνωστό πρόσωπο με το μεγαλύτερο score.
#
# Παράδειγμα:
#   Αν threshold = 0.5 και το score για κάποιο πρόσωπο είναι 0.7 → αναγνωρίζεται ως γνωστό.
#   Αν το score είναι 0.3 → χαρακτηρίζεται ως "Unknown".

# Η ταυτοποίηση ενός προσώπου ως "άγνωστο" γίνεται ως εξής:
#
# 1. Για κάθε ανιχνευμένο πρόσωπο, υπολογίζεται το embedding (διάνυσμα χαρακτηριστικών) με το ArcFace.
# 2. Υπολογίζεται η ομοιότητα (cosine similarity) του embedding αυτού με ΟΛΑ τα embeddings των γνωστών προσώπων (facebank).
#    - Η ομοιότητα score = 1 - cosine(embedding1, embedding2)
#    - Score κοντά στο 1 σημαίνει μεγάλη ομοιότητα (πιθανόν ίδιο πρόσωπο).
# 3. Βρίσκεται το ΜΕΓΙΣΤΟ score (δηλαδή το πιο κοντινό γνωστό πρόσωπο).
# 4. Αν το μέγιστο score είναι ΜΙΚΡΟΤΕΡΟ από το threshold (π.χ. 0.5 ή 0.55), τότε το πρόσωπο χαρακτηρίζεται ως "Unknown".
#    - Δηλαδή: if best_score < threshold: -> "Unknown"
# 5. Αν το μέγιστο score είναι ΜΕΓΑΛΥΤΕΡΟ ή ΙΣΟ με το threshold, τότε το πρόσωπο ταυτοποιείται ως το γνωστό πρόσωπο με το μεγαλύτερο score.

# Παράδειγμα:
# - Αν threshold = 0.6 και το καλύτερο score είναι 0.72 -> το πρόσωπο αναγνωρίζεται ως γνωστό.
# - Αν το καλύτερο score είναι 0.41 -> το πρόσωπο χαρακτηρίζεται ως "Unknown".