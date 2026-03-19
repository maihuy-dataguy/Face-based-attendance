"""Delete user folders and refresh models."""
import shutil

from config import FACES_DIR, FACES_KNN_DIR

from services import encodings
from services import knn
from services import storage


def delete_user_folder_and_retrain(mode, folder):
    """Remove user folder; retrain KNN or rebuild known_faces.pkl. Returns True if deleted."""
    base = FACES_KNN_DIR if mode == 'knn' else FACES_DIR
    path = storage.safe_user_folder(base, folder)
    if not path:
        return False
    try:
        shutil.rmtree(path)
    except OSError:
        return False
    if mode == 'knn':
        knn.retrain_knn_model()
    else:
        encodings.rebuild_known_faces_pickle_full()
    return True
