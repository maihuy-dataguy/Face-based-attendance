"""Delete user folders and refresh models."""
import shutil
from datetime import datetime, timezone

from config import FACES_DIR, FACES_KNN_DIR

from extensions import db
from models import User
from services import encodings
from services import knn
from services import storage


def _soft_delete_knn_user(folder):
    """Set users.deleted_at for KNN user matching folder name (after folder removed)."""
    _, userid_str = storage.parse_folder_display(folder)
    if userid_str == '—':
        return
    u = User.query.filter_by(user_id=userid_str).first()
    if u:
        u.deleted_at = datetime.now(timezone.utc)
        db.session.commit()


def delete_user_folder_and_retrain(mode, folder):
    """Remove user folder; retrain KNN or rebuild known_faces.pkl. KNN: soft-delete in MySQL."""
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
        try:
            _soft_delete_knn_user(folder)
        except Exception:
            db.session.rollback()
            # Folder is already removed; DB soft-delete can be retried manually if needed
    else:
        encodings.rebuild_known_faces_pickle_full()
    return True
