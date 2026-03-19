"""User folders, registration counts, listing."""
import os

from config import FACES_DIR, FACES_KNN_DIR, faces_dir


def totalreg(use_knn=False):
    d = faces_dir(use_knn)
    if not os.path.isdir(d):
        return 0
    return len([x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))])


def parse_folder_display(folder):
    """Folder name is {name}_{id}; split on last underscore."""
    parts = folder.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return folder, '—'


def list_user_rows(base_dir):
    """List of dicts: folder, display_name, user_id for template tables."""
    rows = []
    if not os.path.isdir(base_dir):
        return rows
    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        display_name, user_id = parse_folder_display(name)
        rows.append({'folder': name, 'display_name': display_name, 'user_id': user_id})
    return rows


def safe_user_folder(base, folder):
    """Resolve user folder path or None if invalid / path traversal."""
    if not folder or '..' in folder or '/' in folder or '\\' in folder:
        return None
    base = os.path.normpath(os.path.abspath(base))
    path = os.path.normpath(os.path.abspath(os.path.join(base, folder)))
    if not path.startswith(base + os.sep):
        return None
    if not os.path.isdir(path):
        return None
    return path
