"""HTTP routes (thin handlers; logic lives in services/)."""
import asyncio

from flask import Blueprint, current_app, render_template, request, session, redirect, url_for

from config import FACES_DIR, FACES_KNN_DIR, datetoday2
from extensions import camera_lock
from services import attendance as attendance_svc
from services import capture
from services import storage
from services import users as users_svc

bp = Blueprint('main', __name__)


def _session_use_knn():
    return session.get('use_knn', True)


@bp.route('/')
async def home():
    use_knn = _session_use_knn()
    names, rolls, dates, check_ins, check_outs, l = attendance_svc.extract_attendance(use_knn)
    return render_template(
        'home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
        totalreg=storage.totalreg(use_knn), datetoday2=datetoday2, use_knn=use_knn, active_page='home',
    )


@bp.route('/users')
async def users_list():
    mess = session.pop('flash_mess', None)
    use_knn = _session_use_knn()
    rows = storage.list_user_rows(FACES_KNN_DIR if use_knn else FACES_DIR)
    return render_template(
        'users.html', rows=rows, use_knn=use_knn,
        active_page='users', mess=mess,
    )


@bp.route('/users/delete', methods=['POST'])
async def delete_user():
    mode = request.form.get('mode')
    folder = (request.form.get('folder') or '').strip()
    if mode not in ('knn', 'direct') or not folder:
        session['flash_mess'] = 'Invalid request.'
        return redirect(url_for('main.users_list'))
    if not camera_lock.acquire(blocking=False):
        session['flash_mess'] = 'Another operation is in progress. Please try again.'
        return redirect(url_for('main.users_list'))
    app_obj = current_app._get_current_object()

    def _run_delete():
        with app_obj.app_context():
            return users_svc.delete_user_folder_and_retrain(mode, folder)

    try:
        ok = await asyncio.to_thread(_run_delete)
        if ok:
            session['flash_mess'] = (
                'User deleted; KNN model retrained.' if mode == 'knn' else 'User deleted; known_faces.pkl rebuilt.'
            )
        else:
            session['flash_mess'] = 'Could not delete user (invalid folder or error).'
    finally:
        camera_lock.release()
    return redirect(url_for('main.users_list'))


@bp.route('/toggle_recognition_mode', methods=['POST'])
async def toggle_recognition_mode():
    session['use_knn'] = not session.get('use_knn', True)
    return redirect(request.referrer or url_for('main.home'))


@bp.route('/start', methods=['GET'])
async def start():
    use_knn = _session_use_knn()
    app_obj = current_app._get_current_object()

    if not camera_lock.acquire(blocking=False):
        names, rolls, dates, check_ins, check_outs, l = attendance_svc.extract_attendance(use_knn)
        return render_template(
            'home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
            totalreg=storage.totalreg(use_knn), datetoday2=datetoday2, use_knn=use_knn, active_page='home',
            mess='Camera is in use (e.g. Add User or another Take Attendance). Please wait and try again.',
        )
    try:
        err, _ = capture.check_model_and_reg(use_knn=use_knn)
        if err is not None:
            return render_template('home.html', **err, use_knn=use_knn, active_page='home')

        def _run_camera():
            with app_obj.app_context():
                return capture.attendance_camera(use_knn)

        names, rolls, dates, check_ins, check_outs, l = await asyncio.to_thread(_run_camera)
        return render_template(
            'home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
            totalreg=storage.totalreg(use_knn), datetoday2=datetoday2, use_knn=use_knn, active_page='home',
        )
    finally:
        camera_lock.release()


@bp.route('/add', methods=['POST'])
async def add():
    use_knn = _session_use_knn()
    app_obj = current_app._get_current_object()

    if not camera_lock.acquire(blocking=False):
        names, rolls, dates, check_ins, check_outs, l = attendance_svc.extract_attendance(use_knn)
        return render_template(
            'home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
            totalreg=storage.totalreg(use_knn), datetoday2=datetoday2, use_knn=use_knn, active_page='home',
            mess='Another operation is in progress. Please wait for Take Attendance or Add User to finish, then try again.',
        )
    try:
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']

        def _run_add():
            with app_obj.app_context():
                return capture.add_user(newusername, newuserid, use_knn)

        names, rolls, dates, check_ins, check_outs, l = await asyncio.to_thread(_run_add)
        return render_template(
            'home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
            totalreg=storage.totalreg(use_knn), datetoday2=datetoday2, use_knn=use_knn, active_page='home',
        )
    finally:
        camera_lock.release()


def register_routes(app):
    app.register_blueprint(bp)
