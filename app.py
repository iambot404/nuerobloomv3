from flask import Flask, render_template, jsonify, request, redirect, url_for, session, make_response
import os
from db import get_connection, get_db_connection, DBError
from assessment_routes import register_assessment_routes

from urllib.parse import parse_qs, urlencode
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-change-me')


# WSGI Middleware to correct PATH_INFO when Vercel serverless rewrites route requests
class VercelPathFix:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path_info = environ.get('PATH_INFO', '')

        # 1. Check if Vercel passed original path via __vercel_route or path query param
        qs = environ.get('QUERY_STRING', '')
        if '__vercel_route' in qs or 'path=' in qs:
            params = parse_qs(qs, keep_blank_values=True)
            route_vals = params.pop('__vercel_route', None) or params.pop('path', None)
            if route_vals and route_vals[0] and route_vals[0] not in (':path*', '/:path*'):
                route = route_vals[0]
                if not route.startswith('/'):
                    route = '/' + route
                if len(route) > 1 and route.endswith('/'):
                    route = route.rstrip('/')
            else:
                route = '/'
            # Reconstruct clean QUERY_STRING without internal param
            environ['QUERY_STRING'] = urlencode(params, doseq=True)
            environ['PATH_INFO'] = route
            return self.wsgi_app(environ, start_response)

        # 2. Check if Vercel proxy headers passed original request path
        forwarded = (
            environ.get('HTTP_X_FORWARDED_URI') or
            environ.get('RAW_URI') or
            environ.get('REQUEST_URI')
        )
        if forwarded and forwarded not in ('/api/index', '/api/index.py', '/api', '/api/'):
            clean_path = forwarded.split('?')[0]
            environ['PATH_INFO'] = clean_path if clean_path else '/'
        elif path_info in ('/api/index', '/api/index.py', '/api', '/api/'):
            environ['PATH_INFO'] = '/'
        elif path_info.startswith('/api/index/'):
            environ['PATH_INFO'] = path_info[len('/api/index'):] or '/'
        elif path_info.startswith('/api/index.py/'):
            environ['PATH_INFO'] = path_info[len('/api/index.py'):] or '/'

        return self.wsgi_app(environ, start_response)


# Apply ProxyFix for correct protocol and host headers behind Vercel reverse proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)

# Apply VercelPathFix to handle rewritten URL routing
app.wsgi_app = VercelPathFix(app.wsgi_app)

# Register assessment routes
register_assessment_routes(app)


@app.route('/db-test')
def db_test():
    """Simple endpoint to test DB connectivity. Returns database version on success."""
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT VERSION()')
        version = cursor.fetchone()
        cursor.close()
        conn.close()
        v_str = version.get('version') if isinstance(version, dict) else str(version[0])
        return jsonify({'status': 'ok', 'database_version': v_str, 'provider': 'Supabase PostgreSQL'})
    except Exception as e:
        app.logger.exception('DB test failed')
        return jsonify({'status': 'error', 'error': str(e)}), 500




def insert_user(name, email, hashed_password, contact, role='student'):
    """Insert a user into the users table. Returns (True, None) on success or (False, error_msg) on failure."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        sql = "INSERT INTO users (name, email, password, contact, role) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(sql, (name, email, hashed_password, contact, role))
        conn.commit()
        cursor.close()
        conn.close()
        return True, None
    except Exception as e:
        # detect duplicate email
        err = str(e)
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        if 'Duplicate' in err or 'UNIQUE' in err:
            return False, 'Email already registered'
        return False, err


from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from flask import session
from werkzeug.utils import secure_filename
import uuid


def login_required(role=None):
    """Decorator to require a logged-in user with an optional specific role.

    Usage: @login_required('student') or @login_required()
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            user_id = session.get('user_id')
            user_role = session.get('role')
            if not user_id:
                app.logger.info('Access denied: not logged in')
                # Check if this is an API request
                if request.path.startswith('/api/'):
                    return jsonify({'status': 'error', 'error': 'Not authenticated'}), 401
                return redirect(url_for('login'))
            if role and user_role != role:
                app.logger.info(f"Access denied: role mismatch (required={role}, actual={user_role})")
                # Check if this is an API request
                if request.path.startswith('/api/'):
                    return jsonify({'status': 'error', 'error': 'Access denied'}), 403
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return wrapped
    return decorator


def ensure_profile_column():
    """Ensure that the users table has a profile_photo column. Attempt to add it if missing."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN profile_photo VARCHAR(255) NULL;")
            conn.commit()
            app.logger.info('Added profile_photo column to users table')
        except Exception as inner_e:
            # Ignore if column exists or other harmless error
            app.logger.debug('ensure_profile_column check/skip: %s', inner_e)
        
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN class VARCHAR(100) NULL;")
            conn.commit()
            app.logger.info('Added class column to users table')
        except Exception as inner_e:
            app.logger.debug('ensure_class_column check/skip: %s', inner_e)
        
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN faculty_id INT NULL;")
            conn.commit()
            app.logger.info('Added faculty_id column to users table')
        except Exception as inner_e:
            app.logger.debug('ensure_faculty_id_column check/skip: %s', inner_e)
        
        cursor.close()
        conn.close()
    except Exception as e:
        app.logger.warning('Could not ensure columns: %s', e)


# Try to ensure column exists at startup (best-effort)
ensure_profile_column()


def cleanup_profile_photos_for_disallowed_roles():
    """Remove profile_photo files and DB values for users whose role is not student.

    This is a best-effort cleanup to ensure only students keep profile photos.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, profile_photo, role FROM users WHERE profile_photo IS NOT NULL")
        rows = cursor.fetchall()
        for r in rows:
            role = r.get('role')
            pp = r.get('profile_photo')
            if not pp:
                continue
            if role not in ('student',):
                # attempt delete file on disk
                try:
                    rel = pp.lstrip('/')
                    abs_path = os.path.join(os.path.dirname(__file__), *rel.split('/'))
                    if os.path.exists(abs_path):
                        os.remove(abs_path)
                        app.logger.info('Removed profile photo file for user %s: %s', r.get('id'), abs_path)
                except Exception:
                    app.logger.exception('Failed to remove profile photo file: %s', pp)
                try:
                    upd = conn.cursor()
                    upd.execute('UPDATE users SET profile_photo = NULL WHERE id = %s', (r.get('id'),))
                    upd.close()
                except Exception:
                    app.logger.exception('Failed to clear profile_photo DB for user %s', r.get('id'))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception:
        app.logger.exception('cleanup_profile_photos_for_disallowed_roles failed')


# Best-effort cleanup at startup
cleanup_profile_photos_for_disallowed_roles()


@app.route('/faculty_register', methods=['GET', 'POST'])
def faculty_register():
    return redirect(url_for('signup'))


@app.route('/student_register', methods=['POST'])
def student_register():
    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '')
    contact = request.form.get('contact', '').strip()

    if not (name and email and password):
        return "Name, email and password are required", 400

    hashed = generate_password_hash(password)
    ok, err = insert_user(name, email, hashed, contact, role='student')
    if not ok:
        return f"Registration failed: {err}", 400
    return redirect(url_for('login'))


def authenticate_user(email, password, expected_role=None):
    """Authenticate a user by email and password. Optionally check role.
    Returns (True, user_row_dict) on success, (False, error_msg) on failure.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT id, name, email, password, contact, role, profile_photo FROM users WHERE email = %s', (email,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if not row:
            return False, 'User not found'
        if expected_role and row.get('role') != expected_role:
            return False, 'Role mismatch'
        # stored hash in row['password']
        if not check_password_hash(row.get('password', ''), password):
            return False, 'Invalid password'
        return True, row
    except Exception as e:
        return False, str(e)


@app.route('/login/student', methods=['POST'])
def login_student():
    email = (request.form.get('email') or request.form.get('student_email') or '').strip()
    password = request.form.get('password') or request.form.get('student_password') or ''
    ok, result = authenticate_user(email, password)
    if not ok:
        return f"Login failed: {result}", 401
    # on success set session and redirect to student profile page
    from flask import session
    session['user_id'] = result['id']
    session['role'] = result.get('role', 'student')
    session['name'] = result['name']
    # profile photo (if present)
    if result.get('profile_photo'):
        session['profile_photo'] = result.get('profile_photo')
    return redirect(url_for('student_profile'))


@app.route('/login/faculty', methods=['GET', 'POST'])
def login_faculty():
    return redirect(url_for('login'))


@app.route('/student')
@login_required('student')
def student_page():
    # ensure session has latest profile_photo so page shows uploaded image after refresh
    user_id = session.get('user_id')
    if user_id and not session.get('profile_photo'):
        try:
            conn = get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT profile_photo FROM users WHERE id = %s', (user_id,))
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            if row and row.get('profile_photo'):
                session['profile_photo'] = row.get('profile_photo')
        except Exception:
            app.logger.debug('Could not refresh profile_photo into session')
    return render_template('student.html')


@app.route('/student-dashboard')
@login_required('student')
def student_dashboard():
    """Render student dashboard page"""
    return render_template('student-dashboard.html')


@app.route('/student-profile')
@login_required('student')
def student_profile():
    """Render student profile page"""
    return render_template('student-profile.html')


@app.route('/faculty')
def faculty_page():
    return redirect(url_for('student_profile'))


@app.route('/logout')
def logout():
    from flask import session
    session.clear()
    return redirect(url_for('login'))


@app.route('/upload-profile-photo', methods=['POST'])
@login_required()
def upload_profile_photo():
    """Accept a multipart/form-data upload with field 'photo', save file, update users.profile_photo and return JSON with url."""
    try:
        # ensure only students may have profile photos; run a small cleanup first
        cleanup_profile_photos_for_disallowed_roles()

        user_id = session.get('user_id')
        user_role = session.get('role')
        if not user_id:
            return jsonify({'status': 'error', 'error': 'Not authenticated'}), 401
        if user_role not in ('student',):
            # reject upload from disallowed roles
            return jsonify({'status': 'error', 'error': 'Only students may upload profile photos'}), 403

        if 'photo' not in request.files:
            return jsonify({'status': 'error', 'error': 'No file part'}), 400
        file = request.files['photo']
        if file.filename == '':
            return jsonify({'status': 'error', 'error': 'No selected file'}), 400

        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'png'
        # create unique filename
        unique = f"{uuid.uuid4().hex}.{ext}"
        upload_dir = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        dest = os.path.join(upload_dir, unique)

        # save file to disk; if anything fails later we'll remove it
        file.save(dest)

        # update user row and remove previous image if present
        conn = get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT profile_photo FROM users WHERE id = %s', (user_id,))
            row = cursor.fetchone()
            prev = row.get('profile_photo') if row else None
            # store relative path
            rel_path = f"/static/uploads/{unique}"
            upd = conn.cursor()
            upd.execute('UPDATE users SET profile_photo = %s WHERE id = %s', (rel_path, user_id))
            conn.commit()
            upd.close()
            cursor.close()
        except Exception:
            # cleanup newly saved file on error
            try:
                if os.path.exists(dest):
                    os.remove(dest)
            except Exception:
                app.logger.exception('Failed to remove temp upload after DB error')
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass

        # delete previous photo file if it was in uploads
        if prev:
            try:
                prev_rel = prev.lstrip('/')
                # only remove files that are inside our uploads folder
                if prev_rel.startswith('static/uploads/'):
                    prev_abs = os.path.join(os.path.dirname(__file__), *prev_rel.split('/'))
                    if os.path.exists(prev_abs):
                        os.remove(prev_abs)
                        app.logger.info('Removed previous profile photo for user %s', user_id)
            except Exception:
                app.logger.exception('Failed to remove previous profile photo for user %s', user_id)
        # update session so the uploaded photo persists across refreshes
        try:
            session['profile_photo'] = rel_path
        except Exception:
            app.logger.debug('Could not set session profile_photo (client may not accept cookies)')

        return jsonify({'status': 'ok', 'url': rel_path})
    except Exception as e:
        app.logger.exception('Upload failed')
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/favicon.ico')
def favicon():
    return ('', 204)


@app.route('/')
@app.route('/api/index')
@app.route('/api/index.py')
def home():
    return render_template('index.html')


@app.route('/signin')
def signin():
    return redirect(url_for('login'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        contact = request.form.get('contact', '').strip()

        if not (name and email and password):
            return render_template('signup.html', error="Name, email and password are required"), 400

        hashed = generate_password_hash(password)
        ok, err = insert_user(name, email, hashed, contact, role='student')
        if not ok:
            return render_template('signup.html', error=f"Registration failed: {err}"), 400
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/dashboard')
def dashboard_redirect():
    """Intelligent dashboard redirection based on authentication state"""
    user_id = session.get('user_id')
    if user_id:
        return redirect(url_for('student_profile'))
    return redirect(url_for('list_assessments'))


@app.route('/profile')
def profile_redirect():
    """Intelligent profile redirection based on authentication state"""
    user_id = session.get('user_id')
    if user_id:
        return redirect(url_for('student_profile'))
    return redirect(url_for('login'))


@app.route('/how-it-works')
def how_it_works():
    return redirect('/#how-it-works')


@app.route('/about')
def about():
    return redirect('/#about')


@app.route('/assessments')
def list_assessments():
    """Display list of available assessments for students and guests (public-first)"""
    user_id = session.get('user_id')
    user_role = session.get('role')
    is_authenticated = bool(user_id)
    
    return render_template('assessments_list.html', 
                           is_authenticated=is_authenticated,
                           user_id=user_id,
                           user_role=user_role)


@app.route('/test-api')
def test_api_page():
    """Test page for API debugging"""
    return render_template('test_api.html')


@app.route('/assessment/<int:assessment_id>')
def take_assessment(assessment_id):
    """Display assessment taking interface - public-first for students and guests"""
    user_id = session.get('user_id')
    is_guest = not bool(user_id)
    
    # Special handling for Dyslexia Assessment (ID=1)
    if assessment_id == 1:
        app.logger.info('Serving integrated dyslexia assessment (guest=%s)', is_guest)
        return render_template('dyslexia_integrated.html', assessment_id=assessment_id, is_guest=is_guest, user_id=user_id)
    
    # Special handling for Dyscalculia Assessment (ID=2)
    if assessment_id == 2:
        app.logger.info('Serving integrated dyscalculia assessment (guest=%s)', is_guest)
        return render_template('dyscalculia_test.html', assessment_id=assessment_id, is_guest=is_guest, user_id=user_id)
    
    # Special handling for Dysgraphia Assessment (ID=3)
    if assessment_id == 3:
        app.logger.info('Serving integrated dysgraphia assessment (guest=%s)', is_guest)
        return render_template('dysgraphia_test.html', assessment_id=assessment_id, is_guest=is_guest, user_id=user_id)
    
    # Check if assessment has visual or puzzle questions
    has_visual = False
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('''
            SELECT COUNT(*) as visual_count FROM questions 
            WHERE assessment_id = %s 
            AND question_type IN ('image', 'puzzle', 'visual_matching', 'pattern', 'sequence', 'sorting')
        ''', (assessment_id,))
        result = cursor.fetchone()
        has_visual = result['visual_count'] > 0 if result else False
        cursor.close()
        conn.close()
    except Exception as e:
        app.logger.warning('Could not check visual questions: %s', e)
    
    template = 'assessment_visual.html' if has_visual else 'assessment_test.html'
    return render_template(template, assessment_id=assessment_id, is_guest=is_guest, user_id=user_id)


@app.route('/assessment/results/<int:student_assessment_id>')
@login_required('student')
def view_assessment_results(student_assessment_id):
    """Display assessment results with ML predictions"""
    user_id = session.get('user_id')
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # Verify this assessment belongs to the current user
        cursor.execute('''
            SELECT assessment_id FROM student_assessments 
            WHERE id = %s AND student_id = %s
            AND status = 'completed'
        ''', (student_assessment_id, user_id))
        result = cursor.fetchone()
        assessment_id = result['assessment_id'] if result else None
    finally:
        cursor.close()
        conn.close()
    
    if not assessment_id:
        return "Assessment not found or you don't have access to it.", 404
    
    return render_template('assessment_results.html', 
                          assessment_id=assessment_id,
                          student_assessment_id=student_assessment_id)


@app.route('/dyslexia-results', strict_slashes=False)
@app.route('/dyslexia-results/<int:assessment_id>', strict_slashes=False)
def view_dyslexia_results(assessment_id=1):
    """Display dyslexia assessment results with all attempts (public-friendly)"""
    is_guest = not bool(session.get('user_id'))
    return render_template('dyslexia_results.html', assessment_id=assessment_id, is_guest=is_guest)


@app.route('/dyscalculia-results', strict_slashes=False)
@app.route('/dyscalculia-results/<int:assessment_id>', strict_slashes=False)
def view_dyscalculia_results(assessment_id=2):
    """Display dyscalculia assessment results with all attempts (public-friendly)"""
    is_guest = not bool(session.get('user_id'))
    return render_template('dyscalculia_results.html', assessment_id=assessment_id, is_guest=is_guest)


@app.route('/dysgraphia-results', strict_slashes=False)
@app.route('/dysgraphia-results/<int:assessment_id>', strict_slashes=False)
def view_dysgraphia_results(assessment_id=3):
    """Display dysgraphia assessment results with all attempts (public-friendly)"""
    is_guest = not bool(session.get('user_id'))
    return render_template('dysgraphia_results.html', assessment_id=assessment_id, is_guest=is_guest)



@app.route('/results')
@app.route('/results/')
def view_results_hub_redirect():
    """Redirect generic results request to primary screening report (Dyslexia)"""
    return redirect(url_for('view_dyslexia_results', assessment_id=1))


@app.route('/support')
def support():
    return render_template('support.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = (request.form.get('email') or request.form.get('student_email') or '').strip()
        password = request.form.get('password') or request.form.get('student_password') or ''
        ok, result = authenticate_user(email, password)
        if not ok:
            return render_template('login.html', error=f"Login failed: {result}"), 401
        session['user_id'] = result['id']
        session['role'] = result.get('role', 'student')
        session['name'] = result['name']
        if result.get('profile_photo'):
            session['profile_photo'] = result.get('profile_photo')
        return redirect(url_for('student_profile'))
    return render_template('login.html')


@app.route('/api/student-stats')
@login_required('student')
def get_student_stats():
    """API endpoint to return student statistics for dashboard."""
    user_id = session.get('user_id')
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True, buffered=True)
        
        # Get total assessments completed
        cursor.execute("""
            SELECT COUNT(*) as total FROM student_assessments 
            WHERE student_id = %s AND status = 'completed'
        """, (user_id,))
        t_row = cursor.fetchone()
        total_assessments = t_row.get('total', 0) if isinstance(t_row, dict) else (t_row[0] if t_row else 0)
        
        # Get average score from all completed assessments
        cursor.execute("""
            SELECT AVG(percentage_score) as avg_score FROM student_assessments 
            WHERE student_id = %s AND status = 'completed'
        """, (user_id,))
        result = cursor.fetchone()
        average_score = (result.get('avg_score', 0) if isinstance(result, dict) else (result[0] if result else 0)) or 0
        
        # Get progress by disorder
        cursor.execute("""
            SELECT disorder_type, total_attempts, average_score, last_assessment_date
            FROM student_progress 
            WHERE student_id = %s
        """, (user_id,))
        progress_data = cursor.fetchall()
        
        # Get recent assessments
        cursor.execute("""
            SELECT sa.id, at.name, sa.percentage_score, sa.end_time, at.disorder_type, sa.assessment_id
            FROM student_assessments sa
            JOIN assessment_types at ON sa.assessment_id = at.id
            WHERE sa.student_id = %s AND sa.status = 'completed'
            ORDER BY sa.end_time DESC
            LIMIT 5
        """, (user_id,))
        activities = cursor.fetchall() or []
        
        # Format recent activity for frontend
        recent_activity = []
        for activity in activities:
            aid = activity.get('assessment_id') or 1
            disorder = (activity.get('disorder_type') or '').lower()
            if aid == 1 or 'lex' in disorder:
                res_url = f"/dyslexia-results/{aid}"
            elif aid == 2 or 'calc' in disorder:
                res_url = f"/dyscalculia-results/{aid}"
            elif aid == 3 or 'graph' in disorder:
                res_url = f"/dysgraphia-results/{aid}"
            else:
                res_url = f"/dyslexia-results/{aid}"

            recent_activity.append({
                'id': activity.get('id'),
                'assessment_id': aid,
                'title': activity.get('name', 'Assessment'),
                'score': int(activity.get('percentage_score', 0)),
                'subject': activity.get('disorder_type', 'General'),
                'date': str(activity.get('end_time', 'Recently')),
                'icon': '📖' if 'lex' in disorder else ('🔢' if 'calc' in disorder else ('✍️' if 'graph' in disorder else '📝')),
                'results_url': res_url
            })
        
        # Get performance data (last 5 assessments)
        cursor.execute("""
            SELECT percentage_score, end_time
            FROM student_assessments 
            WHERE student_id = %s AND status = 'completed'
            ORDER BY end_time DESC
            LIMIT 5
        """, (user_id,))
        perf = cursor.fetchall() or []
        performance_data = [{'score': p.get('percentage_score', 0), 'date': p.get('end_time', '')} for p in reversed(perf)]
        
        # Get score distribution
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN percentage_score >= 90 THEN 1 ELSE 0 END) as excellent,
                SUM(CASE WHEN percentage_score >= 80 AND percentage_score < 90 THEN 1 ELSE 0 END) as good,
                SUM(CASE WHEN percentage_score >= 70 AND percentage_score < 80 THEN 1 ELSE 0 END) as average,
                SUM(CASE WHEN percentage_score >= 60 AND percentage_score < 70 THEN 1 ELSE 0 END) as fair,
                SUM(CASE WHEN percentage_score < 60 THEN 1 ELSE 0 END) as poor
            FROM student_assessments 
            WHERE student_id = %s AND status = 'completed'
        """, (user_id,))
        dist = cursor.fetchone() or {}
        score_data = [
            {'label': 'Excellent (90+)', 'count': dist.get('excellent', 0) or 0},
            {'label': 'Good (80-89)', 'count': dist.get('good', 0) or 0},
            {'label': 'Average (70-79)', 'count': dist.get('average', 0) or 0},
            {'label': 'Fair (60-69)', 'count': dist.get('fair', 0) or 0},
            {'label': 'Poor (<60)', 'count': dist.get('poor', 0) or 0}
        ]
        
        return jsonify({
            'total_assessments': total_assessments,
            'average_score': round(float(average_score), 2),
            'recent_activity': recent_activity,
            'performance_data': performance_data,
            'score_distribution': score_data,
            'progress_by_disorder': progress_data
        }), 200
    
    except Exception as e:
        print(f"Error getting student stats: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.route('/api/student-info')
@login_required('student')
def get_student_info():
    """Get current student's information."""
    user_id = session.get('user_id')
    app.logger.info(f'Fetching student info for user_id: {user_id}')
    
    if not user_id:
        return jsonify({'error': 'No user ID in session'}), 401
    
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('''
            SELECT u.id, u.name, u.email, u.contact, u.role, u.class
            FROM users u
            WHERE u.id = %s
        ''', (user_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if row:
            app.logger.info(f'Student info found: {row["name"]} ({row["email"]})')
            return jsonify({
                'id': row.get('id'),
                'name': row.get('name'),
                'email': row.get('email'),
                'contact': row.get('contact'),
                'role': row.get('role'),
                'class': row.get('class')
            }), 200
        else:
            app.logger.warning(f'Student not found for user_id: {user_id}')
            return jsonify({'error': 'Student not found'}), 404
    except Exception as e:
        app.logger.exception('Failed to get student info')
        return jsonify({'error': str(e)}), 500


@app.route('/api/faculties')
def get_faculties():
    """Get list of all faculties (empty since faculty module was removed)."""
    return jsonify({'faculties': []}), 200


@app.route('/api/update-student-info', methods=['POST'])
@login_required('student')
def update_student_info():
    """Update student's personal information including class."""
    user_id = session.get('user_id')
    data = request.get_json() or {}
    
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    contact = data.get('contact', '').strip()
    class_name = data.get('class', '').strip()
    
    # If this is a setup-only update (no name/email provided), fetch existing ones
    is_setup_only = not name and not email
    
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        if is_setup_only:
            # Get existing name and email if not provided
            cursor.execute('SELECT name, email, contact FROM users WHERE id = %s', (user_id,))
            existing = cursor.fetchone()
            if existing:
                name = name or existing.get('name', '')
                email = email or existing.get('email', '')
                contact = contact or existing.get('contact', '')
        
        if not name or not email:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Name and email are required'}), 400
        
        # Check if email is already used by another user (only if email was changed)
        if email:
            cursor.execute('SELECT id FROM users WHERE email = %s AND id != %s', (email, user_id))
            if cursor.fetchone():
                cursor.close()
                conn.close()
                return jsonify({'error': 'Email already in use by another account'}), 400
        
        # Update user info
        upd = conn.cursor()
        upd.execute('''
            UPDATE users 
            SET name = %s, email = %s, contact = %s, class = %s
            WHERE id = %s
        ''', (name, email, contact, class_name, user_id))
        
        conn.commit()
        upd.close()
        cursor.close()
        conn.close()
        
        # Update session
        session['name'] = name
        
        return jsonify({'status': 'ok', 'message': 'Information updated successfully'}), 200
    except Exception as e:
        app.logger.exception('Failed to update student info')
        return jsonify({'error': str(e)}), 500


@app.route('/api/setup-profile', methods=['POST'])
@login_required('student')
def setup_profile():
    """Complete initial student profile setup with class."""
    user_id = session.get('user_id')
    data = request.get_json() or {}
    
    class_name = data.get('class', '').strip()
    
    if not class_name:
        return jsonify({'error': 'Class is required'}), 400
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Update class
        cursor.execute('''
            UPDATE users 
            SET class = %s
            WHERE id = %s
        ''', (class_name, user_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'status': 'ok', 'message': 'Profile setup completed successfully'}), 200
    except Exception as e:
        app.logger.exception('Failed to setup profile')
        return jsonify({'error': str(e)}), 500


@app.route('/api/student/profile', methods=['GET'])
@login_required('student')
def get_student_profile():
    """Get complete student profile information including assessments and statistics."""
    user_id = session.get('user_id')
    print(f'[DEBUG] Getting profile for user_id: {user_id}')
    
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get student profile info
        cursor.execute('''
            SELECT u.id, u.name, u.email, u.contact, u.role, u.class,
                   u.profile_photo, u.created_at
            FROM users u
            WHERE u.id = %s AND u.role = 'student'
        ''', (user_id,))
        
        profile = cursor.fetchone()
        print(f'[DEBUG] Profile retrieved: {profile}')
        
        if not profile:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Student profile not found'}), 404
        
        # Get all assessment attempts for this student
        cursor.execute('''
            SELECT 
                sa.id,
                sa.assessment_id,
                sa.percentage_score,
                sa.status,
                sa.created_at,
                at.disorder_type
            FROM student_assessments sa
            JOIN assessment_types at ON sa.assessment_id = at.id
            WHERE sa.student_id = %s
            ORDER BY sa.created_at DESC
        ''', (user_id,))
        sa_rows = cursor.fetchall() or []
        
        cursor.close()
        conn.close()
        
        # Calculate statistics
        total_assessments = len(sa_rows)
        all_scores = [float(r.get('percentage_score') or 0.0) for r in sa_rows if r.get('percentage_score') is not None]
        avg_score = round(float(sum(all_scores) / len(all_scores)), 1) if all_scores else 0.0
        
        aid_to_disorder = {1: 'dyslexia', 2: 'dyscalculia', 3: 'dysgraphia'}
        
        disorder_map = {
            'dyslexia': {'disorder': 'dyslexia', 'assessment_id': 1, 'attempts': 0, 'average_score': 0.0, 'best_score': 0.0, 'lowest_score': 0.0, '_scores': []},
            'dyscalculia': {'disorder': 'dyscalculia', 'assessment_id': 2, 'attempts': 0, 'average_score': 0.0, 'best_score': 0.0, 'lowest_score': 0.0, '_scores': []},
            'dysgraphia': {'disorder': 'dysgraphia', 'assessment_id': 3, 'attempts': 0, 'average_score': 0.0, 'best_score': 0.0, 'lowest_score': 0.0, '_scores': []}
        }
        
        last_date = None
        attempted_disorders = set()
        
        for row in sa_rows:
            aid = row.get('assessment_id') or 1
            dt = (row.get('disorder_type') or aid_to_disorder.get(aid, 'dyslexia')).lower()
            attempted_disorders.add(dt)
            if dt not in disorder_map:
                disorder_map[dt] = {'disorder': dt, 'assessment_id': aid, 'attempts': 0, 'average_score': 0.0, 'best_score': 0.0, 'lowest_score': 0.0, '_scores': []}
            
            score = float(row.get('percentage_score') or 0.0)
            disorder_map[dt]['_scores'].append(score)
            disorder_map[dt]['assessment_id'] = aid
            if not last_date and row.get('created_at'):
                last_date = row.get('created_at')
        
        for dt, dinfo in disorder_map.items():
            scores = dinfo.pop('_scores', [])
            dinfo['attempts'] = len(scores)
            if scores:
                dinfo['average_score'] = round(float(sum(scores) / len(scores)), 1)
                dinfo['best_score'] = round(float(max(scores)), 1)
                dinfo['lowest_score'] = round(float(min(scores)), 1)
        
        return jsonify({
            'status': 'ok',
            'profile': {
                'id': profile.get('id'),
                'name': profile.get('name'),
                'email': profile.get('email'),
                'contact': profile.get('contact'),
                'role': profile.get('role'),
                'class': profile.get('class'),
                'profile_photo': profile.get('profile_photo'),
                'created_at': str(profile.get('created_at')) if profile.get('created_at') else None
            },
            'statistics': {
                'total_assessments': total_assessments,
                'average_score': avg_score,
                'disorders_attempted': len(attempted_disorders),
                'last_assessment_date': str(last_date) if last_date else None
            },
            'disorder_breakdown': list(disorder_map.values())
        }), 200
        
    except Exception as e:
        app.logger.exception('Failed to get student profile')
        return jsonify({'error': str(e)}), 500


@app.route('/api/student/profile', methods=['PUT'])
@login_required('student')
def update_student_profile():
    """Update student profile information (name, email, contact, class)."""
    user_id = session.get('user_id')
    data = request.get_json()
    
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get current profile
        cursor.execute('SELECT id, name, email, contact, class FROM users WHERE id = %s', (user_id,))
        current = cursor.fetchone()
        
        if not current:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Student profile not found'}), 404
        
        # Prepare updated values
        name = data.get('name', current.get('name')).strip()
        email = data.get('email', current.get('email')).strip()
        contact = data.get('contact', current.get('contact', '')).strip()
        class_name = data.get('class', current.get('class', '')).strip()
        
        # Validate required fields
        if not name or not email:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Name and email are required'}), 400
        
        # Check if email is already used by another user
        if email != current.get('email'):
            cursor.execute('SELECT id FROM users WHERE email = %s AND id != %s', (email, user_id))
            if cursor.fetchone():
                cursor.close()
                conn.close()
                return jsonify({'error': 'Email already in use'}), 409
        
        # Update profile
        upd = conn.cursor()
        upd.execute('''
            UPDATE users 
            SET name = %s, email = %s, contact = %s, class = %s
            WHERE id = %s
        ''', (name, email, contact, class_name, user_id))
        
        conn.commit()
        upd.close()
        
        # Update session
        session['name'] = name
        session['email'] = email
        
        # Fetch and return updated profile
        cursor.execute('''
            SELECT u.id, u.name, u.email, u.contact, u.role, u.class
            FROM users u
            WHERE u.id = %s
        ''', (user_id,))
        
        updated_profile = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return jsonify({
            'status': 'ok',
            'message': 'Profile updated successfully',
            'profile': {
                'id': updated_profile.get('id'),
                'name': updated_profile.get('name'),
                'email': updated_profile.get('email'),
                'contact': updated_profile.get('contact'),
                'role': updated_profile.get('role'),
                'class': updated_profile.get('class')
            }
        }), 200
        
    except Exception as e:
        app.logger.exception('Failed to update student profile')
        return jsonify({'error': str(e)}), 500


@app.route('/api/update-password', methods=['POST'])
@login_required('student')
def update_password():
    """Update student's password."""
    user_id = session.get('user_id')
    data = request.get_json()
    
    current_password = data.get('current_password', '')
    new_password = data.get('new_password', '')
    
    if not current_password or not new_password:
        return jsonify({'error': 'Both passwords are required'}), 400
    
    if len(new_password) < 6:
        return jsonify({'error': 'New password must be at least 6 characters long'}), 400
    
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get current password hash
        cursor.execute('SELECT password FROM users WHERE id = %s', (user_id,))
        row = cursor.fetchone()
        
        if not row:
            cursor.close()
            conn.close()
            return jsonify({'error': 'User not found'}), 404
        
        # Verify current password
        if not check_password_hash(row.get('password', ''), current_password):
            cursor.close()
            conn.close()
            return jsonify({'error': 'Current password is incorrect'}), 401
        
        # Hash new password
        hashed_password = generate_password_hash(new_password)
        
        # Update password
        upd = conn.cursor()
        upd.execute('UPDATE users SET password = %s WHERE id = %s', (hashed_password, user_id))
        conn.commit()
        upd.close()
        cursor.close()
        conn.close()
        
        return jsonify({'status': 'ok', 'message': 'Password updated successfully'}), 200
    except Exception as e:
        app.logger.exception('Failed to update password')
        return jsonify({'error': str(e)}), 500



@app.route('/api/student/dashboard', methods=['GET'])
@login_required('student')
def student_dashboard_api():
    """Get student dashboard data including assessments, progress, and recommendations"""
    try:
        student_id = session.get('user_id')
        
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Helper function to determine risk level
        def get_risk_level(score):
            if score is None:
                return 'Medium'
            try:
                score_f = float(score)
                if score_f >= 75:
                    return 'Low'
                elif score_f >= 50:
                    return 'Medium'
                else:
                    return 'High'
            except Exception:
                return 'Medium'
        
        # Get recent assessments with disorder type from assessment_types
        cursor.execute('''
            SELECT sa.id, at.disorder_type, sa.percentage_score as score, 
                   sa.created_at as date, sa.status
            FROM student_assessments sa
            JOIN assessment_types at ON sa.assessment_id = at.id
            WHERE sa.student_id = %s
            ORDER BY sa.created_at DESC
            LIMIT 5
        ''', (student_id,))
        recent = cursor.fetchall() or []
        
        # Format recent assessments
        for item in recent:
            item['score'] = float(item.get('score')) if item.get('score') is not None else None
            item['date'] = str(item.get('date')) if item.get('date') else ''
            item['disorder_type'] = item.get('disorder_type') or 'General'
            item['risk_level'] = get_risk_level(item.get('score'))
        
        # Get full assessment history
        cursor.execute('''
            SELECT sa.id, at.disorder_type, sa.percentage_score as score, 
                   sa.created_at as date, sa.status
            FROM student_assessments sa
            JOIN assessment_types at ON sa.assessment_id = at.id
            WHERE sa.student_id = %s
            ORDER BY sa.created_at DESC
        ''', (student_id,))
        history = cursor.fetchall() or []
        
        # Format history
        for item in history:
            item['score'] = float(item.get('score')) if item.get('score') is not None else None
            item['date'] = str(item.get('date')) if item.get('date') else ''
            item['disorder_type'] = item.get('disorder_type') or 'General'
            item['risk_level'] = get_risk_level(item.get('score'))
        
        # Get quick stats
        cursor.execute('''
            SELECT COUNT(*) as total_assessments, 
                   AVG(percentage_score) as average_score
            FROM student_assessments sa
            WHERE sa.student_id = %s AND sa.status = 'completed'
        ''', (student_id,))
        stats_result = cursor.fetchone() or {}
        
        # Get disorder-wise progress
        cursor.execute('''
            SELECT at.disorder_type,
                   COUNT(*) as attempts,
                   AVG(percentage_score) as average_score
            FROM student_assessments sa
            JOIN assessment_types at ON sa.assessment_id = at.id
            WHERE sa.student_id = %s AND sa.status = 'completed'
            GROUP BY at.disorder_type
        ''', (student_id,))
        disorder_progress = cursor.fetchall() or []
        
        # Get recommendations
        recommendations = []
        try:
            cursor.execute('''
                SELECT id, disorder_type, recommendation_text as title, 
                       recommendation_details as description, created_at as date
                FROM recommendations
                WHERE student_id = %s
                ORDER BY created_at DESC
                LIMIT 10
            ''', (student_id,))
            recommendations = cursor.fetchall() or []
        except Exception as rec_err:
            app.logger.warning('Could not query recommendations: %s', rec_err)
        
        # Format recommendations
        formatted_recs = []
        for idx, rec in enumerate(recommendations, 1):
            formatted_recs.append({
                'index': idx,
                'id': rec.get('id'),
                'title': rec.get('title') or 'Recommendation',
                'description': rec.get('description') or '',
                'disorder': rec.get('disorder_type') or 'General',
                'date': str(rec.get('date')) if rec.get('date') else '',
                'indicator': '⚠️',
                'risk_class': 'high'
            })
        
        # Build progress dict
        progress_data = {}
        for disorder_type in ['dyslexia', 'dyscalculia', 'dysgraphia']:
            progress_data[disorder_type] = {
                'attempts': 0,
                'average_score': None,
                'risk_level': '-',
                'history': []
            }
        
        for dp in disorder_progress:
            dt = (dp.get('disorder_type') or '').lower().strip()
            if dt in progress_data:
                progress_data[dt]['attempts'] = dp.get('attempts', 0)
                avg_val = float(dp.get('average_score')) if dp.get('average_score') is not None else None
                progress_data[dt]['average_score'] = avg_val
                progress_data[dt]['risk_level'] = get_risk_level(avg_val)
        
        # Get history for mini charts
        for disorder_type in ['dyslexia', 'dyscalculia', 'dysgraphia']:
            cursor.execute('''
                SELECT sa.percentage_score as score, sa.created_at as date, sa.status
                FROM student_assessments sa
                JOIN assessment_types at ON sa.assessment_id = at.id
                WHERE sa.student_id = %s AND LOWER(at.disorder_type) = %s
                  AND sa.status = 'completed'
                ORDER BY sa.created_at
            ''', (student_id, disorder_type))
            scores = cursor.fetchall() or []
            for s in scores:
                s['date'] = str(s.get('date')) if s.get('date') else ''
                s['score'] = float(s.get('score')) if s.get('score') is not None else 0
            progress_data[disorder_type]['history'] = scores
        
        cursor.close()
        conn.close()
        
        tot_assess = stats_result.get('total_assessments') or (len(history) if history else 0)
        avg_score_val = float(stats_result.get('average_score') or 0)
        
        return jsonify({
            'status': 'ok',
            'stats': {
                'total_assessments': tot_assess,
                'average_score': round(avg_score_val, 2),
                'overall_risk': get_risk_level(avg_score_val if tot_assess > 0 else None),
                'latest_disorder': recent[0].get('disorder_type', '-') if recent else '-'
            },
            'recent': recent,
            'history': history,
            'progress': progress_data,
            'recommendations': formatted_recs
        }), 200
        
    except Exception as e:
        app.logger.exception('Failed to get student dashboard data')
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/student/assessment/<int:assessment_id>', methods=['GET'])
@login_required('student')
def student_assessment_details(assessment_id):
    """Get detailed information about a specific student assessment"""
    try:
        student_id = session.get('user_id')
        
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Helper function to determine risk level
        def get_risk_level(score):
            if score is None:
                return 'Medium'
            score = float(score)
            if score >= 75:
                return 'Low'
            elif score >= 50:
                return 'Medium'
            else:
                return 'High'
        
        # Get assessment with disorder type from assessment_types
        cursor.execute('''
            SELECT sa.id, at.disorder_type, sa.percentage_score as score, 
                   sa.created_at as date
            FROM student_assessments sa
            JOIN assessment_types at ON sa.assessment_id = at.id
            WHERE sa.id = %s AND sa.student_id = %s
        ''', (assessment_id, student_id))
        assessment = cursor.fetchone()
        
        if not assessment:
            cursor.close()
            conn.close()
            return jsonify({'status': 'error', 'error': 'Assessment not found'}), 404
        
        # Get related recommendations
        recommendations = []
        cursor.execute('''
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = 'recommendations' LIMIT 1
        ''')
        has_recommendations = cursor.fetchone()
        
        if has_recommendations:
            cursor.execute('''
                SELECT recommendation_text, recommendation_details
                FROM recommendations
                WHERE student_id = %s AND disorder_type = %s
                ORDER BY created_at DESC
                LIMIT 3
            ''', (student_id, assessment['disorder_type']))
            recommendations = cursor.fetchall() or []
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'status': 'ok',
            'assessment': {
                'id': assessment['id'],
                'disorder_type': assessment['disorder_type'],
                'score': assessment['score'],
                'risk_level': get_risk_level(assessment['score']),
                'date': assessment['date'],
                'recommendations': [
                    {'recommendation_text': r.get('recommendation_text', r)} 
                    for r in recommendations
                ]
            }
        }), 200
        
    except Exception as e:
        app.logger.exception('Failed to get assessment details')
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/faculty-dashboard', methods=['GET'])
def faculty_dashboard_page():
    return redirect(url_for('student_dashboard'))


if __name__ == '__main__':
    # Keep debug=True for development; set to False in production
    app.run(debug=True)
