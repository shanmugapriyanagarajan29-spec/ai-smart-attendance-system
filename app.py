"""AI Based Smart Attendance System - Flask backend."""
import base64, csv, hmac, io, os, re, sqlite3
from datetime import datetime
from functools import wraps

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, session, url_for)

BASE = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(BASE, "private_data")      # NOT served to the web
FACES = os.path.join(DATA, "faces")            # cropped grayscale faces only
DB = os.path.join(DATA, "attendance.db")
os.makedirs(FACES, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  MAX_CONTENT_LENGTH=12 * 1024 * 1024)
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "admin123")
THRESHOLD = float(os.environ.get("FACE_THRESHOLD", "70"))  # lower = stricter
ID_RE = re.compile(r"^[A-Za-z0-9_-]{2,20}$")


# ---------- database ----------
def q(sql, args=(), one=False, commit=False):
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    try:
        cur = con.execute(sql, args)
        if commit:
            con.commit()
            return cur.lastrowid
        rows = cur.fetchall()
        return (rows[0] if rows else None) if one else rows
    finally:
        con.close()


def init_db():
    con = sqlite3.connect(DB)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS students(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      student_id TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
      department TEXT NOT NULL, consent_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS attendance(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      student_pk INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
      date TEXT NOT NULL, time TEXT NOT NULL, UNIQUE(student_pk, date));
    """)
    con.commit()
    con.close()


init_db()


# ---------- auth ----------
def admin_required(f):
    @wraps(f)
    def wrap(*a, **k):
        if not session.get("admin"):
            if request.path.startswith("/api/"):
                return jsonify(ok=False, message="Please log in."), 401
            return redirect(url_for("login"))
        return f(*a, **k)
    return wrap


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        u, p = request.form.get("username", ""), request.form.get("password", "")
        if hmac.compare_digest(u, ADMIN_USER) and hmac.compare_digest(p, ADMIN_PASS):
            session["admin"] = True
            return redirect(url_for("records"))
        error = "Invalid username or password."
    return render_template("login.html", page="login", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ---------- face helpers ----------
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def read_face(data_url):
    """Decode a browser image, find the largest face, return 200x200 gray crop."""
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None
    if img is None:
        return None
    img = cv2.equalizeHist(img)
    faces = cascade.detectMultiScale(img, 1.2, 5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return cv2.resize(img[y:y + h, x:x + w], (200, 200))


_cache = {"sig": None, "model": None}


def get_model():
    """Train the LBPH recognizer (re-trains only when face files change)."""
    files = []
    for pk in os.listdir(FACES):
        folder = os.path.join(FACES, pk)
        if pk.isdigit() and os.path.isdir(folder):
            files += [(int(pk), os.path.join(folder, f)) for f in os.listdir(folder)]
    sig = (len(files), max([os.path.getmtime(p) for _, p in files], default=0))
    if sig == _cache["sig"]:
        return _cache["model"]
    model = None
    if files:
        imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for _, p in files]
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(imgs, np.array([pk for pk, _ in files]))
    _cache.update(sig=sig, model=model)
    return model


# ---------- pages ----------
@app.route("/")
def index():
    return render_template("index.html", page="index")


@app.route("/register")
@admin_required
def register():
    return render_template("register.html", page="register")


@app.route("/records")
@admin_required
def records():
    return render_template("records.html", page="records")


# ---------- API ----------
@app.post("/api/register")
@admin_required
def api_register():
    d = request.get_json(silent=True) or {}
    sid = str(d.get("student_id", "")).strip()
    name = str(d.get("name", "")).strip()
    dept = str(d.get("department", "")).strip()
    images = d.get("images") or []
    if not d.get("consent"):
        return jsonify(ok=False, message="Student consent is required."), 400
    if not ID_RE.match(sid):
        return jsonify(ok=False, message="Student ID: 2-20 letters, digits, - or _."), 400
    if not (2 <= len(name) <= 80 and 2 <= len(dept) <= 80):
        return jsonify(ok=False, message="Enter a valid name and department."), 400
    if len(images) < 3 or len(images) > 8:
        return jsonify(ok=False, message="Capture 3 to 8 photos."), 400
    if q("SELECT 1 FROM students WHERE student_id=?", (sid,), one=True):
        return jsonify(ok=False, message="This Student ID is already registered."), 409
    faces = [read_face(i) for i in images]
    if any(f is None for f in faces):
        return jsonify(ok=False, message="A face was not detected in every photo. "
                       "Retake the photos with good lighting."), 400
    pk = q("INSERT INTO students(student_id,name,department,consent_at) VALUES(?,?,?,?)",
           (sid, name, dept, datetime.now().isoformat(timespec="seconds")), commit=True)
    folder = os.path.join(FACES, str(pk))
    os.makedirs(folder, exist_ok=True)
    for n, f in enumerate(faces):
        cv2.imwrite(os.path.join(folder, f"{n}.png"), f)
    return jsonify(ok=True, message=f"{name} registered successfully.")


@app.post("/api/mark")
def api_mark():
    d = request.get_json(silent=True) or {}
    if not d.get("consent"):
        return jsonify(ok=False, message="Consent is required to use face recognition."), 400
    face = read_face(d.get("image", ""))
    if face is None:
        return jsonify(ok=False, message="No face detected. Look at the camera in good light.")
    model = get_model()
    if model is None:
        return jsonify(ok=False, message="No students are registered yet.")
    label, conf = model.predict(face)
    s = q("SELECT * FROM students WHERE id=?", (int(label),), one=True)
    if conf > THRESHOLD or s is None:
        return jsonify(ok=False, message="Face not recognised. Please register first.")
    now = datetime.now()
    try:
        q("INSERT INTO attendance(student_pk,date,time) VALUES(?,?,?)",
          (s["id"], now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")), commit=True)
        return jsonify(ok=True, message=f"Welcome, {s['name']}! Attendance marked at "
                       f"{now.strftime('%H:%M:%S')}.")
    except sqlite3.IntegrityError:
        return jsonify(ok=True, message=f"{s['name']}, your attendance is already marked today.")


def _filters():
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    text = request.args.get("q", "").strip()
    sql, args = " WHERE 1=1", []
    if start:
        sql += " AND a.date>=?"; args.append(start)
    if end:
        sql += " AND a.date<=?"; args.append(end)
    if text:
        sql += " AND (s.name LIKE ? OR s.student_id LIKE ?)"; args += [f"%{text}%"] * 2
    return sql, args


def _records():
    sql, args = _filters()
    return q("SELECT s.student_id, s.name, s.department, a.date, a.time FROM attendance a "
             "JOIN students s ON s.id=a.student_pk" + sql + " ORDER BY a.date DESC, a.time DESC",
             args)


@app.get("/api/records")
@admin_required
def api_records():
    return jsonify([dict(r) for r in _records()])


@app.get("/api/report")
@admin_required
def api_report():
    sql, args = _filters()
    days = q("SELECT COUNT(DISTINCT a.date) c FROM attendance a JOIN students s "
             "ON s.id=a.student_pk" + sql, args, one=True)["c"]
    return jsonify(total_days=days, students=_report_rows(days))


def _report_rows(days):
    start, end = request.args.get("start", ""), request.args.get("end", "")
    out = []
    for s in q("SELECT * FROM students ORDER BY name"):
        sql, args = "SELECT COUNT(*) c FROM attendance WHERE student_pk=?", [s["id"]]
        if start:
            sql += " AND date>=?"; args.append(start)
        if end:
            sql += " AND date<=?"; args.append(end)
        p = q(sql, args, one=True)["c"]
        out.append(dict(student_id=s["student_id"], name=s["name"], department=s["department"],
                        present=p, percent=round(100 * p / days, 1) if days else 0))
    return out


@app.get("/api/export.csv")
@admin_required
def export_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Student ID", "Name", "Department", "Date", "Time"])
    for r in _records():
        w.writerow([r["student_id"], r["name"], r["department"], r["date"], r["time"]])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=attendance.csv"})


@app.get("/api/students")
@admin_required
def api_students():
    return jsonify([dict(r) for r in q(
        "SELECT student_id,name,department,consent_at FROM students ORDER BY name")])


@app.delete("/api/students/<sid>")
@admin_required
def api_delete(sid):
    s = q("SELECT id FROM students WHERE student_id=?", (sid,), one=True)
    if not s:
        return jsonify(ok=False, message="Student not found."), 404
    q("DELETE FROM students WHERE id=?", (s["id"],), commit=True)  # cascades attendance
    folder = os.path.join(FACES, str(s["id"]))
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
        os.rmdir(folder)
    return jsonify(ok=True, message="Student and all face data deleted.")


@app.after_request
def secure(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Permissions-Policy"] = "camera=(self)"
    return resp


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)