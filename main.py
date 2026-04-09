import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import json
import bs4 as bs
import urllib.request
import pickle
import requests
import sqlite3
import hashlib
import os
from datetime import datetime
from functools import wraps
import random
import threading
from icecream import ic 

# ─────────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'netflix-recommender-secret-2024')

DB_PATH = 'users.db'

# ─────────────────────────────────────────────
#  Online RL Agent Import
# ─────────────────────────────────────────────
from online_rl import DQNAgent

# ─────────────────────────────────────────────
#  Database
# ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    UNIQUE NOT NULL,
            email       TEXT    UNIQUE NOT NULL,
            password    TEXT    NOT NULL,
            created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS user_interactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            movie_title     TEXT    NOT NULL,
            action          TEXT    NOT NULL,
            rating          REAL    DEFAULT NULL,
            watch_seconds   INTEGER DEFAULT NULL,
            watch_percent   REAL    DEFAULT NULL,
            from_rec        INTEGER DEFAULT 0,
            rec_position    INTEGER DEFAULT NULL,
            session_id      TEXT    DEFAULT NULL,
            timestamp       TEXT    DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS user_ratings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            movie_title TEXT    NOT NULL,
            rating      REAL    NOT NULL,
            timestamp   TEXT    DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, movie_title),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS rl_rewards (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            movie_title     TEXT    NOT NULL,
            action          TEXT    NOT NULL,
            reward_value    REAL    NOT NULL,
            context_json    TEXT    DEFAULT NULL,
            timestamp       TEXT    DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()

    # ── MIGRATION: thêm cột mới vào bảng cũ nếu chưa có ──────────────────────
    new_columns = [
        ("user_interactions", "watch_seconds",  "INTEGER DEFAULT NULL"),
        ("user_interactions", "watch_percent",  "REAL    DEFAULT NULL"),
        ("user_interactions", "from_rec",       "INTEGER DEFAULT 0"),
        ("user_interactions", "rec_position",   "INTEGER DEFAULT NULL"),
        ("user_interactions", "session_id",     "TEXT    DEFAULT NULL"),
    ]
    for table, col, col_def in new_columns:
        try:
            c.execute("ALTER TABLE rl_rewards ADD COLUMN watch_percent REAL DEFAULT NULL")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # cột đã tồn tại


    conn.commit()
    conn.close()

init_db()

# ─────────────────────────────────────────────
#  RL Reward Function
# ─────────────────────────────────────────────
REWARD_TABLE = {
    'watch_end_high':       +3.0,
    'watch_end_mid':        +1.5,
    'watch_end_low':        +0.2,
    'rewatch':              +4.0,
    'rate_high':            +2.5,
    'rate_mid':             +1.0,
    'rate_low':             -1.5,
    'watchlist_add':        +1.0,
    'watchlist_remove':     -0.5,
    'rec_converted':        +1.5,
    'quick_exit':           -1.0,
    'search':               +0.1,
    'view':                 +0.2,
}

# Tính reward dựa trên hành động và thông tin thêm (như phần trăm xem, rating, v.v.)
def compute_reward(action, extra=None):
    extra = extra or {}
    if action == 'watch_end':
        pct = extra.get('watch_percent', 0)
        if pct >= 0.70:
            return REWARD_TABLE['watch_end_high']
        elif pct >= 0.30:
            return REWARD_TABLE['watch_end_mid']
        else:
            return REWARD_TABLE['watch_end_low']
    elif action == 'rate':
        r = extra.get('rating', 5)
        if r >= 8:
            return REWARD_TABLE['rate_high']
        elif r >= 5:
            return REWARD_TABLE['rate_mid']
        else:
            return REWARD_TABLE['rate_low']
    elif action in REWARD_TABLE:
        return REWARD_TABLE[action]
    return 0.0

# Ghi log reward vào database
def log_reward(user_id, movie_title, action, reward_value, watch_percent=None, context=None):
    conn = get_db()
    conn.execute(
        """INSERT INTO rl_rewards (user_id, movie_title, action, reward_value, watch_percent, context_json)
           VALUES (?,?,?,?,?,?)""",
        (user_id, movie_title.lower(), action, reward_value, watch_percent,
         json.dumps(context) if context else None))
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
#  Helpers – auth
# ─────────────────────────────────────────────
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────
#  Content-Based similarity (lazy loaded)
# ─────────────────────────────────────────────
_cb_data       = None
_cb_similarity = None

def get_cb_data():
    global _cb_data, _cb_similarity
    if _cb_data is None:
        _cb_data = pd.read_csv('main_data.csv')
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(_cb_data['comb'])
        _cb_similarity = cosine_similarity(count_matrix)
        # Initialize RL agent after data is loaded
        if rl_agent is None:
            init_rl_agent()
    return _cb_data, _cb_similarity

# ─────────────────────────────────────────────
#  NLP model (sentiment)
# ─────────────────────────────────────────────
try:
    clf        = pickle.load(open('nlp_model.pkl', 'rb'))
    vectorizer_nlp = pickle.load(open('tranform.pkl', 'rb'))
    NLP_READY  = True
except Exception:
    NLP_READY  = False

# ─────────────────────────────────────────────
#  Cold-start: top-rated movies
# ─────────────────────────────────────────────
def get_top_rated_movies(n=12):
    data, _ = get_cb_data()
    print(data.head())
    if 'vote_average' in data.columns:
        print("Using vote_average for top-rated movies")
        top = (data[['movie_title', 'vote_average']]
               .dropna()
               .sort_values('vote_average', ascending=False)
               .head(n))
        return top['movie_title'].tolist()
    print("vote_average not found, using first N movies as fallback")
    return data['movie_title'].head(n).tolist()

# ─────────────────────────────────────────────
#  Collaborative Filtering (Matrix Factorization)
# ─────────────────────────────────────────────
def build_cf_matrix():
    conn = get_db()
    explicit = pd.read_sql_query(
        "SELECT user_id, movie_title, rating FROM user_ratings", conn)
    implicit = pd.read_sql_query(
        """SELECT user_id, movie_title, SUM(reward_value) as score
           FROM rl_rewards
           WHERE reward_value > 0
           GROUP BY user_id, movie_title""", conn)
    conn.close()

    if explicit.empty and implicit.empty:
        return None, None, None

    frames = []
    if not explicit.empty:
        frames.append(explicit.rename(columns={'rating': 'score'}))
    if not implicit.empty:
        imp = implicit.copy()
        imp['score'] = imp['score'].clip(upper=10).astype(float)
        frames.append(imp)

    combined = pd.concat(frames)
    combined = (combined
                .groupby(['user_id', 'movie_title'], as_index=False)['score']
                .max())

    pivot = combined.pivot(index='user_id', columns='movie_title', values='score').fillna(0)
    return pivot, pivot.index.tolist(), pivot.columns.tolist()

def cf_recommend(user_id, n=12):
    pivot, user_ids, movie_titles = build_cf_matrix()
    if pivot is None or user_id not in user_ids:
        return []
    mat    = csr_matrix(pivot.values)
    k      = min(20, min(mat.shape) - 1)
    if k < 1:
        return []
    U, sigma, Vt = svds(mat.astype(float), k=k)
    sigma_diag   = np.diag(sigma)
    all_preds    = np.dot(np.dot(U, sigma_diag), Vt)
    preds_df     = pd.DataFrame(all_preds, index=user_ids, columns=movie_titles)
    user_row     = pivot.loc[user_id]
    already_seen = user_row[user_row > 0].index.tolist()
    user_preds = (preds_df.loc[user_id]
                  .drop(labels=already_seen, errors='ignore')
                  .sort_values(ascending=False)
                  .head(n))
    return user_preds.index.tolist()

# ─────────────────────────────────────────────
#  Content-Based recommend (single movie)
# ─────────────────────────────────────────────
def cb_recommend(movie_title, n=12):
    data, similarity = get_cb_data()
    m = movie_title.lower()
    if m not in data['movie_title'].unique():
        return []
    idx  = data.loc[data['movie_title'] == m].index[0]
    lst  = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    lst  = lst[1:n+1]
    return [data['movie_title'][i[0]] for i in lst]

# ─────────────────────────────────────────────
#  Hybrid Recommendation (fallback)
# ─────────────────────────────────────────────
def hybrid_recommend(user_id, movie_title, n=12, alpha=0.5):
    data, similarity = get_cb_data()
    all_movies = data['movie_title'].tolist()
    m = movie_title.lower()
    if m in data['movie_title'].unique():
        idx     = data.loc[data['movie_title'] == m].index[0]
        cb_raw  = similarity[idx]
        cb_scores = pd.Series(cb_raw, index=data['movie_title'])
    else:
        cb_scores = pd.Series(0.0, index=all_movies)
    cb_max = cb_scores.max()
    if cb_max > 0:
        cb_scores = cb_scores / cb_max

    pivot, user_ids, movie_titles = build_cf_matrix()
    cf_scores = pd.Series(0.0, index=all_movies)

    if pivot is not None and user_id in user_ids:
        k   = min(20, min(pivot.shape) - 1)
        if k >= 1:
            mat = csr_matrix(pivot.values.astype(float))
            U, sigma, Vt = svds(mat, k=k)
            preds_df = pd.DataFrame(
                np.dot(np.dot(U, np.diag(sigma)), Vt),
                index=user_ids, columns=movie_titles)
            user_preds = preds_df.loc[user_id]
            cf_max = user_preds.max()
            if cf_max > 0:
                user_preds = user_preds / cf_max
            cf_scores.update(user_preds)
        actual_alpha = alpha
    else:
        actual_alpha = 0.0

    hybrid = (1 - actual_alpha) * cb_scores + actual_alpha * cf_scores
    hybrid = hybrid.drop(index=m, errors='ignore')

    conn = get_db()
    bounced = conn.execute(
        """SELECT DISTINCT movie_title FROM user_interactions
           WHERE user_id=? AND action='quick_exit'""",
        (user_id,)).fetchall()
    conn.close()
    bounce_list = [r['movie_title'] for r in bounced]
    hybrid = hybrid.drop(index=bounce_list, errors='ignore')

    return hybrid.nlargest(n).index.tolist()

# ─────────────────────────────────────────────
#  Online RL Agent & State Management
# ─────────────────────────────────────────────
rl_agent = None
rl_movie_features_dict = {}   # title -> genre vector
rl_movie_titles = []          # list of titles
rl_genre_list = []            # list of genre names
user_states = {}              # user_id -> (state_vector, last_update)
user_states_lock = threading.Lock()

def init_rl_agent():
    global rl_agent, rl_movie_titles, rl_genre_list, movie_to_idx

    data, _ = get_cb_data()

    # =========================
    # 1. Build genre list
    # =========================
    all_genres = set()
    for g in data['genres'].dropna():
        for genre in g.split(','):
            all_genres.add(genre.strip())

    rl_genre_list = sorted(all_genres)

    print(f"Initializing RL agent with {len(data)} movies and {len(rl_genre_list)} genres")

    # =========================
    # 2. Movie titles
    # =========================
    rl_movie_titles = [row['movie_title'].lower() for _, row in data.iterrows()]

    # 🔥 mapping cực quan trọng
    movie_to_idx = {title: i for i, title in enumerate(rl_movie_titles)}
    ic(len(movie_to_idx), list(movie_to_idx.keys())[:5])

    # =========================
    # 3. State dim
    # =========================
    state_dim = len(rl_genre_list) + 3
    num_actions = len(rl_movie_titles)

    print(f"State dim: {state_dim}, Num actions: {num_actions}")

    # =========================
    # 4. Init agent (NEW)
    # =========================
    # state_dim = 1255
    # movie
    rl_agent = DQNAgent(
        state_dim=state_dim,
        movie_titles=rl_movie_titles,
        device='cuda'
    )
    checkpoint_path = 'rl_checkpoint.pt'
    if os.path.exists(checkpoint_path):
        success = rl_agent.load_checkpoint(checkpoint_path)
        if success:
            print("✅ Loaded RL agent from checkpoint")
        else:
            print("⚠️ Checkpoint incompatible, starting fresh")
    else:
        print("🆕 No checkpoint found, initializing new RL agent")

    print("Dueling Double DQN agent initialized.")

def get_user_state(user_id):
    """Retrieve current state vector for user from cache or build from DB."""
    with user_states_lock:
        if user_id in user_states:
            return user_states[user_id][0]
    # Not in cache, build from DB
    state = build_state_from_db(user_id)
    with user_states_lock:
        user_states[user_id] = (state, datetime.now())
    return state

def update_user_state(user_id, movie_title, reward, watch_percent=None):
    """Update user state after an interaction."""
    # Get current state
    with user_states_lock:
        if user_id in user_states:
            state, _ = user_states[user_id]
        else:
            state = build_state_from_db(user_id)
            user_states[user_id] = (state, datetime.now())

    # Update state based on new interaction
    movie = movie_title.lower()
    if movie in rl_movie_features_dict:
        genre_vec = rl_movie_features_dict[movie]
        total_reward = state[-3]
        count = state[-2]
        if reward > 0:
            new_total = total_reward + reward
            if new_total > 0:
                genre_pref = (total_reward * state[:len(rl_genre_list)] + reward * genre_vec) / new_total
            else:
                genre_pref = state[:len(rl_genre_list)]
            state[:len(rl_genre_list)] = genre_pref
            state[-3] = new_total
        state[-2] += 1
        if watch_percent is not None:
            old_avg = state[-1]
            new_count = state[-2]
            state[-1] = (old_avg * (new_count - 1) + watch_percent) / new_count

    # Save updated state
    with user_states_lock:
        user_states[user_id] = (state, datetime.now())

def build_state_from_db(user_id):
    conn = get_db()
    rows = conn.execute(
        """SELECT movie_title, reward_value, watch_percent
           FROM rl_rewards
           WHERE user_id=?
           ORDER BY timestamp ASC""",
        (user_id,)
    ).fetchall()
    conn.close()

    genre_pref = np.zeros(len(rl_genre_list), dtype=np.float32)
    total_reward = 0.0
    total_watch = 0.0
    count = 0
    for r in rows:
        movie = r['movie_title'].lower()
        reward = r['reward_value'] if r['reward_value'] else 0.0
        if reward > 0 and movie in rl_movie_features_dict:
            genre_vec = rl_movie_features_dict[movie] # multione-hot vector
            total_reward += reward
            genre_pref += genre_vec * reward
        if r['watch_percent']:
            total_watch += r['watch_percent']
        count += 1
    if total_reward > 0:
        genre_pref /= total_reward
    avg_watch = total_watch / count if count > 0 else 0.0
    state = np.concatenate([genre_pref, [total_reward, count, avg_watch]]).astype(np.float32)
    return state

def rl_recommend_online(user_id, n=12):
    if rl_agent is None:
        return []

    state = get_user_state(user_id)

    # lấy movie đã xem
    conn = get_db()
    watched = conn.execute(
        "SELECT DISTINCT movie_title FROM user_interactions WHERE user_id=?", (user_id,)
    ).fetchall()
    conn.close()

    exclude = []
    for r in watched:
        title = r['movie_title'].lower()
        if title in movie_to_idx:
            exclude.append(movie_to_idx[title])

    recs = rl_agent.recommend(state, top_k=n, exclude=exclude)
    

    print(f"State dim: {len(state)}")
    print(f"Exclude idx: {exclude[:5]}...")
    print(f"RL recs: {recs}")

    return recs

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def get_user_history(user_id, limit=20):
    conn = get_db()
    rows = conn.execute(
        """SELECT movie_title, action, rating, watch_seconds, watch_percent,
                  from_rec, timestamp
           FROM user_interactions
           WHERE user_id=?
           ORDER BY timestamp DESC
           LIMIT ?""",
        (user_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def log_interaction(user_id, movie_title, action, **kwargs):
    conn = get_db()
    conn.execute(
        """INSERT INTO user_interactions
           (user_id, movie_title, action, rating, watch_seconds, watch_percent,
            from_rec, rec_position, session_id)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (user_id, movie_title.lower(), action,
         kwargs.get('rating'),
         kwargs.get('watch_seconds'),
         kwargs.get('watch_percent'),
         int(kwargs.get('from_rec', 0)),
         kwargs.get('rec_position'),
         kwargs.get('session_id')))
    conn.commit()
    conn.close()

def has_watched_before(user_id, movie_title):
    conn = get_db()
    row = conn.execute(
        """SELECT COUNT(*) as cnt FROM user_interactions
           WHERE user_id=? AND movie_title=? AND action='watch_end'""",
        (user_id, movie_title.lower())).fetchone()
    conn.close()
    return row['cnt'] > 0

def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0]  = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')
    return my_list

def get_suggestions():
    data, _ = get_cb_data()
    return list(data['movie_title'].str.capitalize())

# ─────────────────────────────────────────────
#  Routes – Auth
# ─────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form['username'].strip()
        email    = request.form['email'].strip()
        password = request.form['password']
        if not username or not email or not password:
            return render_template('register.html', error="Vui lòng điền đầy đủ thông tin.")
        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO users (username, email, password) VALUES (?,?,?)",
                (username, email, hash_password(password)))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('register.html', error="Username hoặc Email đã tồn tại.")
        conn.close()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username'].strip()
        password = request.form['password']
        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, hash_password(password))).fetchone()
        conn.close()
        if user:
            session['user_id']   = user['id']
            session['username']  = user['username']
            return redirect(url_for('home'))
        return render_template('login.html', error="Sai username hoặc mật khẩu.")
    return render_template('login.html')

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

# ─────────────────────────────────────────────
#  Routes – Main
# ─────────────────────────────────────────────
@app.route("/")
@app.route("/home")
@login_required
def home():
    suggestions  = get_suggestions()
    user_id      = session['user_id']
    username     = session['username']

    conn = get_db()
    interaction_count = conn.execute(
        "SELECT COUNT(*) FROM user_interactions WHERE user_id=?",
        (user_id,)).fetchone()[0]
    conn.close()

    if rl_agent is not None and interaction_count >= 5:
        # RL 
        print("Using RL for recommendations")
        featured_movies = rl_recommend_online(user_id, n=12)
        if not featured_movies:
            featured_movies = get_top_rated_movies(10)
        mode = 'rl_personalised'
    else:
        # no RL
        print("Using non-RL recommendations")
        if interaction_count < 5:
            featured_movies = get_top_rated_movies(10)
            mode = 'cold_start'
            print("cold start with top-rated movies")
        else:
            featured_movies = cf_recommend(user_id, n=12)
            if not featured_movies:
                featured_movies = get_top_rated_movies(10)
            mode = 'personalised'
            print("collaborative filtering based on user interactions")

    history = get_user_history(user_id, limit=10)
    return render_template('home.html',
                           suggestions=suggestions,
                           username=username,
                           featured_movies=featured_movies,
                           mode=mode,
                           history=history)


@app.route("/similarity", methods=["POST"])
@login_required
def similarity():
    movie = request.form['name']
    log_interaction(session['user_id'], movie, 'search')

    reward = compute_reward('search')
    log_reward(session['user_id'], movie, 'search', reward)

    # =========================
    # 🔥 RL UPDATE (NEW)
    # =========================
    if rl_agent is not None:
        current_state = get_user_state(session['user_id'])

        # update state sau khi user xem phim
        update_user_state(session['user_id'], movie, reward)
        next_state = get_user_state(session['user_id'])

        movie_key = movie.lower()

        if movie_key in movie_to_idx:
            action = movie_to_idx[movie_key]  # 🔥 dùng index thay vì feature

            rl_agent.store(
                current_state,
                action,
                reward,
                next_state,
                False
            )

            rl_agent.update()  # online learning

    # =========================
    # phần còn lại giữ nguyên
    # =========================
    data, sim = get_cb_data()
    m = movie.lower()

    if m not in data['movie_title'].unique():
        return 'Sorry! try another movie name'

    conn = get_db()
    cnt = conn.execute(
        "SELECT COUNT(*) FROM user_interactions WHERE user_id=?",
        (session['user_id'],)
    ).fetchone()[0]
    conn.close()

    if cnt >= 5:
        recs = hybrid_recommend(session['user_id'], movie, n=12, alpha=0.4)
    else:
        recs = cb_recommend(movie, n=12)

    if not recs:
        return 'Sorry! try another movie name'

    return "---".join(recs)

@app.route("/recommend", methods=["POST"])
@login_required
def recommend():
    title        = request.form['title']
    cast_ids     = request.form['cast_ids']
    cast_names   = request.form['cast_names']
    cast_chars   = request.form['cast_chars']
    cast_bdays   = request.form['cast_bdays']
    cast_bios    = request.form['cast_bios']
    cast_places  = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id      = request.form['imdb_id']
    poster       = request.form['poster']
    genres       = request.form['genres']
    overview     = request.form['overview']
    vote_average = request.form['rating']
    vote_count   = request.form['vote_count']
    release_date = request.form['release_date']
    runtime      = request.form['runtime']
    status       = request.form['status']
    rec_movies   = request.form['rec_movies']
    rec_posters  = request.form['rec_posters']
    from_rec     = int(request.form.get('from_rec', 0))
    rec_position = request.form.get('rec_position', None)
    ic(title, from_rec, rec_position)
    user_id = session['user_id']
    is_rewatch = has_watched_before(user_id, title)
    ic(is_rewatch)

    # Ghi hành động view vào log và tính reward
    log_interaction(user_id, title, 'view',
                    from_rec=from_rec,
                    rec_position=int(rec_position) if rec_position else None)
    log_reward(user_id, title, 'view', compute_reward('view'),
               context={'from_rec': from_rec, 'genres': genres})

    if is_rewatch:
        # Nếu xem lại một phim đã xem trước đó, ghi log và reward cho hành động rewatch
        log_interaction(user_id, title, 'rewatch', from_rec=from_rec)
        log_reward(user_id, title, 'rewatch', compute_reward('rewatch'),
                   context={'from_rec': from_rec})

    suggestions = get_suggestions()

    rec_movies   = convert_to_list(rec_movies)
    rec_posters  = convert_to_list(rec_posters)
    cast_names   = convert_to_list(cast_names)
    cast_chars   = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays   = convert_to_list(cast_bdays)
    cast_bios    = convert_to_list(cast_bios)
    cast_places  = convert_to_list(cast_places)

    cast_ids     = cast_ids.split(',')
    cast_ids[0]  = cast_ids[0].replace("[", "")
    cast_ids[-1] = cast_ids[-1].replace("]", "")

    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')

    movie_cards  = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    casts        = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]]
                    for i in range(len(cast_profiles))}
    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i],
                                    cast_bdays[i], cast_places[i], cast_bios[i]]
                    for i in range(len(cast_places))}

    reviews_list = []
    reviews_status = []
    try:
        # Thêm User-Agent để tránh bị chặn
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        req = urllib.request.Request(f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt', headers=headers)
        sauce = urllib.request.urlopen(req, timeout=10).read()
        soup = bs.BeautifulSoup(sauce, 'lxml')
        
        # Thử nhiều selector khác nhau (cập nhật theo cấu trúc mới của IMDb)
        review_divs = soup.find_all("div", class_="text show-more__control")  # selector cũ
        if not review_divs:
            review_divs = soup.find_all("div", class_="ipc-html-content-inner-div")  # selector mới
        if not review_divs:
            review_divs = soup.select(".review-text")  # selector khác
        if not review_divs:
            review_divs = soup.find_all("div", {"data-testid": "review-content"})
            
        for tag in review_divs:
            review_text = tag.get_text(strip=True)
            if review_text:
                reviews_list.append(review_text)
                # Nếu có NLP thì phân tích cảm xúc, nếu không thì gán mặc định
                if NLP_READY:
                    vec = vectorizer_nlp.transform(np.array([review_text]))
                    pred = clf.predict(vec)
                    reviews_status.append('Good' if pred else 'Bad')
                else:
                    reviews_status.append('No analysis')  # hoặc bỏ qua
    except Exception as e:
        print(f"Lỗi khi crawl review IMDb: {e}")  # Ghi log lỗi để debug
        pass

    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}
    ic(movie_reviews)

    conn = get_db()
    existing_rating = conn.execute(
        "SELECT rating FROM user_ratings WHERE user_id=? AND movie_title=?",
        (user_id, title.lower())).fetchone()
    conn.close()
    user_rating = existing_rating['rating'] if existing_rating else None

    return render_template('recommend.html',
        title=title, poster=poster, overview=overview,
        vote_average=vote_average, vote_count=vote_count,
        release_date=release_date, runtime=runtime, status=status,
        genres=genres, movie_cards=movie_cards, reviews=movie_reviews,
        casts=casts, cast_details=cast_details,
        username=session.get('username', ''),
        user_rating=user_rating,
        is_rewatch=is_rewatch,
        suggestions=suggestions)

# ─────────────────────────────────────────────
#  Routes – Tracking API (called by JS)
# ─────────────────────────────────────────────
@app.route("/track", methods=["POST"])
@login_required
def track():
    data         = request.get_json()
    movie        = data.get('movie_title', '').strip()
    action       = data.get('action', '')
    watch_secs   = data.get('watch_seconds')
    watch_pct    = data.get('watch_percent')
    from_rec     = int(data.get('from_rec', 0))
    rec_position = data.get('rec_position')
    session_id   = data.get('session_id')

    if not movie or not action:
        return jsonify({'status': 'error', 'message': 'Missing fields'}), 400

    user_id = session['user_id']

    if action == 'watch_start':
        if has_watched_before(user_id, movie):
            log_interaction(user_id, movie, 'rewatch', from_rec=from_rec,
                            session_id=session_id)
            log_reward(user_id, movie, 'rewatch', compute_reward('rewatch'),
                       context={'from_rec': from_rec})

    log_interaction(user_id, movie, action,
                    watch_seconds=watch_secs,
                    watch_percent=watch_pct,
                    from_rec=from_rec,
                    rec_position=rec_position,
                    session_id=session_id)

    extra  = {'watch_percent': watch_pct, 'rating': data.get('rating'), 'from_rec': from_rec}
    reward = compute_reward(action, extra)

    if from_rec and action == 'view':
        reward += compute_reward('rec_converted')
        log_reward(user_id, movie, 'rec_converted', compute_reward('rec_converted'),
                watch_percent=watch_pct,  # ← thêm
                context={'rec_position': rec_position})

    if reward != 0.0:
        log_reward(user_id, movie, action, reward,
               watch_percent=watch_pct,  # ← thêm dòng này
               context={'watch_percent': watch_pct, 'from_rec': from_rec,
                        'rec_position': rec_position})


    # ── Online RL update ─────────────────────────────────────────────
    if rl_agent is not None:
        current_state = get_user_state(user_id)
        update_user_state(user_id, movie, reward, watch_pct)
        next_state = get_user_state(user_id)
        if movie.lower() in rl_movie_features_dict:
            action_feat = rl_movie_features_dict[movie.lower()]
            rl_agent.store_transition(current_state, action_feat, reward, next_state, False)
            # Trigger a training step (non-blocking can be done in thread, but here it's synchronous)
            loss = rl_agent.update()
            # optional: log loss for debugging

    return jsonify({'status': 'ok', 'reward': reward})

# ─────────────────────────────────────────────
#  Routes – Rating API
# ─────────────────────────────────────────────
@app.route("/rate_movie", methods=["POST"])
@login_required
def rate_movie():
    data        = request.get_json()
    movie_title = data.get('movie_title', '').lower()
    rating      = float(data.get('rating', 0))

    if not movie_title or not (1 <= rating <= 10):
        return jsonify({'status': 'error', 'message': 'Invalid input'}), 400

    user_id = session['user_id']

    conn = get_db()
    conn.execute(
        """INSERT INTO user_ratings (user_id, movie_title, rating)
           VALUES (?,?,?)
           ON CONFLICT(user_id, movie_title) DO UPDATE SET rating=excluded.rating,
           timestamp=CURRENT_TIMESTAMP""",
        (user_id, movie_title, rating))
    conn.commit()
    conn.close()

    log_interaction(user_id, movie_title, 'rate', rating=rating)
    reward = compute_reward('rate', {'rating': rating})
    log_reward(user_id, movie_title, 'rate', reward, context={'rating': rating})

    # RL update
    if rl_agent is not None:
        current_state = get_user_state(user_id)
        update_user_state(user_id, movie_title, reward)
        next_state = get_user_state(user_id)
        if movie_title in rl_movie_features_dict:
            action_feat = rl_movie_features_dict[movie_title]
            rl_agent.store_transition(current_state, action_feat, reward, next_state, False)
            rl_agent.update()

    return jsonify({'status': 'ok', 'reward': reward,
                    'message': f'Đã đánh giá {rating}/10'})

# ─────────────────────────────────────────────
#  Routes – RL Data Export (for training)
# ─────────────────────────────────────────────
@app.route("/rl/rewards")
@login_required
def rl_rewards_export():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM rl_rewards ORDER BY timestamp ASC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/rl/user_rewards/<int:uid>")
@login_required
def rl_user_rewards(uid):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM rl_rewards WHERE user_id=? ORDER BY timestamp ASC",
        (uid,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

# ─────────────────────────────────────────────
#  Routes – User Profile
# ─────────────────────────────────────────────
@app.route("/profile")
@login_required
def profile():
    user_id = session['user_id']
    history = get_user_history(user_id, limit=50)

    conn = get_db()
    ratings = conn.execute(
        "SELECT movie_title, rating, timestamp FROM user_ratings WHERE user_id=? ORDER BY timestamp DESC",
        (user_id,)).fetchall()

    rl_stats = conn.execute(
        """SELECT action, COUNT(*) as count, SUM(reward_value) as total_reward,
                  AVG(reward_value) as avg_reward
           FROM rl_rewards WHERE user_id=?
           GROUP BY action ORDER BY total_reward DESC""",
        (user_id,)).fetchall()

    watch_total = conn.execute(
        """SELECT COALESCE(SUM(watch_seconds), 0) as total
           FROM user_interactions
           WHERE user_id=? AND action='watch_end'""",
        (user_id,)).fetchone()['total']

    conn.close()

    return render_template('profile.html',
                           username=session['username'],
                           history=history,
                           ratings=[dict(r) for r in ratings],
                           rl_stats=[dict(r) for r in rl_stats],
                           watch_total_minutes=watch_total // 60)

# ─────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # Ensure RL agent is loaded at startup
    if rl_agent is None:
        get_cb_data()  # will initialize agent
    app.run(debug=True)