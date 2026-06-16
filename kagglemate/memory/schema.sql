-- Experiment tracking database for KaggleMate.
-- One database per competition: competitions/<slug>/experiments.db

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL,
    competition_slug TEXT NOT NULL,
    model_name TEXT NOT NULL DEFAULT 'Unknown',
    cv_score REAL,
    cv_std REAL,
    lb_score REAL,
    metric TEXT DEFAULT 'unknown',
    cv_folds INTEGER DEFAULT 5,
    features TEXT,              -- JSON array of feature column names
    params TEXT,                -- JSON object of model hyperparameters
    feature_importance TEXT,    -- JSON array of [name, importance] pairs
    fold_scores TEXT,           -- JSON array of per-fold scores
    submission_path TEXT,
    script_path TEXT,
    report_path TEXT,
    notes TEXT,
    status TEXT DEFAULT 'completed',  -- completed | failed | running
    error_message TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    lb_updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_exp_competition ON experiments(competition_slug);
CREATE INDEX IF NOT EXISTS idx_exp_created ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_exp_cv_score ON experiments(cv_score);
