-- Experiment tracking database for KaggleMate.
-- One database per competition: competitions/<slug>/experiments.db

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL,
    competition_slug TEXT NOT NULL,
    task_type TEXT,             -- binary_classification | multiclass_classification | regression | time_series
    target_column TEXT,         -- target column name
    id_column TEXT,             -- id column name
    model_name TEXT NOT NULL DEFAULT 'Unknown',
    cv_score REAL,
    cv_std REAL,
    lb_score REAL,
    metric TEXT DEFAULT 'unknown',
    cv_folds INTEGER DEFAULT 5,
    cv_strategy TEXT,           -- e.g. StratifiedKFold
    features TEXT,              -- JSON array of feature column names
    params TEXT,                -- JSON object of model hyperparameters
    feature_importance TEXT,    -- JSON array of [name, importance] pairs
    fold_scores TEXT,           -- JSON array of per-fold scores
    oof_path TEXT,              -- path to out-of-fold predictions CSV
    fold_scores_path TEXT,      -- path to per-fold scores JSON
    config_path TEXT,           -- path to experiment_config.json
    strategy_validation_report_path TEXT,
    submission_validation_report_path TEXT,
    benchmark_result_path TEXT,
    runtime_seconds REAL,       -- script execution time
    script_hash TEXT,           -- sha256 of training script
    submission_hash TEXT,       -- sha256 of submission.csv
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
