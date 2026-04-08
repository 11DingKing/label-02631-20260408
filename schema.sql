-- VGG16 动物图像分类项目 - 数据库 Schema
-- 用于记录训练实验元数据（可选，项目核心功能不依赖数据库）

CREATE TABLE IF NOT EXISTS experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tag             VARCHAR(64)   NOT NULL,
    created_at      DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    config_json     TEXT          NOT NULL,
    num_epochs      INTEGER       NOT NULL,
    learning_rate   REAL          NOT NULL,
    optimizer       VARCHAR(16)   NOT NULL,
    best_val_acc    REAL,
    test_accuracy   REAL,
    test_precision  REAL,
    test_recall     REAL,
    test_f1         REAL,
    model_path      VARCHAR(256),
    output_dir      VARCHAR(256),
    status          VARCHAR(16)   NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS training_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER       NOT NULL,
    epoch           INTEGER       NOT NULL,
    train_loss      REAL          NOT NULL,
    train_acc       REAL          NOT NULL,
    val_loss        REAL          NOT NULL,
    val_acc         REAL          NOT NULL,
    learning_rate   REAL          NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS dataset_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER       NOT NULL,
    split           VARCHAR(16)   NOT NULL,
    class_name      VARCHAR(32)   NOT NULL,
    image_count     INTEGER       NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
