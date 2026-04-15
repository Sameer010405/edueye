const express = require('express');
const cors = require('cors');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const morgan = require('morgan');

// Initialize express app
const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(morgan('dev')); // For logging requests
app.use(express.static('public')); // Serve static files from public directory

// Database setup
const db = new sqlite3.Database(path.join(__dirname, 'data/attendance_system.db'), (err) => {
    if (err) {
        console.error('Error connecting to database:', err);
    } else {
        console.log('Connected to SQLite database');
        initDatabase();
    }
});

// Initialize database tables
function initDatabase() {
    db.serialize(() => {
        // Students table
        db.run(`CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )`);

        // Attendance records table
        db.run(`CREATE TABLE IF NOT EXISTS attendance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            lecture_number INTEGER NOT NULL,
            date TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (student_name) REFERENCES students(name)
        )`);

        // Emotion records table
        db.run(`CREATE TABLE IF NOT EXISTS emotion_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            angry REAL,
            disgust REAL,
            fear REAL,
            happy REAL,
            sad REAL,
            surprise REAL,
            neutral REAL,
            FOREIGN KEY (student_name) REFERENCES students(name)
        )`);
    });
}

// API Endpoints

// Get list of all students
app.get('/api/students', (req, res) => {
    db.all('SELECT name FROM students ORDER BY name', [], (err, rows) => {
        if (err) {
            res.status(500).json({ error: err.message });
            return;
        }
        res.json(rows);
    });
});

// Get today's attendance records
app.get('/api/attendance/today', (req, res) => {
    const today = new Date().toISOString().split('T')[0];
    db.all(
        `SELECT student_name, lecture_number, timestamp 
         FROM attendance_records 
         WHERE date = ? 
         ORDER BY timestamp DESC`,
        [today],
        (err, rows) => {
            if (err) {
                res.status(500).json({ error: err.message });
                return;
            }
            res.json(rows);
        }
    );
});

// Get attendance history for a student
app.get('/api/attendance/history/:name', (req, res) => {
    db.all(
        `SELECT date, COUNT(*) as lectures_attended
         FROM attendance_records 
         WHERE student_name = ?
         GROUP BY date
         ORDER BY date DESC
         LIMIT 30`,
        [req.params.name],
        (err, rows) => {
            if (err) {
                res.status(500).json({ error: err.message });
                return;
            }
            res.json(rows);
        }
    );
});

// Get daily emotion data for a student
app.get('/api/emotion/daily/:name', (req, res) => {
    const today = new Date().toISOString().split('T')[0];
    db.all(
        `SELECT AVG(angry) as angry, AVG(disgust) as disgust,
                AVG(fear) as fear, AVG(happy) as happy,
                AVG(sad) as sad, AVG(surprise) as surprise,
                AVG(neutral) as neutral
         FROM emotion_records 
         WHERE student_name = ? 
         AND date(timestamp) = ?`,
        [req.params.name, today],
        (err, rows) => {
            if (err) {
                res.status(500).json({ error: err.message });
                return;
            }
            res.json(rows[0]);
        }
    );
});

// Get historical emotion data for a student
app.get('/api/emotion/history/:name', (req, res) => {
    db.all(
        `SELECT AVG(angry) as angry, AVG(disgust) as disgust,
                AVG(fear) as fear, AVG(happy) as happy,
                AVG(sad) as sad, AVG(surprise) as surprise,
                AVG(neutral) as neutral
         FROM emotion_records 
         WHERE student_name = ?`,
        [req.params.name],
        (err, rows) => {
            if (err) {
                res.status(500).json({ error: err.message });
                return;
            }
            res.json(rows[0]);
        }
    );
});

// Record attendance (POST endpoint for Python script)
app.post('/api/attendance', (req, res) => {
    const { studentName, lecture, timestamp } = req.body;
    const date = new Date(timestamp).toISOString().split('T')[0];

    // First ensure student exists
    db.run('INSERT OR IGNORE INTO students (name) VALUES (?)', [studentName], (err) => {
        if (err) {
            res.status(500).json({ error: err.message });
            return;
        }

        // Then record attendance
        db.run(
            `INSERT INTO attendance_records (student_name, lecture_number, date, timestamp)
             VALUES (?, ?, ?, ?)`,
            [studentName, lecture, date, timestamp],
            (err) => {
                if (err) {
                    res.status(500).json({ error: err.message });
                    return;
                }
                res.json({ message: 'Attendance recorded successfully' });
            }
        );
    });
});

// Record emotion data (POST endpoint for Python script)
app.post('/api/emotion', (req, res) => {
    const { studentName, timestamp, emotions } = req.body;

    // First ensure student exists
    db.run('INSERT OR IGNORE INTO students (name) VALUES (?)', [studentName], (err) => {
        if (err) {
            res.status(500).json({ error: err.message });
            return;
        }

        // Then record emotion data
        db.run(
            `INSERT INTO emotion_records (
                student_name, timestamp, angry, disgust, fear,
                happy, sad, surprise, neutral
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
            [
                studentName, timestamp,
                emotions.angry, emotions.disgust, emotions.fear,
                emotions.happy, emotions.sad, emotions.surprise,
                emotions.neutral
            ],
            (err) => {
                if (err) {
                    res.status(500).json({ error: err.message });
                    return;
                }
                res.json({ message: 'Emotion data recorded successfully' });
            }
        );
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
