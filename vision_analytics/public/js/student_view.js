// Constants
const API_BASE_URL = 'http://localhost:3000/api';

// Charts
let attendanceChart = null;
let emotionDailyChart = null;
let emotionHistoryChart = null;

// Initialize the student view
async function initStudentView() {
    const urlParams = new URLSearchParams(window.location.search);
    const studentName = urlParams.get('name');

    if (!studentName) {
        window.location.href = '/';
        return;
    }

    document.getElementById('student-name').textContent = studentName;
    await loadStudentData(studentName);
    setInterval(() => loadStudentData(studentName), 30000); // Refresh every 30 seconds
}

// Load all data for a student
async function loadStudentData(studentName) {
    try {
        const [attendanceRes, emotionDailyRes, emotionHistoryRes] = await Promise.all([
            fetch(`${API_BASE_URL}/attendance/history/${encodeURIComponent(studentName)}`),
            fetch(`${API_BASE_URL}/emotion/daily/${encodeURIComponent(studentName)}`),
            fetch(`${API_BASE_URL}/emotion/history/${encodeURIComponent(studentName)}`)
        ]);

        const attendanceData = await attendanceRes.json();
        const emotionDailyData = await emotionDailyRes.json();
        const emotionHistoryData = await emotionHistoryRes.json();

        updateAttendanceChart(attendanceData);
        updateEmotionChart(emotionDailyData, 'daily');
        updateEmotionChart(emotionHistoryData, 'history');
    } catch (error) {
        console.error('Error loading student data:', error);
    }
}

// Update the attendance history chart
function updateAttendanceChart(data) {
    const dates = data.map(record => {
        const date = new Date(record.date);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });
    const lecturesAttended = data.map(record => record.lectures_attended);

    if (attendanceChart) {
        attendanceChart.destroy();
    }

    const ctx = document.getElementById('attendance-chart').getContext('2d');
    attendanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: dates,
            datasets: [{
                label: 'Lectures Attended',
                data: lecturesAttended,
                backgroundColor: '#3498db',
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 8,
                    ticks: {
                        stepSize: 1
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Update emotion charts (daily or history)
function updateEmotionChart(data, type) {
    const chartId = `emotion-${type}-chart`;
    const chartInstance = type === 'daily' ? emotionDailyChart : emotionHistoryChart;
    const emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
    const emotionData = emotions.map(emotion => (data[emotion] * 100).toFixed(1));

    if (chartInstance) {
        chartInstance.destroy();
    }

    const ctx = document.getElementById(chartId).getContext('2d');
    const newChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: emotions.map(e => e.charAt(0).toUpperCase() + e.slice(1)),
            datasets: [{
                data: emotionData,
                backgroundColor: [
                    '#e74c3c', // angry
                    '#8e44ad', // disgust
                    '#f39c12', // fear
                    '#27ae60', // happy
                    '#2c3e50', // sad
                    '#e67e22', // surprise
                    '#95a5a6'  // neutral
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        boxWidth: 15
                    }
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            return `${label}: ${value}%`;
                        }
                    }
                }
            }
        }
    });

    if (type === 'daily') {
        emotionDailyChart = newChart;
    } else {
        emotionHistoryChart = newChart;
    }
}

// Start the student view
document.addEventListener('DOMContentLoaded', initStudentView);
