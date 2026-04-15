// Constants
const API_BASE_URL = 'http://localhost:3000/api';

// DOM Elements
const currentDateEl = document.getElementById('current-date');
const studentGrid = document.getElementById('student-grid');
const loadingEl = document.getElementById('loading');
const filterButtons = document.querySelectorAll('.filter-buttons button');
const cardTemplate = document.getElementById('student-card-template');

// Chart instance
let summaryChart = null;

// Initialize the dashboard
async function initDashboard() {
    updateCurrentDate();
    await loadStudentData();
    setupFilterButtons();
    setInterval(refreshData, 30000); // Refresh every 30 seconds
}

// Update the current date display
function updateCurrentDate() {
    const now = new Date();
    currentDateEl.textContent = now.toLocaleDateString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
}

// Load student data from the API
async function loadStudentData() {
    try {
        loadingEl.style.display = 'block';
        studentGrid.style.display = 'none';

        const [studentsRes, attendanceRes] = await Promise.all([
            fetch(`${API_BASE_URL}/students`),
            fetch(`${API_BASE_URL}/attendance/today`)
        ]);

        const students = await studentsRes.json();
        const attendance = await attendanceRes.json();

        // Process and display the data
        displayStudentCards(students, attendance);
        updateSummaryChart(students.length, attendance);

        loadingEl.style.display = 'none';
        studentGrid.style.display = 'grid';
    } catch (error) {
        console.error('Error loading data:', error);
        loadingEl.textContent = 'Error loading data. Please try again later.';
    }
}

// Create and display student cards
function displayStudentCards(students, attendance) {
    studentGrid.innerHTML = '';
    const attendanceMap = new Map(attendance.map(record => [record.student_name, record]));

    students.forEach(student => {
        const card = createStudentCard(student, attendanceMap.get(student.name));
        studentGrid.appendChild(card);
    });
}

// Create a single student card
function createStudentCard(student, attendanceRecord) {
    const card = cardTemplate.content.cloneNode(true);
    const isPresent = !!attendanceRecord;

    // Fill in the card details
    card.querySelector('.student-name').textContent = student.name;
    card.querySelector('.status').textContent = isPresent ? 'Present' : 'Absent';
    card.querySelector('.status-indicator').classList.add(isPresent ? 'present' : 'absent');
    
    if (isPresent && attendanceRecord.timestamp) {
        const time = new Date(attendanceRecord.timestamp).toLocaleTimeString();
        card.querySelector('.time').textContent = `Since ${time}`;
    }

    // Set up the "View History" link
    const historyLink = card.querySelector('.view-history');
    historyLink.href = `/student.html?name=${encodeURIComponent(student.name)}`;

    // Add data attributes for filtering
    const cardElement = card.querySelector('.student-card');
    cardElement.dataset.status = isPresent ? 'present' : 'absent';

    return card;
}

// Update the summary pie chart
function updateSummaryChart(totalStudents, attendance) {
    const presentCount = new Set(attendance.map(record => record.student_name)).size;
    const absentCount = totalStudents - presentCount;

    if (summaryChart) {
        summaryChart.destroy();
    }

    const ctx = document.getElementById('summary-chart').getContext('2d');
    summaryChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Present', 'Absent'],
            datasets: [{
                data: [presentCount, absentCount],
                backgroundColor: ['#27ae60', '#e74c3c'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// Set up filter button functionality
function setupFilterButtons() {
    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Update active button
            filterButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            // Apply filter
            const filter = button.dataset.filter;
            const cards = studentGrid.querySelectorAll('.student-card');

            cards.forEach(card => {
                if (filter === 'all' || card.dataset.status === filter) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    });
}

// Refresh data periodically
async function refreshData() {
    const attendanceRes = await fetch(`${API_BASE_URL}/attendance/today`);
    const attendance = await attendanceRes.json();
    const cards = studentGrid.querySelectorAll('.student-card');

    cards.forEach(card => {
        const studentName = card.querySelector('.student-name').textContent;
        const attendanceRecord = attendance.find(record => record.student_name === studentName);
        const isPresent = !!attendanceRecord;

        // Update status
        card.querySelector('.status').textContent = isPresent ? 'Present' : 'Absent';
        card.querySelector('.status-indicator').className = `status-indicator ${isPresent ? 'present' : 'absent'}`;
        card.dataset.status = isPresent ? 'present' : 'absent';

        // Update time if present
        if (isPresent && attendanceRecord.timestamp) {
            const time = new Date(attendanceRecord.timestamp).toLocaleTimeString();
            card.querySelector('.time').textContent = `Since ${time}`;
        } else {
            card.querySelector('.time').textContent = '';
        }
    });

    // Update summary chart
    updateSummaryChart(cards.length, attendance);
}

// Start the dashboard
document.addEventListener('DOMContentLoaded', initDashboard);
