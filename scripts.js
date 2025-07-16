// Toggle debug block
function toggleDebug() {
    var dbg = document.getElementById('debug-block');
    if (dbg.style.display === 'none') {
        dbg.style.display = 'block';
    } else {
        dbg.style.display = 'none';
    }
}
// Smooth scroll to visualizations
function scrollToViz() {
    var viz = document.getElementById('visualizations');
    if (viz) viz.scrollIntoView({ behavior: 'smooth' });
}
// Placeholder for future chart integration
// function renderCharts(data) { /* ... */ } 